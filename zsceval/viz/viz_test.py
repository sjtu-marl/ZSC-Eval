import os
import sys
import numpy as np
import pygame
import random
import time
from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args, OLD_LAYOUTS

from zsceval.envs.overcooked.Overcooked_Env import Overcooked
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
import yaml
import pickle, pathlib
import torch
path = "../policy_pool"
os.environ["POLICY_POOL"] = path

from zsceval.algorithms.population.policy_pool import add_path_prefix
from zsceval.runner.shared.base_runner import make_trainer_policy_cls

from zsceval.viz.gradcam import GradCAM
import cv2
import torch.nn as nn
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from collections import deque
from topdown_posterior_fusion import AttentionFuser

def parse_args(args, parser):
    parser = get_overcooked_args(parser)
    parser.add_argument(
        "--use_phi",
        default=False,
        action="store_true",
        help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.",
    )

    parser.add_argument("--test_policy_name", type=str, default="fcp", choices=["fcp", "mep", "traj", "hsp", "sp",
                                                                                "e3t", "cole"])
    parser.add_argument("--model_seed", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--epsilon", type=float, default=0.0, help="stochastic eval epsilon")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--is_cam", type=str, default="False", choices=["ArgMax", "Whole", "False"], help="Whether to use CAM")
    parser.add_argument("--cam_alpha", type=float, default=0.8)
    parser.add_argument("--cam_layers", type=str, default="2", help="'0, 1 ,2' or 'all'")
    # parse_args 안
    parser.add_argument("--win_path_fix", action="store_true",
                        help="Windows에서 PosixPath 들어간 pickle을 안전하게 로드")


    all_args = parser.parse_args(args)
    if all_args.layout_name in OLD_LAYOUTS:
        all_args.old_dynamics = True
    else:
        all_args.old_dynamics = False
    return all_args


class WorkMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, item):
        self.memory.append(item)        
        
    def __len__(self):
        return len(self.memory)
    
    
def nextPosition(position, action: int):
    if action == 0:  # North
        return (position[0] - 1, position[1])
    elif action == 1:  # South
        return (position[0] + 1, position[1])
    elif action == 2:  # East
        return (position[0], position[1] + 1)
    elif action == 3:  # West
        return (position[0], position[1] - 1)
    else:  # Stay
        return position

class _PathFixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 윈도우에서 리눅스/맥의 PosixPath를 WindowsPath로 매핑
        if sys.platform.startswith("win") and module == "pathlib" and name == "PosixPath":
            return pathlib.WindowsPath
        return super().find_class(module, name)

def load_pickle_with_path_fix(path):
    with open(path, "rb") as f:
        return _PathFixUnpickler(f).load()


class EvalPolicy_Play:

    def __init__(self, population_yaml_path, layout_name, test_policy_name, deterministic=True, epsilon=0.5, win_path_fix=False):
        self.population_yaml_path = population_yaml_path
        self.layout_name = layout_name
        self.test_policy_name = test_policy_name
        self.deterministic = deterministic
        self.epsilon = epsilon
        self.population_config = yaml.load(open(self.population_yaml_path, encoding="utf-8"), yaml.Loader)

        policy_config_path = os.path.join("../policy_pool",
                                          self.population_config[self.test_policy_name]["policy_config_path"])
        # policy_config = list(pickle.load(open(policy_config_path, "rb")))
        try:
            if win_path_fix and sys.platform.startswith("win"):
                policy_config = list(load_pickle_with_path_fix(policy_config_path))
            else:
                with open(policy_config_path, "rb") as f:
                    policy_config = list(pickle.load(f))
        except NotImplementedError:
            # 윈도우에서 PosixPath로 터질 때 자동 폴백
            policy_config = list(load_pickle_with_path_fix(policy_config_path))
        
        
        self.policy_args = policy_config[0]
        _, policy_cls = make_trainer_policy_cls(self.policy_args.algorithm_name)  # ex) rmappo
        model_path = add_path_prefix("../policy_pool", self.population_config[self.test_policy_name]["model_path"])
        self.policy = policy_cls(*policy_config, device=torch.device("cpu"))
        self.policy.load_checkpoint(model_path)

    def init_mask_rnn_state(self):
        masks = np.ones((1, 1), dtype=np.float32)
        rnn_states = np.zeros((self.policy_args.recurrent_N, self.policy_args.hidden_size), dtype=np.float32)

        return masks, rnn_states

    def step(self, obs, masks, rnn_states, available_actions, deterministic=False):
        action, actions_prob, rnn_states = self.policy.act(obs, rnn_states, masks, available_actions=available_actions,
                                                           deterministic=deterministic, action_probs=True)
        return action, actions_prob, rnn_states

    @torch.no_grad()
    def get_action(self, obs, available_actions, masks, rnn_states):
        self.policy.prep_rollout()
        epsilon = random.random()
        if not self.deterministic or epsilon < self.epsilon:
            return self.step(obs, masks,
                             rnn_states,
                             available_actions,
                             deterministic=False)
        else:
            return self.step(obs, masks,
                             rnn_states,
                             available_actions,
                             deterministic=True)

    def init_cam(self, cam_layers: str):

        all_conv_layers = [module for module in self.policy.actor.base.cnn if isinstance(module, nn.Conv2d)]

        target_layers = []
        if cam_layers.lower() == "all":
            target_layers = all_conv_layers
        else:
            indices = [int(x.strip()) for x in cam_layers.split(",") if x.strip() != ""]
            for idx in indices:
                if 0 <= idx < len(all_conv_layers):
                    target_layers.append(all_conv_layers[idx])

        cam = GradCAM(model=self.policy.actor, target_layer=target_layers)

        return cam


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.layout_name in ["random0", "random0_medium", "random1", "random3", "small_corridor", "unident_s"]:
        env = Overcooked(all_args, run_dir=None)
    else:
        env = Overcooked_new(all_args, run_dir=None)

    population_yaml_path = os.path.join("./config", all_args.layout_name + "_benchmark.yml")
    test_policy_name = all_args.test_policy_name + str(all_args.model_seed)
    agent0_play = EvalPolicy_Play(population_yaml_path, all_args.layout_name, test_policy_name=test_policy_name, win_path_fix=all_args.win_path_fix)
    masks, rnn_states = agent0_play.init_mask_rnn_state()
    
    if all_args.is_cam:
        cam = agent0_play.init_cam(all_args.cam_layers)

    both_agents_ob, share_obs, available_actions = env.reset()

    start_time = time.time()
    clock = pygame.time.Clock()
    epi_done = False
    human_action_queue = deque(maxlen=32)
    trail = deque(maxlen=5)     # 최근 10 프레임 추적

    agent_attention_fuser = fuser = AttentionFuser(
        shape=(8,5),          # 들어오는 맵 크기와 일치
        fusion="log",         # 또는 "dirichlet"
        sigma=1.8,            # prior 확산(0.7~1.5 튜닝)
        ior_sigma=1.0,        # IOR 범위
        ior_strength=0.05,    # IOR 강도(0.05~0.15)
        ior_k=2,              # 최근 K개 fixation만 IOR
        momentum=True,        # 시선 관성 사용
        momentum_scale=1,     # 1칸 드리프트
        eta_min=0.35, eta_max=0.75,  # prior 가중치 범위
        # --- VSTM(단기기억) ---
        stm_capacity=4,       # 3~4 권장
        stm_tau=5.0,          # 3~8 프레임 감쇠
        stm_sigma=1.2,        # 공간 퍼짐
        eta_stm=0.2           # STM 부스트 비중(0.1~0.3)
    )
    
    map_size = (1212, 758)
    target_H = 960
    pad_top = target_H - 758
    
    try:
        image = env.play_render()
        screen = pygame.display.set_mode((image.shape[1], image.shape[0]))
        screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[..., ::-1], 1))), (0, 0))
        pygame.display.flip()

        while not epi_done:
            
            clock.tick(6.67)
            # enqueue keydown events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        human_action_queue.append(Direction.NORTH)
                    elif event.key == pygame.K_DOWN:
                        human_action_queue.append(Direction.SOUTH)
                    elif event.key == pygame.K_LEFT:
                        human_action_queue.append(Direction.WEST)
                    elif event.key == pygame.K_RIGHT:
                        human_action_queue.append(Direction.EAST)
                    elif event.key == pygame.K_SPACE:
                        human_action_queue.append(Action.INTERACT)
                    

            a0, a0_prob, rnn_states = agent0_play.get_action(np.expand_dims(both_agents_ob[0], axis=0),
                                                    available_actions, masks,
                                                    rnn_states)
            
            a1_action = human_action_queue.popleft() if human_action_queue else Action.STAY
            a1 = Action.ACTION_TO_INDEX[a1_action]

            joint_action = np.array([[int(a0)], [int(a1)]])

            if all_args.is_cam:
                cam_heatmap = cam(np.expand_dims(both_agents_ob[0], axis=0),
                                  available_actions, rnn_states, masks, target_action=int(a0))
                
                A_t, fix_rc, prior_pi, eta_used = agent_attention_fuser.step(cam_heatmap, hit=None, use_adaptive_eta=True)
            both_agents_ob, share_obs, reward, done, info, available_actions = env.step(joint_action)
            agent_position = env.base_env.state.players[0].position
            next_pos = nextPosition(agent_position, a0)

            epi_done = done[0]

            # render
            image = env.play_render(action_probs=a0_prob)
            if all_args.is_cam == "ArgMax":
                # filter max heatmap
                
                max_idx = np.argmax(cam_heatmap)
                max_row, max_col = np.unravel_index(max_idx, cam_heatmap.shape)
                cam_filtered = np.zeros_like(cam_heatmap)
                cam_filtered[max_row, max_col] = cam_heatmap[max_row, max_col]

                cam_resized = cv2.resize(cam_filtered, (1212, 758), interpolation=cv2.INTER_LINEAR)
                cam_resized = np.pad(cam_resized, ((pad_top, 0), (0, 0)), mode='constant', constant_values=0)
                
                heat_u8 = (cam_resized * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)[:, :, ::-1].astype(np.float32)
                
                # smoothing
                cam_soft = cv2.GaussianBlur(cam_resized.astype(np.float32), (0, 0), sigmaX=3, sigmaY=3)
                cam_soft = np.power(np.clip(cam_soft, 0.0, 1.0), 0.8)
                alpha = (cam_soft * all_args.cam_alpha)[..., None]  # (H, W, 1)

                img_f = image.astype(np.float32)
                blended = img_f * (1.0 - alpha) + heatmap_color * alpha
                image = blended.clip(0, 255).astype(np.uint8)
                
                Hh, Wh = cam_heatmap.shape[:2]          # 예: 8x5
                Hi, Wi = image.shape[:2]               # 원본 이미지 크기
                
                c, r = next_pos

                cx = int(c*151.5) + 76         
                cy = int(r*151.5)+pad_top + 76               
                
                trail.append((cx, cy))
                trail_mask = np.zeros((Hi, Wi), dtype=np.float32)
                
                # (A) 연속 선 + 가우시안 블러 (연속감↑)
                if len(trail) >= 2:
                    pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
                    # 먼저 얇은 폴리라인으로 1.0 intensity를 그린 뒤
                    cv2.polylines(trail_mask, [pts], isClosed=False, color=1.0,
                                thickness=2, lineType=cv2.LINE_AA)
                    # 부드럽게 퍼지게 블러
                    trail_mask = cv2.GaussianBlur(trail_mask, (0, 0), sigmaX=2, sigmaY=2)

                # (B) 페이딩 점(원) 추가 (최근 점은 진하게, 오래된 점은 옅게)
                for i, (tx, ty) in enumerate(reversed(trail)):
                    a = 0.35 * (0.88 ** i)
                    if a < 0.02:
                        break
                    r = max(2, int(2 * 0.9))  # 점 반지름
                    cv2.circle(trail_mask, (tx, ty), 50, a, thickness=-1, lineType=cv2.LINE_AA)

                # 3) 트레일 색상으로 수동 블렌딩 (RGB)
                imagef = image.astype(np.float32)
                imagef = imagef * (1.0 - trail_mask[..., None]) + np.array([255, 200, 80], dtype=np.float32) * trail_mask[..., None]
                image = imagef.clip(0, 255).astype(np.uint8)
                
                
            elif all_args.is_cam == "Whole":
                # filter max heatmap
                
                cam_resized = cv2.resize(A_t, (1212, 758), interpolation=cv2.INTER_LINEAR)
                cam_resized = np.pad(cam_resized, ((pad_top, 0), (0, 0)), mode='constant', constant_values=0)

                # cam_resized = cv2.resize(A_t, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                heat_u8 = (cam_resized * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)[:, :, ::-1].astype(np.float32)

                # smoothing
                cam_soft = cv2.GaussianBlur(cam_resized.astype(np.float32), (0, 0), sigmaX=3, sigmaY=3)
                cam_soft = np.power(np.clip(cam_soft, 0.0, 1.0), 0.8)
                alpha = (cam_soft * all_args.cam_alpha)[..., None]  # (H, W, 1)

                img_f = image.astype(np.float32)
                blended = img_f * (1.0 - alpha) + heatmap_color * alpha
                image = blended.clip(0, 255).astype(np.uint8)
                
                Hh, Wh = cam_heatmap.shape[:2]          # 예: 8x5
                Hi, Wi = image.shape[:2]               # 원본 이미지 크기
                c, r = fix_rc
                # c, r = next_pos
            
                cx = int(c*151.5) + 76         
                cy = int(r*151.5)+pad_top + 76               
                
                trail.append((cx, cy))
                trail_mask = np.zeros((Hi, Wi), dtype=np.float32)
                
                # (A) 연속 선 + 가우시안 블러 (연속감↑)
                if len(trail) >= 2:
                    pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
                    # 먼저 얇은 폴리라인으로 1.0 intensity를 그린 뒤
                    cv2.polylines(trail_mask, [pts], isClosed=False, color=1.0,
                                thickness=2, lineType=cv2.LINE_AA)
                    # 부드럽게 퍼지게 블러
                    trail_mask = cv2.GaussianBlur(trail_mask, (0, 0), sigmaX=2, sigmaY=2)

                # (B) 페이딩 점(원) 추가 (최근 점은 진하게, 오래된 점은 옅게)
                for i, (tx, ty) in enumerate(reversed(trail)):
                    a = 0.35 * (0.88 ** i)
                    if a < 0.02:
                        break
                    r = max(2, int(2 * 0.9))  # 점 반지름
                    cv2.circle(trail_mask, (tx, ty), 50, a, thickness=-1, lineType=cv2.LINE_AA)

                # 3) 트레일 색상으로 수동 블렌딩 (RGB)
                imagef = image.astype(np.float32)
                imagef = imagef * (1.0 - trail_mask[..., None]) + np.array([255, 200, 80], dtype=np.float32) * trail_mask[..., None]
                image = imagef.clip(0, 255).astype(np.uint8)


            elif all_args.is_cam == "False":
                pass

            screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[..., ::-1], 1))), (0, 0))
            pygame.display.flip()

            end_time = time.time()
            game_time = end_time - start_time

        print('finish_time : ', game_time)

    finally:
        cam.remove_hooks()
        pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1:])
