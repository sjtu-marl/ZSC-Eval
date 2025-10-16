import os
import sys
import time
import numpy as np
import pygame
import random
from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args, OLD_LAYOUTS

from zsceval.envs.overcooked.Overcooked_Env import Overcooked
import yaml
import pickle
import torch
path = "../policy_pool"
os.environ["POLICY_POOL"] = path

from zsceval.algorithms.population.policy_pool import add_path_prefix
from zsceval.runner.shared.base_runner import make_trainer_policy_cls

from zsceval.viz.gradcam import GradCAM
import cv2

def parse_args(args, parser):
    parser = get_overcooked_args(parser)
    parser.add_argument(
        "--use_phi",
        default=False,
        action="store_true",
        help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.",
    )

    parser.add_argument("--population_yaml_path", default="./config/random3_benchmark.yml", type=str)
    parser.add_argument("--algo", type=str, default="FCP", choices=["FCP", "SP"], help="algo key in population yaml")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--epsilon", type=float, default=0.0, help="stochastic eval epsilon")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--is_cam", action="store_true")

    all_args = parser.parse_args(args)
    if all_args.layout_name in OLD_LAYOUTS:
        all_args.old_dynamics = True
    else:
        all_args.old_dynamics = False
    return all_args


class EvalPolicy_Play:

    def __init__(self, population_yaml_path, layout_name, test_policy_name, deterministic=True, epsilon=0.5):
        self.population_yaml_path = population_yaml_path
        self.layout_name = layout_name
        self.test_policy_name = test_policy_name
        self.deterministic = deterministic
        self.epsilon = epsilon
        self.population_config = yaml.load(open(self.population_yaml_path, encoding="utf-8"), yaml.Loader)

        policy_config_path = os.path.join("../policy_pool",
                                          self.population_config[self.test_policy_name]["policy_config_path"])
        policy_config = list(pickle.load(open(policy_config_path, "rb")))
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
        action, rnn_states = self.policy.act(obs, rnn_states, masks, available_actions=available_actions,
                                             deterministic=deterministic)
        return action, rnn_states

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

    def init_cam(self):
        target_layer = self.policy.actor.base.cnn[4]
        cam = GradCAM(model=self.policy.actor, target_layer=target_layer)
        return cam


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    env = Overcooked(all_args, run_dir=None)

    agent0_play = EvalPolicy_Play(all_args.population_yaml_path, all_args.layout_name, test_policy_name="fcp1")
    masks, rnn_states = agent0_play.init_mask_rnn_state()
    
    if all_args.is_cam:
        cam = agent0_play.init_cam()

    both_agents_ob, share_obs, available_actions = env.reset()

    clock = pygame.time.Clock()
    epi_done = False
    try:
        image = env.play_render()
        screen = pygame.display.set_mode((image.shape[1], image.shape[0]))
        screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[..., ::-1], 1))), (0, 0))
        pygame.display.flip()

        while not epi_done:
            clock.tick(6.67)

            a0, rnn_states = agent0_play.get_action(np.expand_dims(both_agents_ob[0], axis=0),
                                                    available_actions, masks,
                                                    rnn_states)
            a1 = random.randint(0, 5)
            joint_action = np.array([[int(a0)], [int(a1)]])

            if all_args.is_cam:
                cam_heatmap = cam(np.expand_dims(both_agents_ob[0], axis=0),
                                  available_actions, rnn_states, masks, target_action=int(a0))

            both_agents_ob, share_obs, reward, done, info, available_actions = env.step(joint_action)
            epi_done = done[0]

            # render
            image = env.play_render()
            if all_args.is_cam:
                cam_resized = cv2.resize(cam_heatmap, (image.shape[1], image.shape[0]))        # float[0..1]
                heat = (cam_resized * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:, :, ::-1]
                image = cv2.addWeighted(image, 0.6, heatmap_color, 0.5, 0)

            display_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[..., ::-1], 1))), (0, 0))
            pygame.display.flip()

    finally:
        pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1:])
