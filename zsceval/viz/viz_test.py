import argparse
import os
import sys
import time
import numpy as np
import pygame
import random
from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args, OLD_LAYOUTS
# 환경변수 설정
path = "../policy_pool"
os.environ["POLICY_POOL"] = path

from zsceval.envs.overcooked.Overcooked_Env import Overcooked
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import  Action
from zsceval.human_exp.agent_pool import ZSCEvalAgentPool


def idx_to_action_tuple(a0_idx: int, a1_idx: int):
    return (Action.INDEX_TO_ACTION[a0_idx], Action.INDEX_TO_ACTION[a1_idx])


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

    all_args = parser.parse_args(args)
    if all_args.layout_name in OLD_LAYOUTS:
        all_args.old_dynamics = True
    else:
        all_args.old_dynamics = False
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    env = Overcooked(all_args, run_dir=None)

    pool = ZSCEvalAgentPool(all_args.population_yaml_path, all_args.layout_name, deterministic=all_args.deterministic, epsilon=all_args.epsilon)
    agent0 = pool.get_agent(all_args.algo)
    # agent1 = pool.get_agent(args.algo if args.mirror_same_policy else args.algo)

    # Pygame window
    clock = pygame.time.Clock()

    # Overcooked 클래스의 reset()은 반환값이 있음
    both_agents_ob, share_obs, available_actions = env.reset()
    done = False
    step_count = 0

    while True:
        # a0 = int(agent0(env.base_env.state.to_dict(), 0))
        a0 = random.randint(0, 5)
        a1 = random.randint(0, 5)
        joint_action = np.array([[a0], [a1]])

        both_agents_ob, share_obs, reward, done, info, available_actions = env.step(joint_action)
        episode_done = done[0] if isinstance(done, list) else done
        step_count += 1

        if step_count % 50 == 0:
            print(f"Step {step_count}: Reward = {reward}")
            print(f"현재 상태:\n{env.base_env}")

        if episode_done:
            break

if __name__ == "__main__":
    main(sys.argv[1:])
