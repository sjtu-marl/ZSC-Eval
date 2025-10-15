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
from zsceval.human_exp.agent_pool import ZSCEvalAgentPool

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
            # a0 = int(agent0(env.base_env.state.to_dict(), 0))
            a0 = random.randint(0, 5)
            a1 = random.randint(0, 5)
            joint_action = np.array([[a0], [a1]])

            both_agents_ob, share_obs, reward, done, info, available_actions = env.step(joint_action)
            epi_done = done[0]

            # render
            image = env.play_render()
            screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[..., ::-1], 1))), (0, 0))
            pygame.display.flip()

    finally:
        pygame.quit()

if __name__ == "__main__":
    main(sys.argv[1:])
