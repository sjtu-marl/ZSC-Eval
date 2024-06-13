import argparse

from zsceval.config import scientific_notation

OLD_LAYOUTS = [
    "random0",
    "random0_medium",
    "random1",
    "random3",
    "small_corridor",
    "unident_s",
]


def get_overcooked_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--layout_name",
        type=str,
        default="cramped_room",
        help="Name of Submap, 40+ in choice. See /src/data/layouts/.",
    )
    parser.add_argument("--num_agents", type=int, default=1, help="number of players")
    parser.add_argument(
        "--use_timestep_feature",
        action="store_true",
        default=False,
        help="add timestep as a feature",
    )
    parser.add_argument(
        "--use_identity_feature",
        action="store_true",
        default=False,
        help="add id as a feature",
    )
    parser.add_argument(
        "--use_agent_policy_id",
        default=False,
        action="store_true",
        help="Add policy id into share obs, default False",
    )
    parser.add_argument(
        "--initial_reward_shaping_factor",
        type=float,
        default=1.0,
        help="Shaping factor of potential dense reward.",
    )
    parser.add_argument(
        "--reward_shaping_factor",
        type=float,
        default=1.0,
        help="Shaping factor of potential dense reward.",
    )
    parser.add_argument(
        "--reward_shaping_horizon",
        type=scientific_notation,
        default=2.5e6,
        help="Shaping factor of potential dense reward.",
    )
    parser.add_argument(
        "--random_start_prob",
        default=0.0,
        type=float,
        help="Probability to use a random start state, default 0.",
    )
    parser.add_argument("--use_random_terrain_state", default=False, action="store_true")
    parser.add_argument("--use_random_player_pos", default=False, action="store_true")
    parser.add_argument("--overcooked_version", default="old", type=str, choices=["new", "old"])
    parser.add_argument("--random_index", default=False, action="store_true")
    parser.add_argument("--use_hsp", default=False, action="store_true")
    parser.add_argument("--w0_offset", default=0, type=int)
    parser.add_argument(
        "--w0",
        type=str,
        default="1,1,1,1",
        help="Weight vector of dense reward 0 in overcooked env.",
    )
    parser.add_argument(
        "--w1",
        type=str,
        default="1,1,1,1",
        help="Weight vector of dense reward 1 in overcooked env.",
    )

    parser.add_argument("--num_initial_state", type=int, default=5)
    parser.add_argument("--replay_return_threshold", type=float, default=0.75)
    return parser
