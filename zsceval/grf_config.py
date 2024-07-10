import argparse
import json


def get_grf_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="academy_3_vs_1_with_keeper",
        help="name of scenarios",
        choices=[
            "academy_3_vs_1_with_keeper",
        ],
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="simple115v2_custom",
        help="feature",
        choices=["simple115v2_custom", "simple115v2"],
    )
    parser.add_argument(
        "--action_set",
        type=str,
        default="default",
        choices=["default", "v2"],
    )

    parser.add_argument("--num_agents", type=int, default=1, help="number of players")
    parser.add_argument(
        "--use_agent_policy_id",
        default=False,
        action="store_true",
        help="Add policy id into share obs, default False",
    )
    parser.add_argument("--obs_last_action", default=False, action="store_true")

    parser.add_argument(
        "--rewards",
        type=str,
        default="scoring,checkpoints",
        choices=["scoring,checkpoints", "scoring"],
    )
    parser.add_argument("--share_reward", default=False, action="store_true")
    parser.add_argument("--reward_shaping", default=True, action="store_false")
    parser.add_argument("--reward_config", type=json.loads, default={"score": 5.0, "checkpoints": 1.0})

    parser.add_argument("--random_index", default=False, action="store_true")
    parser.add_argument("--use_hsp", default=False, action="store_true")
    parser.add_argument("--w0_offset", default=0, type=int)
    parser.add_argument(
        "--w0",
        type=str,
        default="1,1,1,1",
        help="Weight vector of dense reward 0 in grf env.",
    )

    return parser
