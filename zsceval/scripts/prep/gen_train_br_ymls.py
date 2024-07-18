import argparse
import itertools

import yaml
from loguru import logger

from zsceval.utils.bias_agent_vars import LAYOUTS_EXPS, LAYOUTS_KS, LAYOUTS_NS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BR-Div")
    parser.add_argument("-l", "--layout", type=str, required=True, help="layout name")
    parser.add_argument("--bias_agent_version", type=str, default="hsp")
    args = parser.parse_args()

    if args.layout == "all":
        layouts = list(LAYOUTS_EXPS.keys())
    else:
        layouts = [args.layout]

    for layout in layouts:
        bench_dict = yaml.load(
            open(
                f"../policy_pool/{layout}/hsp/s1/{args.bias_agent_version}/benchmarks-s{LAYOUTS_KS[layout] * 2}.yml",
            ),
            Loader=yaml.FullLoader,
        )

        # logger.debug(pformat(bench_dict))

        bias_agents = list(bench_dict.keys())
        bias_agents.remove("agent_name")

        agent_dict = {
            "br_agent": {
                "policy_config_path": f"{layout}/policy_config/mlp_policy_config.pkl",
                "featurize_type": "ppo",
                "train": True,
            }
        }
        num_agents = LAYOUTS_NS[layout]
        cnt = 1
        for n_ba in range(num_agents - 1, 0, -1):
            logger.info(f"{n_ba} bias agents")
            if n_ba == 1:
                combs = itertools.combinations([ba for ba in bias_agents if "final" not in ba], n_ba)
            else:
                combs = itertools.combinations(bias_agents, n_ba)

            for ba_t in combs:
                _dict = {}
                for ba in ba_t:
                    _dict[ba] = bench_dict[ba]
                _dict |= agent_dict

                with open(f"../policy_pool/{layout}/hsp/s1/{args.bias_agent_version}/train_br_{cnt}.yml", "w") as f:
                    yaml.dump(_dict, f)
                cnt += 1
        logger.success(f"gen ymls for {cnt-1} pairs")
