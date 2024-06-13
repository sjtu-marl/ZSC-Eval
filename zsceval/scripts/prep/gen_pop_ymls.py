import argparse
import os
import os.path as osp

from loguru import logger

policy_pool_dir = "../policy_pool"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("layout", type=str)
    parser.add_argument("algo", type=str, choices=["traj", "mep"])
    parser.add_argument("-s", "--population_size", type=int, default=15)
    args = parser.parse_args()

    if args.layout == "all":
        args.layout = [
            "random0",
            "random0_medium",
            "random1",
            "random3",
            "small_corridor",
            "unident_s",
            "random0_m",
            "random1_m",
            "random3_m",
            "academy_3_vs_1_with_keeper",
        ]
    else:
        args.layout = [args.layout]

    logger.info(f"Generate templates for {args.layout}")
    for layout in args.layout:
        source_dir = osp.join(policy_pool_dir, layout, args.algo, "s1")
        os.makedirs(source_dir, exist_ok=True)
        s1_yml_path = osp.join(
            source_dir,
            f"train-s{args.population_size}.yml",
        )
        logger.info(f"Writing train yml for {args.algo} S1 in {s1_yml_path}")

        with open(s1_yml_path, "w", encoding="utf-8") as s1_yml:
            s1_yml.write(
                f"""\
{args.algo}_adaptive:
    policy_config_path: {layout}/policy_config/rnn_policy_config.pkl
    featurize_type: ppo
    train: False
"""
            )
            for p_i in range(args.population_size):
                s1_yml.write(
                    f"""\
{args.algo}{p_i+1}:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: True
"""
                )
