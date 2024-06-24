import argparse
import os
import os.path as osp

from loguru import logger

policy_pool_dir = "../policy_pool"

S1_POP_EXPS = {
    "fcp": {
        15: "sp",
        10: "sp",
        5: "sp",
    },
    "mep": {
        15: "mep-S1-s15",
        10: "mep-S1-s10",
        5: "mep-S1-s5",
    },
    "traj": {
        15: "traj-S1-s15",
        10: "traj-S1-s10",
        5: "traj-S1-s5",
    },
}

# Overcooked
TOTAL_SIZE_LIST = [15, 10, 5]
POP_SIZE_LIST = [12, 8, 4]
# TOTAL_SIZE_LIST = [10, 5]
# POP_SIZE_LIST = [8, 4]
# TOTAL_SIZE_LIST = [5]
# POP_SIZE_LIST = [4]

# GRF
# TOTAL_SIZE_LIST = [5]
# POP_SIZE_LIST = [3]

N_REPEAT = 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("layout", type=str)
    parser.add_argument("alg", type=str)

    args = parser.parse_args()

    if args.layout == "all":
        layouts = [
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
        layouts = [args.layout]

    for layout in layouts:
        for TOTAL_SIZE, POP_SIZE in zip(TOTAL_SIZE_LIST, POP_SIZE_LIST):
            exp = S1_POP_EXPS[args.alg][TOTAL_SIZE]
            source_dir = osp.join(policy_pool_dir, layout, args.alg, "s1", exp)
            pt_lst = os.listdir(source_dir)
            logger.debug(pt_lst)
            pop_alg = args.alg if args.alg != "fcp" else "sp"
            pt_lst.sort(key=lambda pt: int(pt.split("_", 1)[0][len(pop_alg) :]))
            if args.alg == "fcp":
                pt_lst = pt_lst[: TOTAL_SIZE * 3]
                logger.info(f"pop size {len(pt_lst)}: {pt_lst}")
            yml_dir = osp.join(
                policy_pool_dir,
                layout,
                args.alg,
                "s2",
            )
            os.makedirs(yml_dir, exist_ok=True)
            for n_r in range(N_REPEAT):
                yml_path = osp.join(
                    policy_pool_dir,
                    layout,
                    args.alg,
                    "s2",
                    f"train-s{POP_SIZE*3}-{exp}-{n_r+1}.yml",
                )
                logger.info(f"Writing S2 yml for {exp} seed {n_r} in {yml_path}")
                yml = open(
                    yml_path,
                    "w",
                    encoding="utf-8",
                )
                yml.write(
                    f"""\
{args.alg}_adaptive:
    policy_config_path: {layout}/policy_config/rnn_policy_config.pkl
    featurize_type: ppo
    train: True
"""
                )
                for p_i in range(1, POP_SIZE + 1):
                    pt_i = (TOTAL_SIZE // N_REPEAT * n_r + p_i - 1) % TOTAL_SIZE + 1
                    actor_names = [
                        f"{pop_alg}{pt_i}_init_actor.pt",
                        f"{pop_alg}{pt_i}_mid_actor.pt",
                        f"{pop_alg}{pt_i}_final_actor.pt",
                    ]
                    for actor_name in actor_names:
                        assert actor_name in pt_lst, (actor_name, pt_lst)
                    yml.write(
                        f"""\
{pop_alg}{p_i}_1:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {os.path.join(layout, args.alg, "s1", exp, actor_names[0])}
{pop_alg}{p_i}_2:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {os.path.join(layout, args.alg, "s1", exp, actor_names[1])}
{pop_alg}{p_i}_3:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {os.path.join(layout, args.alg, "s1", exp, actor_names[2])}
"""
                    )
                yml.close()
