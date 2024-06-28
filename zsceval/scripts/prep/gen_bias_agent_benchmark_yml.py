import argparse

from loguru import logger

from zsceval.utils.bias_agent_vars import LAYOUTS_EXPS, LAYOUTS_KS


def parse_args():
    parser = argparse.ArgumentParser(description="zsceval", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-l", "--layout", type=str, required=True, help="layout name")
    parser.add_argument("--eval_result_dir", type=str, default="eval/results")
    parser.add_argument("--policy_pool_path", type=str, default="../policy_pool")
    parser.add_argument("--bias_agent_version", type=str, default="hsp")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    layout = args.layout
    assert layout in list(LAYOUTS_EXPS.keys()) + ["all"]
    if layout == "all":
        layout = list(LAYOUTS_EXPS.keys())
    else:
        layout = [layout]
    policy_version = args.bias_agent_version

    for l in layout:
        logger.info(f"layout: {l}")
        K = LAYOUTS_KS[l]
        s_exps = LAYOUTS_EXPS[l]

        assert len(s_exps) == K
        # generate HSP evaluation config
        benchmark_yml_path = f"{args.policy_pool_path}/{l}/hsp/s1/{policy_version}/benchmarks-s{K * 2}.yml"
        with open(
            benchmark_yml_path,
            "w",
            encoding="utf-8",
        ) as f:
            for i, exp_i in enumerate(s_exps):
                f.write(
                    f"""\
bias{i+1}_mid:
    policy_config_path: {l}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {l}/hsp/s1/{policy_version}/hsp{exp_i}_mid_w0_actor.pt\n"""
                )
                f.write(
                    f"""\
bias{i+1}_final:
    policy_config_path: {l}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {l}/hsp/s1/{policy_version}/hsp{exp_i}_final_w0_actor.pt\n"""
                )
            f.write(
                f"""\
agent_name:
    policy_config_path: {l}/policy_config/rnn_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {l}/algorithm/s2/population/seed.pt"""
            )
        logger.success(f"write to {benchmark_yml_path}")
