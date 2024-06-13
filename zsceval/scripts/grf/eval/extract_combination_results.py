import argparse
import glob
import itertools
import json
import os
import os.path as osp
import sys
from collections import defaultdict
from pprint import pformat, pprint

import numpy as np
import yaml
from loguru import logger
from rich.pretty import pretty_repr
from scipy.stats import bootstrap, trim_mean

# 1. return
# 2. normalized score
# for different positions

ALG_EXPS = {
    "sp": {
        "fcp/s1/sp",
    },
    "fcp": {
        "fcp/s2/fcp-S2-s9",
    },
    "mep": {
        "mep/s2/mep-S2-s9",
    },
    "traj": {
        "traj/s2/traj-S2-s9",
    },
    "hsp": {
        "hsp/s2/hsp-S2-s9",
    },
    "cole": {
        "cole/s2/cole-S2-s15",
    },
    "e3t": {
        "e3t/s1/e3t",
    },
}

LAYOUT_2_N = {"academy_3_vs_1_with_keeper": 3}

BR_YML_PATH = "../../policy_pool/{layout}/hsp"
BIAS_RESULT_DIR = "eval/results/{layout}/bias"
EVAL_RESULT_DIR = "eval/results/{layout}/{algo}"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layout", type=str, required=True)
    parser.add_argument("-a", "--algorithm", type=str, action="append", required=True)
    args = parser.parse_args()
    return args


def get_agent_pairs(population: list, agent_name: str, num_agents: int):
    all_agent_pairs = []
    for n in range(num_agents - 1, 0, -1):
        if len(population) < n:
            continue
        pairs = list(itertools.product(population, repeat=n))
        for pop_pair in pairs:
            for a_i_tuple in itertools.combinations(range(num_agents), num_agents - n):
                p_i = 0
                _c = []
                for i in range(num_agents):
                    if i in a_i_tuple:
                        _c.append(agent_name)
                    else:
                        _c.append(pop_pair[p_i])
                        p_i += 1
                all_agent_pairs.append(_c)
    return all_agent_pairs


def extract_one_exp(layout, algo, exp, seed):
    ego_name = f"{exp}-{seed}"
    file_name = f"eval-{ego_name}.json"
    file_path = osp.join(EVAL_RESULT_DIR.format(layout=layout, algo=algo if algo != "sp" else "fcp"), file_name)
    exp_dict = json.load(open(file_path, "r", encoding="utf-8"))
    return ego_name, exp_dict


def scipy_iqm(data):
    return trim_mean(data, 0.25)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    args = get_args()
    layout = args.layout
    algos = args.algorithm
    num_agents = LAYOUT_2_N[layout]
    logger.success(args)

    br_yml_dir = BR_YML_PATH.format(layout=layout)
    br_yml_pattern = osp.join(br_yml_dir, "train_br_*.yml")
    eval_result = defaultdict(dict)
    agent_pair_results = {}
    for br_yml_path in glob.glob(br_yml_pattern):
        logger.debug(br_yml_path)
        yml_dict = yaml.load(open(br_yml_path, "r", encoding="utf-8"), Loader=yaml.FullLoader)
        yml_name = os.path.basename(br_yml_path)

        br_result_path = os.path.join(BIAS_RESULT_DIR.format(layout=layout), f"{yml_name}.json")
        logger.info(f"br_result_path {br_result_path}")
        br_result = json.load(open(br_result_path, "r", encoding="utf-8"))
        br_agent_ep_sparse_r = br_result["either-br_agent-eval_ep_sparse_r"]
        logger.info(f"br_agent_ep_sparse_r: {br_agent_ep_sparse_r}")

        bias_agent_names = list(yml_dict.keys())
        bias_agent_names.remove("br_agent")
        agent_pair = "-".join(sorted(bias_agent_names))
        agent_pair_results[agent_pair] = br_agent_ep_sparse_r
        logger.info(f"bias agents: {bias_agent_names}")

    final_result = defaultdict(lambda: [[], []])
    for algo in algos:
        eval_result[algo] = defaultdict(dict)
        if isinstance(ALG_EXPS[algo], dict):
            for exp, pop_list in ALG_EXPS[algo].items():
                for pop_name in pop_list:
                    eval_result[algo][exp][pop_name] = defaultdict(dict)
                    for seed in range(1, 4):
                        eval_result[algo][exp][pop_name][seed] = defaultdict(lambda: [[], []])
                        agent_name, exp_dict = extract_one_exp(algo, exp, pop_name, seed)
                        for agent_pair, agent_pair_res in agent_pair_results.items():
                            bias_agent_names = agent_pair.split("-")
                            combs = get_agent_pairs(bias_agent_names, agent_name, num_agents)
                            logger.debug(f"combs: {pretty_repr(combs)}")
                            for comb in combs:
                                res_name = "-".join(comb) + "-eval_ep_sparse_r"
                                eval_result[algo][exp][pop_name][seed][agent_pair][0].append(exp_dict[res_name])
                                eval_result[algo][exp][pop_name][seed][agent_pair][1].append(
                                    exp_dict[res_name] / agent_pair_results[agent_pair]
                                )
                            for i in range(2):
                                eval_result[algo][exp][pop_name][seed][agent_pair][i] = np.mean(
                                    eval_result[algo][exp][pop_name][seed][agent_pair][i]
                                )
                        pop_res_list = [[], []]
                        for v in eval_result[algo][exp][pop_name][seed].values():
                            pop_res_list[0].append(v[0])
                            pop_res_list[1].append(v[1])
                        eval_result[algo][exp][pop_name][seed]["iqm"] = [list(map(IQM, pop_res_list))]
                        eval_result[algo][exp][pop_name][seed]["iqm"] += [
                            bootstrap([pop_res_list[i]], scipy_iqm).confidence_interval for i in range(2)
                        ]
                    final_result[(algo, exp, pop_name)] = np.mean(
                        [eval_result[algo][exp][pop_name][seed]["iqm"] for seed in range(1, 4)],
                        axis=0,
                    )
        else:
            for exp in ALG_EXPS[algo]:
                eval_result[algo][exp][""] = defaultdict(dict)
                for seed in range(1, 4):
                    eval_result[algo][exp][""][seed] = defaultdict(lambda: [[], []])
                    agent_name, exp_dict = extract_one_exp(algo, exp, "", seed)
                    for agent_pair, agent_pair_res in agent_pair_results.items():
                        bias_agent_names = agent_pair.split("-")
                        combs = get_agent_pairs(bias_agent_names, agent_name, num_agents)
                        logger.debug(f"combs: {pretty_repr(combs)}")
                        for comb in combs:
                            res_name = "-".join(comb) + "-eval_ep_sparse_r"
                            eval_result[algo][exp][""][seed][agent_pair][0].append(exp_dict[res_name])
                            eval_result[algo][exp][""][seed][agent_pair][1].append(
                                exp_dict[res_name] / agent_pair_results[agent_pair]
                            )
                        for i in range(2):
                            eval_result[algo][exp][""][seed][agent_pair][i] = np.mean(
                                eval_result[algo][exp][""][seed][agent_pair][i]
                            )
                    exp_res_list = [[], []]
                    for v in eval_result[algo][exp][""][seed].values():
                        exp_res_list[0].append(v[0])
                        exp_res_list[1].append(v[1])
                    eval_result[algo][exp][""][seed]["iqm"] = [list(map(IQM, exp_res_list))]
                    eval_result[algo][exp][""][seed]["iqm"] += [
                        bootstrap([exp_res_list[i]], scipy_iqm).confidence_interval for i in range(2)
                    ]
                final_result[(algo, exp, "")] = np.mean(
                    [eval_result[algo][exp][""][seed]["iqm"] for seed in range(1, 4)], axis=0
                )

        os.makedirs(f"eval/results/{layout}/{algo}", exist_ok=True)
        with open(f"eval/results/{layout}/{algo}/eval_results.json", "w", encoding="utf-8") as f:
            json.dump(eval_result[algo], f)
        logger.success(f'results save in {f"eval/results/{layout}/{algo}/eval_results.json"}')

    logger.success(f"eval_result:\n{pretty_repr(dict(final_result), max_depth=300)}")
