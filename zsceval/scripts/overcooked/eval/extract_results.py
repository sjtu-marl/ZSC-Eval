import argparse
import glob
import itertools
import json
import os.path as osp
import re
import sys
from pprint import pformat

import yaml
from loguru import logger
from scipy.stats import trim_mean

from zsceval.utils.bias_agent_vars import LAYOUTS_KS, LAYOUTS_NS

ALG_EXPS = {
    "sp": ["sp"],
    "fcp": [
        "fcp-S2-s12",
        "fcp-S2-s24",
        "fcp-S2-s36",
    ],
    "mep": [
        "mep-S2-s12",
        "mep-S2-s24",
        "mep-S2-s36",
    ],
    "traj": [
        "traj-S2-s12",
        "traj-S2-s24",
        "traj-S2-s36",
    ],
    "hsp": [
        "hsp-S2-s12",
        "hsp-S2-s24",
        "hsp-S2-s36",
    ],
    "cole": [
        "cole-S2-s25",
        "cole-S2-s50",
        "cole-S2-s75",
    ],
    "e3t": [
        "e3t",
    ],
}

BIAS_YML_PATH = "../../policy_pool/{layout}/hsp/s1/hsp/benchmarks-s{N}.yml"
BIAS_RESULT_DIR = "eval/results/{layout}/bias"

EVAL_RESULT_DIR = "eval/results/{layout}/{algo}"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layout", type=str, required=True)
    parser.add_argument("-a", "--algorithm", type=str, action="append", required=True)
    parser.add_argument("--eval_result_dir", type=str, default="eval/results")
    parser.add_argument("--policy_pool_path", type=str, default="../policy_pool")
    parser.add_argument("--bias_agent_version", type=str, default="hsp")
    args = parser.parse_args()
    return args


def scipy_iqm(data):
    return trim_mean(data, 0.25)


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


if __name__ == "__main__":
    args = get_args()
    layout = args.layout
    algos = args.algorithm
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    # logger.add(sys.stdout, level="DEBUG")
    logger.info(args)
    num_agents = LAYOUTS_NS[layout]

    bias_yml_path = BIAS_YML_PATH.format(layout=layout, N=LAYOUTS_KS[layout] * 2)
    yml_dict = yaml.load(open(bias_yml_path, encoding="utf-8"), Loader=yaml.FullLoader)
    bias_agent_names = yml_dict.keys()
    bias_agent_names = [name for name in bias_agent_names if name != "agent_name"]
    logger.debug(f"bias agents\n{bias_agent_names}")
    bias_result_dir = BIAS_RESULT_DIR.format(layout=layout)
    bias_agent_comb_results = {}
    # {comb: result}
    agent_name = "ego_name"
    for a_n in bias_agent_names:
        agent_name = "ego_name"

        if "final" in a_n:
            version_name = a_n.replace("bias", "hsp")
            eval_file_path = osp.join(bias_result_dir, f"eval-{version_name.split('_')[0]}.json")
            eval_result = json.load(open(eval_file_path, encoding="utf-8"))

            agents = (f"{version_name}_w0",)
            combs = get_agent_pairs(agents, agent_name, num_agents)
            # logger.debug(f"{a_n} {combs}")

            for comb in combs:
                data_name = "-".join(comb)
                actual_agent_name = f"{version_name}_w1"
                data_name = f"{data_name}-eval_ep_sparse_r".replace(agent_name, actual_agent_name)
                _comb = []
                for c in comb:
                    if c == f"{version_name}_w0":
                        _comb.append(a_n)
                    else:
                        _comb.append(c)
                _comb = tuple(_comb)
                bias_agent_comb_results[tuple(_comb)] = eval_result[data_name]
        elif "mid" in a_n:
            pass
        else:
            raise ValueError(f"Unknown bias agent name {a_n}")

    mid_result_file_paths = glob.glob(f"{bias_result_dir}/eval-br_*.json")
    logger.debug(mid_result_file_paths)
    mid_results = {}
    actual_agent_name = "br_agent"
    pattern = r"^(?!either).+eval_ep_sparse_r$"
    for m_p in mid_result_file_paths:
        eval_result = json.load(open(m_p, encoding="utf-8"))
        for k, v in eval_result.items():
            if re.match(pattern, k):
                data_name = k.replace("-eval_ep_sparse_r", "").replace(actual_agent_name, agent_name)
                comb = tuple(data_name.split("-"))
                mid_results[comb] = v
    # logger.debug(pformat(mid_results))

    for a_n in bias_agent_names:
        if "mid" in a_n:
            agents = (a_n,)
            combs = get_agent_pairs(agents, agent_name, num_agents)
            # logger.debug(f"{a_n} {combs}")
            for comb in combs:
                bias_agent_comb_results[tuple(comb)] = mid_results[tuple(comb)]

    logger.debug(pformat(bias_agent_comb_results))

    logger.info(f"Layout: {layout}")

    for alg in algos:
        alg_result = {}
        # exp: [pos0, pos1, ..., overall]
        for exp_name in ALG_EXPS[alg]:
            eval_result_dir = EVAL_RESULT_DIR.format(layout=layout, algo=alg)
            pos_results = [[] for _ in range(num_agents)]
            for seed in range(1, 6):
                actual_agent_name = f"{exp_name}-{seed}"
                eval_result = json.load(open(f"{eval_result_dir}/eval-{actual_agent_name}.json", encoding="utf-8"))
                # logger.debug(pformat(eval_result))
                combs = get_agent_pairs(bias_agent_names, f"{actual_agent_name}", num_agents)

                for comb in combs:
                    for pos in range(num_agents):
                        if comb[pos] == actual_agent_name:
                            data_name = "-".join(comb)
                            data_name = f"{data_name}-eval_ep_sparse_r"
                            br_comb = (*comb[:pos], agent_name, *comb[pos + 1 :])
                            if bias_agent_comb_results[br_comb] > 0:
                                pos_results[pos].append(
                                    min(1, eval_result[f"{data_name}"] / bias_agent_comb_results[br_comb])
                                )
                            else:
                                pos_results[pos].append(1)
            overall_score = scipy_iqm(sum(pos_results, []))
            alg_result[exp_name] = [scipy_iqm(r) for r in pos_results]
            alg_result[exp_name].append(overall_score)

        logger.info(f"Algorithm: {alg}\n{pformat(alg_result)}")
