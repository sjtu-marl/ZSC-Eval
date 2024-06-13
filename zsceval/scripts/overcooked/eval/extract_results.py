import argparse
import json
import os
import os.path as osp
import sys
from collections import defaultdict
from pprint import pformat, pprint

import numpy as np
import yaml
from loguru import logger

# 1. return
# 2. normalized score
# for different positions

ALG_EXPS = {
    "sp": [
        "",
    ],
    "fcp": [
        {
            "": ["-_s12", "-_s24", "-_s36"],
        }
    ],
    "mep": [
        {
            "": ["-_s12_s5-S1", "-_s24_s10-S1", "-_s36_s15-S1"],
        }
    ],
    "traj": [
        {
            "": ["-_s12_s5-S1", "-_s24_s10-S1", "-_s36_s15-S1"],
        }
    ],
    "hsp": [
        {
            "": ["-_s12", "-_s24", "-_s36"],
        }
    ],
    "cole": [
        {
            "": [
                "-_s25",
                "-_s50",
                "-_s75",
            ],
        }
    ],
    "e3t": [
        "0.1",
    ],
}

feature_version = "notime"
# hsp_version = "aggressive-br"
hsp_version = ""

BIAS_YML_PATH = "../policy_pool/{layout}/hsp/benchmarks-br-s20.yml"
# BIAS_YML_PATH = "../policy_pool/{layout}/hsp/benchmarks-notime-aggressive-br.yml"
BIAS_RESULT_DIR = "eval/results/{layout}/bias"

EVAL_RESULT_DIR = "eval/results/{layout}/{algo}"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layout", type=str, required=True)
    parser.add_argument("-a", "--algorithm", type=str, action="append", required=True)
    args = parser.parse_args()
    return args


def IQM(data):
    q1, q3 = np.percentile(data, [25, 75])

    iqr_data = data[(data >= q1) & (data <= q3)]

    iqm = np.mean(iqr_data)

    return iqm


if __name__ == "__main__":
    logger.remove()
    # logger.add(sys.stdout, level="DEBUG")
    logger.add(sys.stdout, level="SUCCESS")
    # logger.add(sys.stdout, level="INFO")
    args = get_args()
    layout = args.layout
    algos = args.algorithm
    logger.success(args)

    bias_yml_path = BIAS_YML_PATH.format(layout=layout)
    yml_dict = yaml.load(open(bias_yml_path, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    bias_agent_names = yml_dict.keys()
    bias_agent_version_names = {
        name: "_".join(osp.basename(yml_dict[name]["model_path"]["actor"]).split("_")[:-2]) for name in bias_agent_names
    }
    logger.info(f"bias agent versions\n{bias_agent_version_names}")
    bias_agent_names = [name for name in bias_agent_names if name != "agent_name"]
    bias_agent_returns = defaultdict(list)
    # hspN: [return for hspN in position 1, return for hspN in position 0]
    bias_result_dir = BIAS_RESULT_DIR.format(layout=layout)
    for a_n in bias_agent_names:
        version_name = bias_agent_version_names[a_n]
        if "final" in version_name:
            # actual train index
            exp_name = f"{version_name}_w1-{version_name}_w0"
            file_name = f"eval-{exp_name}.json"
            file_path = osp.join(bias_result_dir, file_name)
            k = f"{exp_name}-eval_ep_sparse_r"
            bias_agent_returns[a_n].append(json.load(open(file_path, "r", encoding="utf-8"))[k])

            exp_name = f"{version_name}_w0-{version_name}_w1"
            file_name = f"eval-{exp_name}.json"
            file_path = osp.join(bias_result_dir, file_name)
            k = f"{exp_name}-eval_ep_sparse_r"
            bias_agent_returns[a_n].append(json.load(open(file_path, "r", encoding="utf-8"))[k])
        elif "mid" in version_name:
            # actual train index
            exp_name = f"{version_name}_w0-br_{version_name}_w0"
            file_name = f"train-{exp_name}.json"
            file_path = osp.join(bias_result_dir, file_name)
            mid_result_dict = json.load(open(file_path, "r", encoding="utf-8"))
            k = f"either-br_{version_name}_w0-eval_ep_sparse_r-as_agent_0"
            bias_agent_returns[a_n].append(mid_result_dict[k])
            k = f"either-br_{version_name}_w0-eval_ep_sparse_r-as_agent_1"
            bias_agent_returns[a_n].append(mid_result_dict[k])
        else:
            raise NotImplementedError

    logger.success("bias agent returns\n" + pformat(dict(bias_agent_returns)))
    bias_average = [
        [bias_agent_returns[a_n][0] for a_n in bias_agent_names],
        [bias_agent_returns[a_n][1] for a_n in bias_agent_names],
    ]
    bias_average = [np.mean(bias_average[0]), np.mean(bias_average[1])]

    logger.success("bias agent average\n" + pformat(bias_average))

    # exp_name: [return for hspN in position 1, return for hspN in position 0, normalized ...]
    def extract_one_exp(exp, suffix, tag, algo):
        # exp_returns = [[], []]
        tag_bias_agent_names = [a_n for a_n in bias_agent_names if tag in a_n]
        exp_returns = [[], [], [], []]
        prefix = f"{suffix}-" if suffix else ""
        suffix = f"-{suffix}" if suffix else ""
        # logger.warning(f"algo {algo} prefix {prefix} suffix {suffix}")
        for seed in range(1, 6):
            # if algo == "mep":
            #     ego_name = f"{algo}-p-pop{exp.split('-', 1)[1]}-S2-{seed}"
            # elif algo == "sp":
            if algo in ["sp", "e3t"]:
                ego_name = f"{algo}{suffix}-{seed}"
                # ego_name = f"{algo}-{exp.split('-', 1)[1]}-{seed}"
            # elif algo == "fcp":
            # ego_name = f"{algo}-pop{exp.split('-', 1)[1]}-S2-{seed}"
            # ego_name = f"{algo}-random_pos-random_state6-pop{exp.split('-', 1)[1]}-S2-{seed}"
            else:
                ego_name = f"{algo}{suffix}-pop{exp.split('-', 1)[1]}-S2-{seed}"
            file_name = f"eval-{ego_name}.json"
            # file_name = f"eval-{ego_name}-{hsp_version}.json"
            file_path = osp.join(
                EVAL_RESULT_DIR.format(layout=layout, algo=algo if algo != "sp" else "fcp"),
                file_name,
            )
            result_dict = json.load(open(file_path, "r", encoding="utf-8"))

            # k = f"either-{ego_name}-eval_ep_sparse_r-as_agent_0"
            # exp_returns[0].append(result_dict[k])

            # k = f"either-{ego_name}-eval_ep_sparse_r-as_agent_1"
            # exp_returns[1].append(result_dict[k])

            p0_scores = []
            for bias_an in tag_bias_agent_names:
                k = f"{ego_name}-{bias_an}-eval_ep_sparse_r"
                s = result_dict[k]
                p0_scores.append(s)
            exp_returns[0].append(np.median(p0_scores))

            p1_scores = []
            for bias_an in tag_bias_agent_names:
                k = f"{bias_an}-{ego_name}-eval_ep_sparse_r"
                s = result_dict[k]
                p1_scores.append(s)
            exp_returns[1].append(np.median(p1_scores))

            p0_norm_scores = []
            for bias_an in tag_bias_agent_names:
                k = f"{ego_name}-{bias_an}-eval_ep_sparse_r"
                if bias_agent_returns[bias_an][0] > 10:
                    n_s = result_dict[k] / bias_agent_returns[bias_an][0]
                    # p0_norm_scores.append(min(1.0, n_s))
                    p0_norm_scores.append(n_s)
            logger.debug(f"{p0_norm_scores}")
            exp_returns[2].append(
                np.median(p0_norm_scores),
                # [
                #     np.mean(p0_norm_scores),
                #     np.max(p0_norm_scores),
                #     np.min(p0_norm_scores),
                # ]
            )

            p1_norm_scores = []
            for bias_an in tag_bias_agent_names:
                k = f"{bias_an}-{ego_name}-eval_ep_sparse_r"
                if bias_agent_returns[bias_an][1] > 10:
                    n_s = result_dict[k] / bias_agent_returns[bias_an][1]
                    # p1_norm_scores.append(min(1.0, n_s))
                    p1_norm_scores.append(n_s)
                # logger.debug(f"{result_dict[k]} {bias_agent_returns[bias_an][1]} {result_dict[k] / bias_agent_returns[bias_an][1]}")
            logger.debug(f"{p1_norm_scores}")
            exp_returns[3].append(
                np.median(p1_norm_scores),
                # [
                #     np.mean(p1_norm_scores),
                #     np.max(p1_norm_scores),
                #     np.min(p1_norm_scores),
                # ]
            )
        logger.debug("detail\n" + pformat(exp_returns))
        # logger.warning(f"x{prefix}x")
        if "cole" not in prefix and algo not in ["sp", "e3t"]:
            _exp_name = f"{exp.split('_', 1)[1]}"
        elif algo in ["sp", "e3t"]:
            _exp_name = algo
        else:
            _exp_name = f"{exp.split('_', 1)[1]}"

        ego_agent_returns[_exp_name] = [
            np.mean(r, 0) if len(np.array(r).shape) == 1 else np.mean(r, 0).tolist() for r in exp_returns
        ]
        # ego_agent_returns[exp] = [
        #     np.median(r, 0) if len(np.array(r).shape) == 1 else np.median(r, 0).tolist()
        #     for r in exp_returns
        # ]
        # exp_returns = [np.array(r) for r in exp_returns]
        # # pprint(exp_returns)
        # ego_agent_returns[exp] = [
        #     IQM(r) if len(r.shape) == 1 else [IQM(r[:, c]) for c in range(r.shape[1])]
        #     for r in exp_returns
        # ]

    for algo in algos:
        logger.success(algo)
        for exp in ALG_EXPS[algo]:
            # total_ego_agent_returns = []
            tag_ego_agent_returns = {"mid": [], "final": [], "total": []}
            for tag in ["mid", "final"]:
                ego_agent_returns = defaultdict(list)
                logger.info(f"Tag: {tag}")
                if isinstance(exp, dict):
                    for exp_name, pop_names in exp.items():
                        for pop_name in pop_names:
                            logger.info(f"exp: {exp_name}, pop: {pop_name}")
                            extract_one_exp(pop_name, exp_name, tag, algo)
                            str_dict = {
                                exp: [f"{v:.4f}" for v in vals] for exp, vals in dict(ego_agent_returns).items()
                            }
                            logger.info("ego agent returns: " + pformat(str_dict))
                            # total_ego_agent_returns.append(ego_agent_returns)
                            tag_ego_agent_returns[tag].append(ego_agent_returns)
                            tag_ego_agent_returns["total"].append(ego_agent_returns)
                else:
                    logger.info(f"exp: {exp}, pop: ''")
                    extract_one_exp(algo, exp, tag, algo)
                    str_dict = {exp: [f"{v:.4f}" for v in vals] for exp, vals in dict(ego_agent_returns).items()}
                    logger.info("ego agent returns: " + pformat(str_dict))
                    # total_ego_agent_returns.append(ego_agent_returns)
                    tag_ego_agent_returns[tag].append(ego_agent_returns)
                    tag_ego_agent_returns["total"].append(ego_agent_returns)
            for tag in ["mid", "final", "total"]:
                logger.success(f"Tag: {tag}")
                for pop in ego_agent_returns.keys():
                    scores = np.mean([np.array(ret[pop]) for ret in tag_ego_agent_returns[tag]], axis=0).tolist()
                    score = np.mean(scores[:2]), np.mean(scores[2:4])
                    logger.success(
                        f"{pop}: {['{:.4f}'.format(s) for s in scores]}, {['{:.4f}'.format(s) for s in score]}"
                    )
            # logger.success("Tag: total")
            # for pop in ego_agent_returns.keys():
            #     scores = np.mean(
            #         [np.array(ret[pop]) for ret in total_ego_agent_returns], axis=0
            #     ).tolist()
            #     score = np.mean(scores[:2]), np.mean(scores[2:4])
            #     logger.success(
            #         f"{pop}: {['{:.4f}'.format(s) for s in scores]}, {['{:.4f}'.format(s) for s in score]}"
            #     )
        os.makedirs(f"eval/results/{layout}/{algo}", exist_ok=True)
        with open(f"eval/results/{layout}/{algo}/eval_results.json", "w", encoding="utf-8") as f:
            # json.dump(ego_agent_returns, f)
            json.dump(tag_ego_agent_returns, f)
        logger.success(f'results save in {f"eval/results/{layout}/{algo}/eval_results.json"}')
