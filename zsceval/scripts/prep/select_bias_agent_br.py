import argparse
import glob
import itertools
import json
import math
import os
import random
from collections import defaultdict
from itertools import permutations

import numpy as np
import pandas as pd
import tqdm
from dppy.finite_dpps import FiniteDPP
from loguru import logger
from numba import jit

LAYOUT_2_N = {"academy_3_vs_1_with_keeper": 3}

ENV_2_Threshold = {"overcooked": 10, "grf": 0.05}


def compute_metric(events: dict, event_types: list, num_agents: int):
    event_types_bi = [
        [f"{w0_i}-{k}_by_agent{a_i}" for a_i in range(num_agents) if a_i != w0_i]
        for k in event_types
        for w0_i in range(num_agents)
    ]
    event_types_bi = sum(event_types_bi, start=[])
    event_types_bi = sorted(event_types_bi, key=lambda x: "by_agent0" not in x)

    def empty_event_count():
        return {k: 0 for k in event_types_bi}

    ec = defaultdict(empty_event_count)
    for exp in events.keys():
        exp_i = int(exp.split("_")[0][3:])
        exp_ec = events[exp]
        for k in event_types_bi:
            ec[exp_i][k] += exp_ec.get(k, 0)
    exps = sorted(ec.keys())
    # logger.info(f"exps: {exps}")
    event_np = np.array([[ec[i][k] for k in event_types_bi] for i in exps])
    df = pd.DataFrame(event_np, index=exps, columns=event_types_bi)
    # logger.info(f"event df shape {df.shape}")
    event_ratio_np = event_np / (event_np.max(axis=0) + 1e-3).reshape(1, -1)

    return exps, event_ratio_np, df


def select_policies_dpp(exps, metric_np, K, similarity_matrix: np.ndarray = None, rng=None):
    if similarity_matrix is None:
        logger.debug("compute similarity_matrix")
        similarity_matrix = metric_np @ metric_np.T
    D = FiniteDPP("likelihood", **{"L": similarity_matrix})
    D.sample_exact_k_dpp(size=K, random_state=rng)
    selected_policies = D.list_of_samples[-1]

    selected_matrix = similarity_matrix[selected_policies][:, selected_policies]
    det = np.linalg.det(selected_matrix)
    return sorted(exps[i] for i in selected_policies), det


@jit(nopython=True)
def policies_det(selected_exps, similarity_matrix: np.ndarray = None) -> float:
    selected_matrix = similarity_matrix[selected_exps][:, selected_exps]
    det = np.linalg.det(selected_matrix)
    return det


def parse_args():
    parser = argparse.ArgumentParser(description="BR-Div", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", type=str, default="overcooked", choices=["overcooked", "grf"])
    parser.add_argument("-l", "--layout", type=str, required=True, help="layout name")
    parser.add_argument("--k", type=int, default=15, help="number of selected policies")
    parser.add_argument("--N", type=int, default=100000, help="number of random sample")
    parser.add_argument(
        "--eval_result_dir",
        type=str,
        default="eval/results",
    )
    parser.add_argument("--policy_pool_path", type=str, default="../policy_pool")
    parser.add_argument("--bias_agent_version", type=str, default="hsp")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    layout = args.layout
    if layout in LAYOUT_2_N:
        num_agents = LAYOUT_2_N[layout]
    else:
        num_agents = 2
    overcooked_version = "old"
    if layout in ["random0_m", "random1_m", "random3_m"]:
        overcooked_version = "new"
    K = args.k
    policy_version = args.bias_agent_version
    np.random.seed(0)
    random.seed(0)

    if args.env == "overcooked":
        if overcooked_version == "old":
            pass

            # event_types = SHAPED_INFOS + ["sparse_r"]
            event_types = [
                "put_onion_on_X",
                "put_dish_on_X",
                "put_soup_on_X",
                "pickup_onion_from_X",
                "pickup_onion_from_O",
                "pickup_dish_from_X",
                "pickup_dish_from_D",
                "pickup_soup_from_X",
                "USEFUL_DISH_PICKUP",  # counted when #taken_dishes < #cooking_pots + #partially_full_pots and no dishes on the counter
                "SOUP_PICKUP",  # counted when soup in the pot is picked up (not a soup placed on the table)
                "PLACEMENT_IN_POT",  # counted when some ingredient is put into pot
                "delivery",
                "STAY",
                # "MOVEMENT",
                # "IDLE_MOVEMENT",
                # "IDLE_INTERACT_X",
                # "IDLE_INTERACT_EMPTY",
                "sparse_r",
            ]
        else:
            event_types = [
                "put_onion_on_X",
                "put_tomato_on_X",
                "put_dish_on_X",
                "put_soup_on_X",
                "pickup_onion_from_X",
                "pickup_onion_from_O",
                "pickup_tomato_from_X",
                "pickup_tomato_from_T",
                "pickup_dish_from_X",
                "pickup_dish_from_D",
                "pickup_soup_from_X",
                "USEFUL_DISH_PICKUP",  # counted when #taken_dishes < #cooking_pots + #partially_full_pots and no dishes on the counter
                "SOUP_PICKUP",  # counted when soup in the pot is picked up (not a soup placed on the table)
                "PLACEMENT_IN_POT",  # counted when some ingredient is put into pot
                "viable_placement",
                "optimal_placement",
                "catastrophic_placement",
                "useless_placement",  # pot an ingredient to a useless recipe
                "potting_onion",
                "potting_tomato",
                "cook",
                "delivery",
                "deliver_size_two_order",
                "deliver_size_three_order",
                "deliver_useless_order",
                "STAY",
                "MOVEMENT",
                "IDLE_MOVEMENT",
                "IDLE_INTERACT",
            ]
    elif args.env == "grf":
        # from zsceval.envs.grf.grf_env import SHAPED_INFOS

        event_types = [
            "actual_pass",
            "shot",
            "catch",
            "assist",
            "possession",
            "score",
        ]

    events = dict()
    eval_result_dir = os.path.join(args.env, args.eval_result_dir, layout, "bias")
    logfiles = glob.glob(f"{eval_result_dir}/eval-hsp*.json")
    logfiles = [l_f for l_f in logfiles if "mid" not in l_f]

    # logger.info(logfiles)
    logger.success(f"{len(logfiles)} models")
    n = len(logfiles)
    exclude = set()

    for logfile in logfiles:
        for e in exclude:
            if e in logfile:
                continue
        with open(logfile, encoding="utf-8") as f:
            eval_result = json.load(f)
            hsp_exp_name = os.path.basename(logfile).split(".json")[0].split("-")[1].split("_")[0]
            agents = []
            for a_i in range(num_agents):
                # example: hsp36_final_w0-hsp36_final_w1-hsp36_final_w2
                agents.append(f"{hsp_exp_name}_final_w{a_i}")
            exp_name = f"{hsp_exp_name}"
            full_exp_name = "-".join(agents)
            if eval_result[f"either-{agents[0]}-eval_ep_sparse_r"] <= ENV_2_Threshold[args.env]:
                logger.warning(f"exp {exp_name} has 0 sparse reward")
                exclude.add(hsp_exp_name)
                continue
            event_dict = defaultdict(list)
            for k in event_types:
                # only the events of the biased agent are recorded
                pairs = permutations(agents)
                for pair in pairs:
                    pair_name = "-".join(pair)
                    w0_i = -1
                    for a_i, a_name in enumerate(pair):
                        if "w0" in a_name:
                            w0_i = a_i
                    for a_i, a_name in enumerate(pair):
                        if "w0" in a_name:
                            continue
                        event_dict[f"{w0_i}-{k}_by_agent{a_i}"].append(
                            eval_result[f"{pair_name}-eval_ep_{k}_by_agent{a_i}"]
                        )
            for k, v in event_dict.items():
                event_dict[k] = np.mean(v)
            events[exp_name] = event_dict
    logger.info(f"{exclude}")
    logger.info(f"exp num {len(events.keys())}")
    logger.info(f"{list(events.keys())}")
    for e in exclude:
        for k in list(events.keys()):
            if e == k:
                events.pop(k)
    # logger.debug(pformat(events))
    exps, metric_np, df = compute_metric(events, event_types, num_agents)
    logger.info(metric_np.shape)
    df.to_excel(f"{eval_result_dir}/event_count_br.xlsx", sheet_name="Events")

    K = min(K, len(exps))
    logger.warning(f"K was set to {K} in {layout}")
    max_det = 0
    max_step = 0
    similarity_matrix = metric_np @ metric_np.T
    N = min(args.N, math.comb(len(exps), K))
    if N < args.N:
        for i, comb in tqdm.tqdm(enumerate(itertools.combinations(range(len(exps)), K))):
            comb = np.array(comb)
            _det = policies_det(comb, similarity_matrix)
            if _det > max_det:
                max_det = _det
                max_step = i
                s_exps = np.array(exps)[comb].tolist()
                logger.info(f"max det {max_det} at step {max_step}")
    else:
        tqdm_iter = tqdm.tqdm(range(0, N))
        for i in tqdm_iter:
            _exps, _det = select_policies_dpp(exps, metric_np, K, similarity_matrix, np.random.RandomState(i))
            if _det > max_det:
                max_det = _det
                max_step = i
                s_exps = _exps
                logger.info(f"max det {max_det} at step {max_step}: {s_exps}")
                tqdm_iter.desc = f"max det {max_det:.10f} at {max_step}"
    logger.success(f"det {max_det} at {max_step}")
    result_dir = f"prep/results/{layout}"
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{policy_version}-s{K}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"selection": s_exps, "det": max_det}, f)
        logger.success(f"results saved in {result_path}")

    logger.info(f"Selected exps {s_exps}")

    # generate evaluation config
#     benchmark_yml_path = f"{args.policy_pool_path}/{layout}/hsp/s1/{policy_version}/benchmarks-s{args.k * 2}.yml"
#     with open(
#         benchmark_yml_path,
#         "w",
#         encoding="utf-8",
#     ) as f:
#         for i, exp_i in enumerate(s_exps):
#             f.write(
#                 f"""\
# hsp{i+1}_mid:
#     policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
#     featurize_type: ppo
#     train: False
#     model_path:
#         actor: {layout}/hsp/s1/{policy_version}/hsp{exp_i}_mid_w0_actor.pt\n"""
#             )
#             f.write(
#                 f"""\
# hsp{i+1}_final:
#     policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
#     featurize_type: ppo
#     train: False
#     model_path:
#         actor: {layout}/hsp/s1/{policy_version}/hsp{exp_i}_final_w0_actor.pt\n"""
#             )
#         f.write(
#             f"""\
# agent_name:
#     policy_config_path: {layout}/policy_config/rnn_policy_config.pkl
#     featurize_type: ppo
#     train: False
#     model_path:
#         actor: {layout}/algorithm/s2/population/seed.pt"""
#         )
#     logger.success(f"write to {benchmark_yml_path}")
