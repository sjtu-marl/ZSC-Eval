import argparse
import glob
import json
import os
import random
from collections import defaultdict
from itertools import permutations

import numpy as np
import pandas as pd
from loguru import logger

from zsceval.utils.bias_agent_vars import LAYOUTS_EXPS


def compute_metric(events: dict, event_types: list, num_agents: int):
    event_types_bi = [
        [f"{w0_i}-{k}_by_agent{a_i}" for a_i in range(num_agents) if a_i != w0_i]
        for w0_i in range(num_agents)
        for k in event_types
    ]
    event_types_bi = sum(event_types_bi, start=[])

    def empty_event_count():
        return {k: 0 for k in event_types_bi}

    ec = defaultdict(empty_event_count)
    for exp in events.keys():
        exp_i = int(exp.split("_")[0][3:])
        exp_ec = events[exp]
        for k in event_types_bi:
            ec[exp_i][k] += exp_ec.get(k, 0)
    exps = sorted(ec.keys())
    logger.info(f"exps: {exps}")
    event_np = np.array([[ec[i][k] for k in event_types_bi] for i in exps])
    df = pd.DataFrame(event_np, index=exps, columns=event_types_bi)
    logger.info(f"event df shape {df.shape}")
    event_ratio_np = event_np / (event_np.max(axis=0) + 1e-3).reshape(1, -1)

    return exps, event_ratio_np, df


def select_policies(runs, metric_np, K):
    S = []
    n = len(runs)
    S.append(np.random.randint(0, n))
    for _ in range(1, K):
        v = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            if i not in S:
                for j in S:
                    v[i] += abs(metric_np[i] - metric_np[j]).sum()
            else:
                v[i] = -1e9
        x = v.argmax()
        S.append(x)
    S = sorted([runs[i] for i in S])
    return S


MEP_EXPS = {
    5: "mep-S1-s5",
    10: "mep-S1-s10",
    15: "mep-S1-s15",
}


def parse_args():
    parser = argparse.ArgumentParser(description="hsp", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--env", type=str, default="Overcooked")
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("-l", "--layout", type=str, required=True, help="layout name")
    parser.add_argument("-k", type=int, default=6, help="number of selected policies")
    parser.add_argument("-s", type=int, default=5, help="population size of S1")
    parser.add_argument("-S", type=int, default=12, help="population size of training")
    parser.add_argument("--eval_result_dir", type=str, default="eval/results")
    parser.add_argument("--policy_pool_path", type=str, default="../policy_pool")
    parser.add_argument("--bias_agent_version", type=str, default="hsp")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    layout = args.layout
    overcooked_version = "old"
    if layout in ["random0_m", "random1_m", "random3_m"]:
        overcooked_version = "new"
    K = args.k
    policy_version = args.bias_agent_version
    np.random.seed(0)
    random.seed(0)

    if args.env.lower() == "overcooked":
        if overcooked_version == "old":
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
    else:
        event_types = [
            "actual_pass",
            "shot",
            "catch",
            "assist",
            "possession",
            "score",
        ]

    events = dict()
    eval_result_dir = os.path.join(args.env.lower(), args.eval_result_dir, layout, "bias")
    logger.info(f"eval result dir {eval_result_dir}")
    logfiles = glob.glob(f"{eval_result_dir}/eval*{policy_version}*.json")
    logfiles = [l_f for l_f in logfiles if "mid" not in l_f]
    # logger.success(f"{logfiles}")
    logger.success(f"{len(logfiles) // 2} models")

    exclude = set(map(lambda x: f"hsp{x}", LAYOUTS_EXPS[layout]))
    for logfile in logfiles:
        for e in exclude:
            if e in logfile:
                logfiles.remove(logfile)
                break
    for logfile in logfiles:
        for e in exclude:
            if e in logfile:
                continue
        with open(logfile, encoding="utf-8") as f:
            eval_result = json.load(f)
            hsp_exp_name = os.path.basename(logfile).split("eval-")[1].split(".")[0]
            agents = []
            for a_i in range(args.num_agents):
                # example: hsp36_final_w0-hsp36_final_w1-hsp36_final_w2
                agents.append(f"{hsp_exp_name}_final_w{a_i}")
            exp_name = f"{hsp_exp_name}"
            full_exp_name = "-".join(agents)
            if eval_result[f"{full_exp_name}-eval_ep_sparse_r"] <= 0.1:
                logger.warning(f"exp {exp_name} has 0 sparse reward")
                exclude.add(hsp_exp_name)
                continue
            event_dict = defaultdict(list)
            for k in event_types:
                # only the events of the biased agent are recorded
                agent_names = [f"{hsp_exp_name}_final_w{a_i}" for a_i in range(args.num_agents)]
                pairs = permutations(agent_names)
                for pair in pairs:
                    pair_name = "-".join(pair)
                    w0_i = -1
                    for a_i, a_name in enumerate(pair):
                        if "w0" in a_name:
                            w0_i = a_i
                            break
                    for a_i, a_name in enumerate(pair):
                        if "w0" in a_name:
                            continue
                        event_dict[f"{w0_i}-{k}_by_agent{a_i}"].append(
                            eval_result[f"{pair_name}-eval_ep_{k}_by_agent{a_i}"]
                        )
            for k, v in event_dict.items():
                event_dict[k] = np.mean(v)
            events[exp_name] = event_dict

    logger.info(f"size {len(exclude)} {exclude}")
    logger.info(f"exp num {len(events.keys())}")
    for e in exclude:
        for k in list(events.keys()):
            if e == k:
                events.pop(k)
    logger.info(f"filtered exp num {len(events.keys())}")

    exps, metric_np, df = compute_metric(events, event_types, args.num_agents)
    df.to_excel(f"{eval_result_dir}/event_count_{policy_version}.xlsx", sheet_name="Events")

    runs = select_policies(exps, metric_np, K)
    logger.success(f"selected runs: {runs}")

    # generate HSP training config
    os.makedirs(f"{args.policy_pool_path}/{layout}/hsp/s2", exist_ok=True)
    mep_exp = MEP_EXPS[args.s]
    for seed in range(1, 6):
        with open(
            f"{args.policy_pool_path}/{layout}/hsp/s2/train-s{args.S}-{args.bias_agent_version}_{mep_exp}-{seed}.yml",
            "w",
        ) as f:
            f.write(
                f"""\
hsp_adaptive:
    policy_config_path: {layout}/policy_config/rnn_policy_config.pkl
    featurize_type: ppo
    train: True
"""
            )
            assert (args.S - int(K)) % 3 == 0, (args.S, K)
            POP_SIZE = (args.S - int(K)) // 3
            TOTAL_SIZE = args.s
            for p_i in range(1, POP_SIZE + 1):
                pt_i = (TOTAL_SIZE // 5 * (seed - 1) + p_i - 1) % TOTAL_SIZE + 1
                f.write(
                    f"""\
mep{p_i}_1:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {os.path.join(layout, "mep", "s1", mep_exp, f"mep{pt_i}_init_actor.pt")}
mep{p_i}_2:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {os.path.join(layout, "mep", "s1", mep_exp, f"mep{pt_i}_mid_actor.pt")}
mep{p_i}_3:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {os.path.join(layout, "mep", "s1", mep_exp, f"mep{pt_i}_final_actor.pt")}
"""
                )
            for i, run_i in enumerate(runs):
                f.write(
                    f"""\
hsp{i+1}_final:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/hsp/s1/{policy_version}/hsp{run_i}_final_w0_actor.pt\n"""
                )
