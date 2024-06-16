import argparse
import glob
import os
import socket
import sys

import numpy as np
import wandb

wandb_name = "your wandb name"
POLICY_POOL_PATH = "../policy_pool"

from loguru import logger


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_target_index(array, percentile: float):
    """
    Find the index of the target value in the array based on the percentile.
    array: numpy array
    percentile: float, between 0 and 1
    return: index of the target value in the array
    """
    q3 = np.nanpercentile(array, int(percentile * 100))

    array_without_nan = array[~np.isnan(array)]

    index_of_max = np.nanargmax(array_without_nan)
    logger.debug(f"max index {index_of_max}/{len(array_without_nan)}")

    filtered_array = array_without_nan[index_of_max + 1 :]

    if filtered_array.size > 0:
        relative_index = (np.abs(filtered_array - q3)).argmin()
        original_index = np.where(array == filtered_array[relative_index])[0][0]

        return original_index, q3
    else:
        return len(array) - 1, np.nanmax(array)


def extract_S2_models(layout, algorithm, exp, population: str):
    population = population.split("-", 1)[1]
    if algorithm == "fcp":
        exp_name = f"{exp}-pop{population}-S2"
    elif algorithm == "mep":
        if len(exp.split("-")) == 2:
            exp_name = f"{exp.split('-')[0]}-{exp.split('-')[1]}-pop{population}-S2"
        else:
            exp_name = f"{exp}-pop{population}-S2"

    elif algorithm == "traj":
        exp_name = f"{exp}-pop{population}-S2"
    elif algorithm == "hsp":
        exp_name = f"{exp}-pop{population}-S2"
    elif algorithm == "cole":
        exp_name = f"{exp}-{population}"
    else:
        raise NotImplementedError

    logger.info(f"exp {exp_name}")
    api = wandb.Api()
    runs = api.runs(
        f"{wandb_name}/GRF",
        # f"{wandb_name}/Overcooked-new",
        filters={
            "$and": [
                {"config.experiment_name": exp_name},
                {"config.scenario_name": layout},
                # {"config.layout_name": layout},
                {"state": "finished"},
                {"tags": {"$nin": ["hidden", "unused"]}},
            ]
        },
        order="+config.seed",
    )
    if not exp_name.endswith("-S2"):
        exp_name += "-S2"
        exp_name = exp + "-pop_" + exp_name.split(exp + "-", 1)[1]
    runs = list(runs)
    run_ids = [r.id for r in runs]
    logger.info(f"num of runs: {len(runs)}")
    seeds = set()
    global percentile
    for i, run_id in enumerate(run_ids):
        run = runs[i]
        if run.state == "finished":
            policy_name = f"{algorithm}_adaptive"
            history = run.history()
            # history = history[["_step", f"train/{algorithm}_adaptive-average_episode_rewards"]]
            history = history[["_step", f"either-{algorithm}_adaptive-ep_sparse_r"]]
            steps = history["_step"].to_numpy().astype(int)
            # ep_sparse_r = history[f"train/{algorithm}_adaptive-average_episode_rewards"].to_numpy()
            ep_sparse_r = history[f"either-{algorithm}_adaptive-ep_sparse_r"].to_numpy()
            i_max_ep_sparse_r, max_ep_sparse_r = find_target_index(ep_sparse_r, percentile)
            max_ep_sparse_r_step = steps[i_max_ep_sparse_r]
            files = run.files()
            actor_pts = [f for f in files if f.name.startswith(f"{policy_name}/actor_periodic")]
            actor_versions = [int(f.name.split("_")[-1].split(".pt")[0]) for f in actor_pts]
            actor_versions.sort()
            version = find_nearest(actor_versions, max_ep_sparse_r_step)
            logger.info(
                f"actor version {version} / {actor_versions[-1]}, sparse_r {max_ep_sparse_r:.3f}/{np.nanmax(ep_sparse_r):.3f}"
            )
            ckpt = run.file(f"{policy_name}/actor_periodic_{version}.pt")
            tmp_dir = f"tmp/{layout}/{exp_name}"
            logger.info(f"Fetch {tmp_dir}/{policy_name}/actor_periodic_{version}.pt")
            ckpt.download(f"{tmp_dir}", replace=True)
            algo_s2_dir = f"{POLICY_POOL_PATH}/{layout}/{algorithm}/s2"
            seed = run.config["seed"]
            os.makedirs(f"{algo_s2_dir}/{exp_name}", exist_ok=True)
            os.system(f"mv {tmp_dir}/{policy_name}/actor_periodic_{version}.pt {algo_s2_dir}/{exp_name}/{seed}.pt")
            seeds.add(seed)
            logger.success(f"{layout} {algorithm} {exp_name} {seed}")


if __name__ == "__main__":
    layout = sys.argv[1]
    assert layout in [
        "academy_3_vs_1_with_keeper",
        "random0_m",
        "random1_m",
        "random3_m",
        "all",
    ]
    if layout == "all":
        layout = ["academy_3_vs_1_with_keeper"]
    else:
        layout = [layout]
    algorithm = sys.argv[2]
    if len(sys.argv) > 3:
        percentile = float(sys.argv[3])
    else:
        percentile = 0.8
    assert algorithm in ["traj", "mep", "fcp", "cole", "hsp"]
    algorithm = [algorithm]
    ALG_EXPS = {
        "fcp": {
            "fcp": ["-_s9"],
        },
        "mep": {
            "mep": [
                "-_s9_s5-S1",
            ],
        },
        "traj": {
            "traj": ["-_s9_s5-S1"],
        },
        "hsp": {
            "hsp": [
                "-_s9",
            ],
        },
        "cole": {
            "cole": ["-s15"],
        },
    }

    hostname = socket.gethostname()
    logger.add(f"./extract_log/extract_{layout}_{algorithm}_S2_models.log")
    logger.info(f"hostname: {hostname}")
    for l in layout:
        for algo in algorithm:
            logger.info(f"for layout {l}")
            for exp, pop_list in ALG_EXPS[algo].items():
                for pop in pop_list:
                    extract_S2_models(l, algo, exp, pop)
