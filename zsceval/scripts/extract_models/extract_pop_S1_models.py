import os
import socket
import sys

import numpy as np
import wandb
from loguru import logger

wandb_name = "hogebein"
POLICY_POOL_PATH = "../policy_pool"


def extract_pop_S1_models(layout, algo, exp, env):
    api = wandb.Api()
    if "overcooked" in env.lower():
        layout_config = "config.layout_name"
    else:
        layout_config = "config.scenario_name"
    runs = api.runs(
        f"{wandb_name}/{env}",
        filters={
            "$and": [
                {"config.experiment_name": exp},
                {layout_config: layout},
                {"state": "finished"},
                {"tags": {"$nin": ["hidden", "unused"]}},
            ]
        },
        order="+config.seed",
    )
    runs = list(runs)
    run_ids = [r.id for r in runs]
    logger.info(f"num of runs: {len(runs)}")
    for i, run_id in enumerate(run_ids):
        run = runs[i]
        seed = run.config["seed"]
        if run.state == "finished":
            logger.info(f"Run: {run_id} Seed: {seed}")
            files = run.files()
            for policy_id in range(1, run.config["population_size"] + 1):
                policy_name = f"{algo}{policy_id}"
                history = run.history()
                ep_name = "-".join([policy_name] * run.config["num_agents"])
                history = history[["_step", f"{ep_name}-ep_sparse_r"]]

                steps = history["_step"].to_numpy().astype(int)
                ep_sparse_r = history[f"{ep_name}-ep_sparse_r"].to_numpy()
                final_ep_sparse_r = np.mean(ep_sparse_r[-5:])
                logger.info(f"{policy_name} Run: {run_id} Return: {final_ep_sparse_r}")
                actor_pts = [f for f in files if f.name.startswith(f"{policy_name}/actor_periodic")]
                actor_versions = [eval(f.name.split("_")[-1].split(".pt")[0]) for f in actor_pts]
                actor_pts = {v: p for v, p in zip(actor_versions, actor_pts)}
                actor_versions = sorted(actor_versions)
                max_actor_versions = max(actor_versions) + 1
                max_steps = max(steps)
                ep_sparse_r = [0 if np.isnan(x) else x for x in ep_sparse_r]

                new_steps = [steps[0]]
                new_ep_sparse_r = [ep_sparse_r[0]]
                for s, er in zip(steps[1:], ep_sparse_r[1:]):
                    l_s = new_steps[-1]
                    l_er = new_ep_sparse_r[-1]
                    for w in range(l_s + 1, s, 100):
                        new_steps.append(w)
                        new_ep_sparse_r.append(l_er + (er - l_er) * (w - l_s) / (s - l_s))
                steps = new_steps
                ep_sparse_r = new_ep_sparse_r

                # select checkpoints
                selected_pts = dict(init=0, mid=-1, final=max_steps)
                mid_ep_sparse_r = final_ep_sparse_r / 2
                min_delta = 1e9
                for s, score in zip(steps, ep_sparse_r):
                    if min_delta > abs(mid_ep_sparse_r - score):
                        min_delta = abs(mid_ep_sparse_r - score)
                        selected_pts["mid"] = s

                selected_pts = {k: int(v / max_steps * max_actor_versions) for k, v in selected_pts.items()}
                for tag, exp_version in selected_pts.items():
                    version = actor_versions[0]
                    for actor_version in actor_versions:
                        if abs(exp_version - version) > abs(exp_version - actor_version):
                            version = actor_version
                    logger.info(f"{policy_name} {tag} Expected: {exp_version} Found {version}")
                    ckpt = actor_pts[version]
                    tmp_dir = f"tmp/{layout}/{exp}"
                    ckpt.download(tmp_dir, replace=True)
                    pop_s1_dir = f"{POLICY_POOL_PATH}/{layout}/{algo}/s1"
                    pop_s1_path = f"{pop_s1_dir}/{exp}/{policy_name}_{tag}_actor.pt"
                    os.makedirs(f"{pop_s1_dir}/{exp}", exist_ok=True)
                    logger.success(f"pt store in {pop_s1_path}")
                    os.system(f"mv {tmp_dir}/{policy_name}/actor_periodic_{version}.pt {pop_s1_path}")


if __name__ == "__main__":
    layout = sys.argv[1]
    env = sys.argv[2]
    assert layout in [
        "random0",
        "random0_medium",
        "random1",
        "random3",
        "small_corridor",
        "unident_s",
        "random0_m",
        "random1_m",
        "random3_m",
        "inverse_marshmallow_experiment",
        "subobjective",
        "academy_3_vs_1_with_keeper",
        "all",
    ], layout
    if layout == "all":
        layout = [
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
        layout = [layout]
    hostname = socket.gethostname()
    algo_exp_names = [

        #("mep", "mep-S1-s510"),
        ("mep", "mep-S1-s5"),
        # ("traj", "traj-S1-s10"),
        #("traj", "traj-S1-s15"),

    ]

    logger.info(f"hostname: {hostname}")
    for l in layout:
        for algo, exp in algo_exp_names:
            logger.info(f"Extracting {algo} {exp} for {l}")
            extract_pop_S1_models(l, algo, exp, env)
