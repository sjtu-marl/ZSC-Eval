#!/usr/bin/env python
import os
import pprint
import socket
import sys
from itertools import product
from pathlib import Path

import numpy as np
import setproctitle
import torch
import wandb
from loguru import logger

from zsceval.config import get_config
from zsceval.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocDummyBatchVecEnv
from zsceval.envs.overcooked.Overcooked_Env import Overcooked
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
from zsceval.overcooked_config import get_overcooked_args
from zsceval.utils.train_util import get_base_run_dir, setup_seed


def make_train_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.overcooked_version == "old":
                    env = Overcooked(all_args, run_dir)
                else:
                    env = Overcooked_new(all_args, run_dir)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        # return ShareSubprocVecEnv(
        #     [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
        # )
        return ShareSubprocDummyBatchVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)],
            all_args.dummy_batch_size,
        )


def make_eval_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.overcooked_version == "old":
                    env = Overcooked(all_args, run_dir, evaluation=True)
                else:
                    env = Overcooked_new(all_args, run_dir, evaluation=True)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocDummyBatchVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)],
            all_args.dummy_batch_size,
        )
        # return ShareSubprocVecEnv(
        #     [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
        # )


def parse_args(args, parser):
    parser = get_overcooked_args(parser)
    parser.add_argument(
        "--use_phi",
        default=False,
        action="store_true",
        help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.",
    )

    parser.add_argument("--use_task_v_out", default=False, action="store_true")
    # all_args = parser.parse_known_args(args)[0]
    all_args = parser.parse_args(args)

    from zsceval.overcooked_config import OLD_LAYOUTS

    if all_args.layout_name in OLD_LAYOUTS:
        all_args.old_dynamics = True
    else:
        all_args.old_dynamics = False
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
            all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    # rpg
    if all_args.use_hsp:

        def parse_value(s):
            if s.startswith("r"):
                if "[" in s:
                    s = s[2:-1]
                    l, r, n = s.split(":")
                    l, r, n = float(l), float(r), int(n)
                    # return np.random.choice(np.linspace(l, r, n))
                    return np.linspace(l, r, n).tolist()
            elif s.startswith("["):
                s = s[1:-1]
                v_s = list(map(float, s.split(":")))
                return v_s
                # return np.random.choice(v_s)
            return [float(s)]

        # compute all w0 candidates

        w0 = []
        bias_index = []
        for s_i, s in enumerate(all_args.w0.split(",")):
            s = parse_value(s)
            w0.append(s)
            if len(s) > 1:
                bias_index.append(s_i)
        bias_index = np.array(bias_index)
        w0_candidates = list(map(list, product(*w0)))
        w0_candidates = [cand for cand in w0_candidates if sum(np.array(cand)[bias_index] != 0) <= 3]
        logger.info(f"bias index {bias_index}")
        logger.info(f"num w0_candidates {len(w0_candidates)}")
        candidates_str = ""
        for c_i in range(len(w0_candidates)):
            candidates_str += f"{c_i+1}: {w0_candidates[c_i]}\n"
        # logger.info(
        #     f"w0_candidates:\n {pprint.pformat(w0_candidates, width=150, compact=True)}"
        # )
        logger.info(f"w0_candidates:\n{candidates_str}")
        w0 = w0_candidates[(all_args.seed + all_args.w0_offset) % len(w0_candidates)]
        all_args.w0 = ""
        for s in w0:
            all_args.w0 += str(s) + ","
        all_args.w0 = all_args.w0[:-1]

        w1 = []
        for s in all_args.w1.split(","):
            w1.append(parse_value(s))
        w1_candidates = list(map(list, product(*w1)))
        logger.debug(f"w1_candidates:\n {pprint.pformat(w1_candidates, compact=True, width=200)}")
        w1 = w1_candidates[(all_args.seed) % len(w1_candidates)]
        all_args.w1 = ""
        for s in w1:
            all_args.w1 += str(s) + ","
        all_args.w1 = all_args.w1[:-1]

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    base_run_dir = Path(get_base_run_dir())
    run_dir = (
        base_run_dir / all_args.env_name / all_args.layout_name / all_args.algorithm_name / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    all_args.run_dir = run_dir

    # wandb
    if all_args.overcooked_version == "new":
        project_name = all_args.env_name + "-new"
    else:
        project_name = all_args.env_name
    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=project_name,
            entity=all_args.wandb_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(all_args.seed),
            group=all_args.layout_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
            tags=all_args.wandb_tags,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "_"
        + str(all_args.layout_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    # torch.manual_seed(all_args.seed)
    # torch.cuda.manual_seed_all(all_args.seed)
    # np.random.seed(all_args.seed)
    setup_seed(all_args.seed)

    # env init
    envs = make_train_env(all_args, run_dir)
    eval_envs = make_eval_env(all_args, run_dir) if all_args.use_eval else None
    num_agents = all_args.num_agents

    logger.info(pprint.pformat(all_args.__dict__, compact=True))
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        from zsceval.runner.shared.overcooked_runner import OvercookedRunner as Runner
    else:
        from zsceval.runner.separated.overcooked_runner import (
            OvercookedRunner as Runner,
        )

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish(quiet=True)
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    # logger.add(sys.stdout, level="INFO")
    main(sys.argv[1:])
