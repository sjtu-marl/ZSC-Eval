#!/usr/bin/env python
import os
import socket
import sys
from pathlib import Path

import setproctitle
import torch
import wandb
from loguru import logger
from rich.pretty import pretty_repr

from zsceval.config import get_config
from zsceval.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocDummyBatchVecEnv
from zsceval.envs.grf.grf_env import FootballEnv
from zsceval.grf_config import get_grf_args
from zsceval.utils.train_util import setup_seed


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "GRF":
                env = FootballEnv(all_args, seed=all_args.seed * 50000 + rank * 10000, evaluation=True)
            else:
                raise NotImplementedError("Can not support the " + all_args.env_name + "environment.")
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocDummyBatchVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)],
            all_args.dummy_batch_size,
        )


def parse_args(args, parser):
    parser = get_grf_args(parser)
    parser.add_argument("--store_traj", default=False, action="store_true")
    # population
    parser.add_argument(
        "--population_yaml_path",
        type=str,
        help="Path to yaml file that stores the population info.",
    )

    # evaluation
    parser.add_argument("--agent_name", type=str, help="name of the agent to evaluate")
    parser.add_argument(
        "--population_size",
        type=int,
        default=11,
        help="Population size involved in training.",
    )

    # result
    parser.add_argument(
        "--eval_result_path",
        type=str,
        help="eval/results/{scenario}/{exp}",
        required=True,
    )

    all_args = parser.parse_args(args)

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    assert all_args.algorithm_name == "population"

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
    run_dir = (
        Path(os.path.expanduser("~") + "/ZSC/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    all_args.run_dir = run_dir

    eval_result_dir = Path(os.path.dirname(all_args.eval_result_path))

    if not eval_result_dir.exists():
        os.makedirs(str(eval_result_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.wandb_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(all_args.seed),
            group=all_args.scenario_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
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
        + str(all_args.scenario_name)
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

    logger.info(
        pretty_repr(
            {
                "Scenario": all_args.scenario_name,
                "Exp Name": str(all_args.algorithm_name)
                + "_"
                + str(all_args.experiment_name)
                + "_seed"
                + str(all_args.seed),
            }
        )
    )
    # env init
    eval_envs = make_eval_env(all_args)
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": eval_envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        from zsceval.runner.shared.grf_runner import GRFRunner as Runner
    else:
        raise NotImplementedError

    runner = Runner(config)

    # load population
    logger.info(f"population_yaml_path: {all_args.population_yaml_path}")
    runner.policy.load_population(all_args.population_yaml_path, evaluation=True)
    population_agents = [name for name, _, _, _ in runner.policy.all_policies() if all_args.agent_name not in name]
    combs = runner.get_agent_pairs(population_agents, all_args.agent_name)

    logger.debug(f"population {population_agents}")
    # logger.info(f"{len(combs)} pairs:\n{combs}")
    logger.info(f"{len(combs)} pairs")

    if all_args.n_eval_rollout_threads % len(combs) != 0:
        logger.warning(f"n_eval_rollout_threads should be multiples of {len(combs)}")
    assert all_args.eval_episodes % all_args.n_eval_rollout_threads == 0

    # configure mapping from (env_id, agent_id) to policy_name
    map_ea2p = dict()
    for e in range(all_args.n_eval_rollout_threads):
        comb = combs[e % len(combs)]
        for a_i in range(num_agents):
            map_ea2p[(e, a_i)] = comb[a_i]
    runner.policy.set_map_ea2p(map_ea2p)
    runner.population_size = all_args.population_size

    runner.evaluate_with_multi_policy()
    if eval_envs is not None:
        # post process
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    main(sys.argv[1:])
