#!/usr/bin/env python
import os
import socket
import sys
from argparse import Namespace
from pathlib import Path
from pprint import pformat

import setproctitle
import torch
import wandb
import yaml
from loguru import logger

from zsceval.config import get_config
from zsceval.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocDummyBatchVecEnv
from zsceval.envs.grf.grf_env import FootballEnv
from zsceval.envs.wrappers.env_policy import PartialPolicyEnv
from zsceval.grf_config import get_grf_args
from zsceval.utils.train_util import get_base_run_dir, setup_seed


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "GRF":
                env = FootballEnv(all_args, seed=all_args.seed * 50000 + rank * 10000, evaluation=True)
            else:
                raise NotImplementedError("Can not support the " + all_args.env_name + "environment.")
            return env

        return init_env

    # BUG: should be share
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocDummyBatchVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)],
            all_args.dummy_batch_size,
        )


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "GRF":
                env = FootballEnv(all_args, seed=all_args.seed * 50000 + rank * 10000)
                env = PartialPolicyEnv(all_args, env)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocDummyBatchVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)],
            all_args.dummy_batch_size,
        )


def parse_args(args, parser):
    parser = get_grf_args(parser)

    parser.add_argument("--use_advantage_prioritized_sampling", default=False, action="store_true")
    parser.add_argument("--uniform_sampling_repeat", default=0, type=int)
    parser.add_argument("--use_task_v_out", default=False, action="store_true")
    # population
    parser.add_argument(
        "--population_yaml_path",
        type=str,
        help="Path to yaml file that stores the population info.",
    )
    # mep
    parser.add_argument(
        "--stage",
        type=int,
        default=2,
        help="Stages of MEP training. 1 for Maximum-Entropy PBT. 2 for FCP-like training.",
    )
    parser.add_argument(
        "--mep_use_prioritized_sampling",
        default=False,
        action="store_true",
        help="Use prioritized sampling in MEP stage 2.",
    )
    parser.add_argument(
        "--mep_prioritized_alpha",
        type=float,
        default=3.0,
        help="Alpha used in softing prioritized sampling probability.",
    )
    # population
    parser.add_argument(
        "--population_size",
        type=int,
        default=5,
        help="Population size involved in training.",
    )
    parser.add_argument(
        "--adaptive_agent_name",
        type=str,
        required=True,
        help="Name of training policy at Stage 2.",
    )

    # train and eval batching
    parser.add_argument(
        "--train_env_batch",
        type=int,
        default=1,
        help="Number of parallel threads a policy holds",
    )
    parser.add_argument(
        "--eval_env_batch",
        type=int,
        default=1,
        help="Number of parallel threads a policy holds",
    )

    # fixed policy actions inside env threads
    parser.add_argument(
        "--use_policy_in_env",
        default=True,
        action="store_false",
        help="Use loaded policy to move in env threads.",
    )

    # eval with fixed policy
    parser.add_argument("--eval_policy", default="", type=str)
    parser.add_argument("--eval_result_path", default=None, type=str)

    # cole
    parser.add_argument(
        "--generation_interval",
        type=int,
        default=10,
        help="num of episodes for a generation",
    )
    parser.add_argument("--cole_ucb_factor", type=float, default=1.0)
    parser.add_argument("--prioritized_alpha", type=float, default=1)
    parser.add_argument("--num_generation", type=int, default=150)
    parser.add_argument("--population_play_ratio", type=float, default=3)
    parser.add_argument(
        "--init_agent_ratio",
        help="args.population_size // args.init_agent_ratio random agents will be stored",
        type=int,
        default=5,
    )

    # all_args = parser.parse_known_args(args)[0]
    all_args = parser.parse_args(args)
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    assert all_args.algorithm_name in ["cole"], all_args.algorithm_name
    assert all_args.num_generation >= all_args.population_size, (
        all_args.num_generation,
        all_args.population_size,
    )
    all_args.num_env_steps = (
        all_args.generation_interval * all_args.num_generation * all_args.n_rollout_threads * all_args.episode_length
    )
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        assert n_gpu == 1 or all_args.data_parallel
        print(f"choose to use {n_gpu} gpu...")
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
        base_run_dir / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    all_args.run_dir = run_dir
    project_name = all_args.env_name
    # wandb
    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=project_name,
            entity=all_args.wandb_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(all_args.seed),
            group=all_args.scenario_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
            tags=all_args.wandb_tags,
        )
        all_args.id = run.id
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
        all_args.id = curr_run

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

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    logger.info(pformat(all_args.__dict__, compact=True))
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
        from zsceval.runner.shared.grf_runner import GRFRunner as Runner
    else:
        raise NotImplementedError

    runner = Runner(config)

    # load population
    # print("population_yaml_path: ", all_args.population_yaml_path)
    logger.info("population_yaml_path: ", all_args.population_yaml_path)

    #  override policy config
    population_config = yaml.load(open(all_args.population_yaml_path), yaml.Loader)

    logger.info(f"population_config: {population_config}")

    override_policy_config = {}
    # MARK: the ego agent
    agent_name = all_args.adaptive_agent_name
    #! overide policy and buffer config
    override_policy_config[agent_name] = (
        Namespace(
            use_agent_policy_id=all_args.use_agent_policy_id,
            num_v_out=all_args.num_v_out,
            use_task_v_out=all_args.use_task_v_out,
            use_policy_vhead=all_args.use_policy_vhead,
            use_proper_time_limits=all_args.use_proper_time_limits,
            entropy_coefs=all_args.entropy_coefs,
            entropy_coef_horizons=all_args.entropy_coef_horizons,
            use_peb=all_args.use_peb,
            data_parallel=all_args.data_parallel,
        ),
        *runner.policy_config[1:],
    )
    for policy_name in population_config:
        if policy_name != agent_name:
            override_policy_config[policy_name] = (
                Namespace(
                    entropy_coefs=all_args.entropy_coefs,
                    entropy_coef_horizons=all_args.entropy_coef_horizons,
                    use_proper_time_limits=all_args.use_proper_time_limits,
                    use_peb=all_args.use_peb,
                    data_parallel=all_args.data_parallel,
                ),
                None,
                runner.policy_config[2],
                None,
            )  # only override share_obs_space

    runner.policy.load_population(
        all_args.population_yaml_path,
        evaluation=False,
        override_policy_config=override_policy_config,
    )
    # MARK: init before train
    runner.trainer.init_population()

    runner.train_cole()

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
