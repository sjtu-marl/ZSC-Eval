import importlib
import os
import pickle
from collections.abc import Iterable

import numpy as np
import torch
import wandb
from loguru import logger
from tensorboardX import SummaryWriter

from zsceval.utils.shared_buffer import SharedReplayBuffer

webhook_url = "slack hook url"


def _t2n(x):
    return x.detach().cpu().numpy()


def make_trainer_policy_cls(algorithm_name, use_single_network=False):
    algorithm_dict = {
        "rmappo": (
            "zsceval.algorithms.r_mappo.r_mappo.R_MAPPO",
            "zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy.R_MAPPOPolicy",
        ),
        "mappo": (
            "zsceval.algorithms.r_mappo.r_mappo.R_MAPPO",
            "zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy.R_MAPPOPolicy",
        ),
        "population": (
            "zsceval.algorithms.population.trainer_pool.TrainerPool",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "mep": (
            "zsceval.algorithms.population.mep.MEP_Trainer",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "adaptive": (
            "zsceval.algorithms.population.mep.MEP_Trainer",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "cole": (
            "zsceval.algorithms.population.cole.COLE_Trainer",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "traj": (
            "zsceval.algorithms.population.traj.Traj_Trainer",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
    }

    if algorithm_name not in algorithm_dict:
        raise NotImplementedError

    train_algo_module, train_algo_class = algorithm_dict[algorithm_name][0].rsplit(".", 1)
    policy_module, policy_class = algorithm_dict[algorithm_name][1].rsplit(".", 1)

    TrainAlgo = getattr(importlib.import_module(train_algo_module), train_algo_class)
    Policy = getattr(importlib.import_module(policy_module), policy_class)

    return TrainAlgo, Policy


class Runner:
    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]
        if config.__contains__("render_envs"):
            self.render_envs = config["render_envs"]

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_single_network = self.all_args.use_single_network
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / "gifs")
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / "logs")
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / "models")
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        TrainAlgo, Policy = make_trainer_policy_cls(self.algorithm_name, use_single_network=self.use_single_network)

        share_observation_space = (
            self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        )

        # policy network
        self.policy = Policy(
            self.all_args,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0],
            device=self.device,
        )

        logger.info(
            f"Action space {self.envs.action_space[0]}, Obs space {self.envs.observation_space[0].shape}, Share obs space {share_observation_space.shape}"
        )

        # dump policy config to allow loading population in yaml form
        self.policy_config = (
            self.all_args,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0],
        )
        policy_config_path = os.path.join(self.run_dir, "policy_config.pkl")
        pickle.dump(self.policy_config, open(policy_config_path, "wb"))
        print(f"Pickle dump policy config at {policy_config_path}")
        if "store" in self.experiment_name:
            exit()

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)

        # buffer
        if self.algorithm_name != "population":
            # population-based trainer creates buffer inside trainer
            self.buffer = SharedReplayBuffer(
                self.all_args,
                self.num_agents,
                self.envs.observation_space[0],
                share_observation_space,
                self.envs.action_space[0],
            )

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self, num_steps: int = 0):
        self.trainer.prep_training()
        self.trainer.adapt_entropy_coef(num_steps)
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        self.log_system()
        return train_infos

    def save(self, save_critic: bool = False):
        if self.use_single_network:
            policy_model = self.trainer.policy.model
            torch.save(policy_model.state_dict(), str(self.save_dir) + "/model.pt")
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
            if save_critic:
                policy_critic = self.trainer.policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self):
        if self.use_single_network:
            policy_model_state_dict = torch.load(str(self.model_dir) + "/model.pt", map_location=self.device)
            self.policy.model.load_state_dict(policy_model_state_dict)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir) + "/actor.pt", map_location=self.device)
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not (self.all_args.use_render or self.all_args.use_eval):
                policy_critic_state_dict = torch.load(str(self.model_dir) + "/critic.pt", map_location=self.device)
                self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            if isinstance(v, Iterable):
                if len(v) == 0:
                    continue
                v = np.mean(v)
            if self.use_wandb:
                wandb.log({f"train/{k}": v}, step=total_num_steps)
            else:
                self.writter.add_scalars(f"train/{k}", {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if isinstance(v, Iterable):
                if len(v) == 0:
                    continue
                v = np.mean(v)
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_system(self):
        pass
        # RRAM
        # mem = psutil.virtual_memory()
        # total_mem = float(mem.total) / 1024 / 1024 / 1024
        # used_mem = float(mem.used) / 1024 / 1024 / 1024
        # if used_mem / total_mem > 0.95:
        #     slack = slackweb.Slack(url=webhook_url)
        #     host_name = socket.gethostname()
        #     slack.notify(
        #         text="Host {}: occupied memory is *{:.2f}*%!".format(
        #             host_name, used_mem / total_mem * 100
        #         )
        #     )
