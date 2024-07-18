import copy
import itertools
import json
import pprint
import time
from collections import defaultdict
from os import path as osp
from typing import Dict

import numpy as np
import torch
import wandb
from loguru import logger
from scipy.stats import rankdata
from tqdm import tqdm

from zsceval.runner.shared.base_runner import Runner
from zsceval.utils.log_util import eta, get_table_str


def _t2n(x):
    return x.detach().cpu().numpy()


class OvercookedRunner(Runner):
    """
    A wrapper to start the RL agent training algorithm.
    """

    def __init__(self, config):
        super().__init__(config)

        # for training br
        self.br_best_sparse_r = 0
        self.br_eval_json = {}

    def run(self):
        # train sp
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0

        for episode in range(episodes):
            s_time = time.time()
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)

                # Obser reward and next obs
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                obs = np.stack(obs)
                total_num_steps += self.n_rollout_threads
                self.envs.anneal_reward_shaping_factor([total_num_steps] * self.n_rollout_threads)
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)
            e_time = time.time()
            logger.trace(f"Rollout time: {e_time - s_time:.3f}s")

            # compute return and update network
            s_time = time.time()
            self.compute()
            train_infos = self.train(total_num_steps)
            e_time = time.time()
            logger.trace(f"Update models time: {e_time - s_time:.3f}s")

            # post process
            s_time = time.time()
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode < 50:
                if episode % 2 == 0:
                    # self.save(episode)
                    self.save(total_num_steps)
            elif episode < 100:
                if episode % 5 == 0:
                    self.save(total_num_steps)
                    # self.save(episode)
            else:
                if episode % self.save_interval == 0 or episode == episodes - 1:
                    self.save(total_num_steps)
                    # self.save(episode)

            # log information
            if episode % self.log_interval == 0 or episode == episodes - 1:
                end = time.time()
                eta_t = eta(start, end, self.num_env_steps, total_num_steps)
                log_data = list(
                    {
                        "Layout": self.all_args.layout_name,
                        "Algorithm": self.algorithm_name,
                        "Experiment": self.experiment_name,
                        "Seed": self.all_args.seed,
                        "Episodes": episode,
                        "Total Episodes": episodes,
                        "Timesteps": total_num_steps,
                        "Total Timesteps": self.num_env_steps,
                        "FPS": int(total_num_steps / (end - start)),
                        "ETA": eta_t,
                    }.items()
                )
                logger.info("training process:\n" + get_table_str(log_data))
                logger.info(
                    "Layout {} Algo {} Exp {} Seed {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}, ETA {}.".format(
                        self.all_args.layout_name,
                        self.algorithm_name,
                        self.experiment_name,
                        self.all_args.seed,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                        eta_t,
                    )
                )
                # shaped reward
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                logger.info("average episode rewards is {:.3f}".format(train_infos["average_episode_rewards"]))

                env_infos = defaultdict(list)
                if self.env_name == "Overcooked":
                    if self.all_args.overcooked_version == "old":
                        from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
                            SHAPED_INFOS,
                        )

                        shaped_info_keys = SHAPED_INFOS
                    else:
                        from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
                            SHAPED_INFOS,
                        )

                        shaped_info_keys = SHAPED_INFOS
                    for info in infos:
                        for a in range(self.num_agents):
                            env_infos[f"ep_sparse_r_by_agent{a}"].append(info["episode"]["ep_sparse_r_by_agent"][a])
                            env_infos[f"ep_shaped_r_by_agent{a}"].append(info["episode"]["ep_shaped_r_by_agent"][a])
                            for i, k in enumerate(shaped_info_keys):
                                env_infos[f"ep_{k}_by_agent{a}"].append(info["episode"]["ep_category_r_by_agent"][a][i])
                        env_infos["ep_sparse_r"].append(info["episode"]["ep_sparse_r"])
                        env_infos["ep_shaped_r"].append(info["episode"]["ep_shaped_r"])
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                if self.use_wandb:
                    wandb.log({"train/ETA": eta_t}, step=total_num_steps)
                logger.info(f'average sparse rewards is {np.mean(env_infos["ep_sparse_r"]):.3f}')
            # eval
            if episode % self.eval_interval == 0 and self.use_eval or episode == episodes - 1:
                self.eval(total_num_steps)
            e_time = time.time()
            logger.trace(f"Post update models time: {e_time - s_time:.3f}s")

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        obs = np.stack(obs)

        # replay buffer
        if self.use_centralized_V:
            share_obs = share_obs
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(self.buffer.available_actions[step]),
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = share_obs
        else:
            share_obs = obs

        bad_masks = np.array([[[0.0] if info["bad_transition"] else [1.0]] * self.num_agents for info in infos])

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            bad_masks=bad_masks,
            available_actions=available_actions,
        )

    def restore(self):
        if self.use_single_network:
            policy_model_state_dict = torch.load(str(self.model_dir) + "/model.pt", map_location=self.device)
            self.policy.model.load_state_dict(policy_model_state_dict)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir), map_location=self.device)
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not (self.all_args.use_render or self.all_args.use_eval):
                policy_critic_state_dict = torch.load(str(self.model_dir) + "/critic.pt", map_location=self.device)
                self.policy.critic.load_state_dict(policy_critic_state_dict)

    def save(self, step, save_critic: bool = False):
        # logger.info(f"save sp periodic_{step}.pt")
        if self.use_single_network:
            policy_model = self.trainer.policy.model
            torch.save(
                policy_model.state_dict(),
                str(self.save_dir) + f"/model_periodic_{step}.pt",
            )
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(
                policy_actor.state_dict(),
                str(self.save_dir) + f"/actor_periodic_{step}.pt",
            )
            if save_critic:
                policy_critic = self.trainer.policy.critic
                torch.save(
                    policy_critic.state_dict(),
                    str(self.save_dir) + f"/critic_periodic_{step}.pt",
                )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_env_infos = defaultdict(list)
        if self.env_name == "Overcooked":
            if self.all_args.overcooked_version == "old":
                from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
                    SHAPED_INFOS,
                )

                shaped_info_keys = SHAPED_INFOS
            else:
                from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
                    SHAPED_INFOS,
                )

                shaped_info_keys = SHAPED_INFOS
        eval_average_episode_rewards = []
        eval_obs, _, eval_available_actions = self.eval_envs.reset()
        eval_obs = np.stack(eval_obs)

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for _ in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                np.concatenate(eval_available_actions),
                deterministic=not self.all_args.eval_stochastic,
            )
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            (
                eval_obs,
                _,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            eval_obs = np.stack(eval_obs)
            eval_average_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        for eval_info in eval_infos:
            for a in range(self.num_agents):
                eval_env_infos[f"eval_ep_sparse_r_by_agent{a}"].append(eval_info["episode"]["ep_sparse_r_by_agent"][a])
                eval_env_infos[f"eval_ep_shaped_r_by_agent{a}"].append(eval_info["episode"]["ep_shaped_r_by_agent"][a])
                for i, k in enumerate(shaped_info_keys):
                    eval_env_infos[f"eval_ep_{k}_by_agent{a}"].append(
                        eval_info["episode"]["ep_category_r_by_agent"][a][i]
                    )
            eval_env_infos["eval_ep_sparse_r"].append(eval_info["episode"]["ep_sparse_r"])
            eval_env_infos["eval_ep_shaped_r"].append(eval_info["episode"]["ep_shaped_r"])

        eval_env_infos["eval_average_episode_rewards"] = np.sum(eval_average_episode_rewards, axis=0)
        logger.success(
            f'eval average sparse rewards {np.mean(eval_env_infos["eval_ep_sparse_r"]):.3f} {len(eval_env_infos["eval_ep_sparse_r"])} episodes, total num timesteps {total_num_steps}/{self.num_env_steps}'
        )
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        envs = self.envs
        obs, _, available_actions = envs.reset()
        obs = np.stack(obs)

        for episode in tqdm(range(self.all_args.render_episodes)):
            rnn_states = np.zeros(
                (self.n_rollout_threads, *self.buffer.rnn_states.shape[2:]),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []
            for step in range(self.episode_length):
                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    np.concatenate(available_actions),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
                # Obser reward and next obs
                obs, _, rewards, dones, infos, available_actions = envs.step(actions)
                obs = np.stack(obs)

                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

            logger.info("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

    def evaluate_one_episode_with_multi_policy(self, policy_pool: Dict, map_ea2p: Dict):
        """Evaluate one episode with different policy for each agent.
        Params:
            policy_pool (Dict): a pool of policies. Each policy should support methods 'step' that returns actions given observation while maintaining hidden states on its own, and 'reset' that resets the hidden state.
            map_ea2p (Dict): a mapping from (env_id, agent_id) to policy name
        """
        # warnings.warn("Evaluation with multi policy is not compatible with async done.")
        [policy.reset(self.n_eval_rollout_threads, self.num_agents) for _, policy in policy_pool.items()]
        for e in range(self.n_eval_rollout_threads):
            for agent_id in range(self.num_agents):
                if not map_ea2p[(e, agent_id)].startswith("script:"):
                    policy_pool[map_ea2p[(e, agent_id)]].register_control_agent(e, agent_id)
        if self.all_args.algorithm_name == "cole":
            c_a_str = {
                p_name: len(policy_pool[p_name].control_agents)
                for p_name in self.generated_population_names + [self.trainer.agent_name]
            }
            logger.debug(f"control agents num:\n{c_a_str}")

        eval_env_infos = defaultdict(list)
        eval_obs, _, eval_available_actions = self.eval_envs.reset()

        extract_info_keys = []  # ['stuck', 'can_begin_cook']
        infos = None
        for _ in range(self.all_args.episode_length):
            eval_actions = np.full((self.n_eval_rollout_threads, self.num_agents, 1), fill_value=0).tolist()
            for _, policy in policy_pool.items():
                if len(policy.control_agents) > 0:
                    policy.prep_rollout()
                    policy.to(self.device)
                    obs_lst = [eval_obs[e][a] for (e, a) in policy.control_agents]
                    avail_action_lst = [eval_available_actions[e][a] for (e, a) in policy.control_agents]
                    info_lst = None
                    if infos is not None:
                        info_lst = {k: [infos[e][k][a] for e, a in policy.control_agents] for k in extract_info_keys}
                    agents = policy.control_agents
                    actions = policy.step(
                        np.stack(obs_lst, axis=0),
                        agents,
                        info=info_lst,
                        deterministic=not self.all_args.eval_stochastic,
                        available_actions=np.stack(avail_action_lst),
                    )
                    for action, (e, a) in zip(actions, agents):
                        eval_actions[e][a] = action
            # Observe reward and next obs
            eval_actions = np.array(eval_actions)
            (
                eval_obs,
                _,
                _,
                _,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)

            infos = eval_infos

        if self.all_args.overcooked_version == "old":
            from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
                SHAPED_INFOS,
            )

            shaped_info_keys = SHAPED_INFOS
        else:
            from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
                SHAPED_INFOS,
            )

            shaped_info_keys = SHAPED_INFOS

        for eval_info in eval_infos:
            for a in range(self.num_agents):
                for i, k in enumerate(shaped_info_keys):
                    eval_env_infos[f"eval_ep_{k}_by_agent{a}"].append(
                        eval_info["episode"]["ep_category_r_by_agent"][a][i]
                    )
                eval_env_infos[f"eval_ep_sparse_r_by_agent{a}"].append(eval_info["episode"]["ep_sparse_r_by_agent"][a])
                eval_env_infos[f"eval_ep_shaped_r_by_agent{a}"].append(eval_info["episode"]["ep_shaped_r_by_agent"][a])
            eval_env_infos["eval_ep_sparse_r"].append(eval_info["episode"]["ep_sparse_r"])
            eval_env_infos["eval_ep_shaped_r"].append(eval_info["episode"]["ep_shaped_r"])

        return eval_env_infos

    def evaluate_with_multi_policy(self, policy_pool=None, map_ea2p=None, num_eval_episodes=None):
        """Evaluate with different policy for each agent."""
        policy_pool = policy_pool or self.policy.policy_pool
        map_ea2p = map_ea2p or self.policy.map_ea2p
        num_eval_episodes = num_eval_episodes or self.all_args.eval_episodes
        logger.debug(f"evaluate {self.population_size} policies with {num_eval_episodes} episodes")
        eval_infos = defaultdict(list)
        for _ in tqdm(
            range(max(1, num_eval_episodes // self.n_eval_rollout_threads)),
            desc="Evaluate with Population",
        ):
            eval_env_info = self.evaluate_one_episode_with_multi_policy(policy_pool, map_ea2p)
            for k, v in eval_env_info.items():
                for e in range(self.n_eval_rollout_threads):
                    agent0, agent1 = map_ea2p[(e, 0)], map_ea2p[(e, 1)]
                    for log_name in [
                        f"{agent0}-{agent1}-{k}",
                    ]:
                        if k in ["eval_ep_sparse_r", "eval_ep_shaped_r"]:
                            eval_infos[log_name].append(v[e])
                        elif (
                            getattr(self.all_args, "stage", 1) == 1
                            or not self.all_args.use_wandb
                            or ("br" in self.trainer.agent_name)
                        ):
                            eval_infos[log_name].append(v[e])

                    if k in ["eval_ep_sparse_r", "eval_ep_shaped_r"]:
                        for log_name in [
                            f"either-{agent0}-{k}",
                            f"either-{agent0}-{k}-as_agent_0",
                            f"either-{agent1}-{k}",
                            f"either-{agent1}-{k}-as_agent_1",
                        ]:
                            eval_infos[log_name].append(v[e])

        logger.success(
            "eval average sparse rewards:\n{}".format(
                pprint.pformat(
                    {
                        k: f"{np.mean(v):.2f}"
                        for k, v in eval_infos.items()
                        if "ep_sparse_r" in k and "by_agent" not in k
                    },
                    compact=True,
                    width=200,
                )
            )
        )

        eval_infos2dump = {k: np.mean(v) for k, v in eval_infos.items()}

        if hasattr(self.trainer, "agent_name"):
            br_sparse_r = f"either-{self.trainer.agent_name}-eval_ep_sparse_r"
            br_sparse_r = np.mean(eval_infos[br_sparse_r])

            if br_sparse_r >= self.br_best_sparse_r:
                self.br_best_sparse_r = br_sparse_r
                logger.success(
                    f"best eval br sparse reward {self.br_best_sparse_r:.2f} at {self.total_num_steps} steps"
                )
                self.br_eval_json = copy.deepcopy(eval_infos2dump)

                if getattr(self.all_args, "eval_result_path", None):
                    logger.debug(f"dump eval_infos to {self.all_args.eval_result_path}")
                    with open(self.all_args.eval_result_path, "w", encoding="utf-8") as f:
                        json.dump(self.br_eval_json, f)
        elif getattr(self.all_args, "eval_result_path", None):
            logger.debug(f"dump eval_infos to {self.all_args.eval_result_path}")
            with open(self.all_args.eval_result_path, "w", encoding="utf-8") as f:
                json.dump(eval_infos2dump, f)

        return eval_infos

    def naive_train_with_multi_policy(self, reset_map_ea2t_fn=None, reset_map_ea2p_fn=None):
        """This is a naive training loop using TrainerPool and PolicyPool.

        To use PolicyPool and TrainerPool, you should first initialize population in policy_pool, with either:
        >>> self.policy.load_population(population_yaml_path)
        >>> self.trainer.init_population()
        or:
        >>> # mannually register policies
        >>> self.policy.register_policy(policy_name="ppo1", policy=rMAPPOpolicy(args, obs_space, share_obs_space, act_space), policy_config=(args, obs_space, share_obs_space, act_space), policy_train=True)
        >>> self.policy.register_policy(policy_name="ppo2", policy=rMAPPOpolicy(args, obs_space, share_obs_space, act_space), policy_config=(args, obs_space, share_obs_space, act_space), policy_train=True)
        >>> self.trainer.init_population()

        To bind (env_id, agent_id) to different trainers and policies:
        >>> map_ea2t = {(e, a): "ppo1" if a == 0 else "ppo2" for e in range(self.n_rollout_threads) for a in range(self.num_agents)}
        # Qs: 2p? n_eval_rollout_threads?
        >>> map_ea2p = {(e, a): "ppo1" if a == 0 else "ppo2" for e in range(self.n_eval_rollout_threads) for a in range(self.num_agents)}
        >>> self.trainer.set_map_ea2t(map_ea2t)
        >>> self.policy.set_map_ea2p(map_ea2p)

        # MARK
        Note that map_ea2t is for training while map_ea2p is for policy evaluations

        WARNING: Currently do not support changing map_ea2t and map_ea2p when training. To implement this, we should take the first obs of next episode in the previous buffers and feed into the next buffers.
        """

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0
        env_infos = defaultdict(list)
        self.eval_info = dict()
        self.env_info = dict()

        for episode in range(0, episodes):
            self.total_num_steps = total_num_steps
            if self.use_linear_lr_decay:
                self.trainer.lr_decay(episode, episodes)

            # reset env agents
            if reset_map_ea2t_fn is not None:
                map_ea2t = reset_map_ea2t_fn(episode)
                self.trainer.reset(
                    map_ea2t,
                    self.n_rollout_threads,
                    self.num_agents,
                    load_unused_to_cpu=True,
                )
                if self.all_args.use_policy_in_env:
                    load_policy_cfg = np.full((self.n_rollout_threads, self.num_agents), fill_value=None).tolist()
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            trainer_name = map_ea2t[(e, a)]
                            if trainer_name not in self.trainer.on_training:
                                load_policy_cfg[e][a] = self.trainer.policy_pool.policy_info[trainer_name]
                    self.envs.load_policy(load_policy_cfg)

            # init env
            obs, share_obs, available_actions = self.envs.reset()

            # replay buffer
            if self.use_centralized_V:
                share_obs = share_obs
            else:
                share_obs = obs

            s_time = time.time()
            self.trainer.init_first_step(share_obs, obs, available_actions)

            for step in range(self.episode_length):
                # Sample actions
                actions = self.trainer.step(step)

                # Observe reward and next obs
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                total_num_steps += self.n_rollout_threads
                self.envs.anneal_reward_shaping_factor(self.trainer.reward_shaping_steps())

                bad_masks = np.array([[[0.0] if info["bad_transition"] else [1.0]] * self.num_agents for info in infos])

                self.trainer.insert_data(
                    share_obs,
                    obs,
                    rewards,
                    dones,
                    bad_masks=bad_masks,
                    infos=infos,
                    available_actions=available_actions,
                )

            # update env infos
            episode_env_infos = defaultdict(list)
            ep_returns_per_trainer = defaultdict(lambda: [[] for _ in range(self.num_agents)])
            e2ta = dict()
            if self.env_name == "Overcooked":
                if self.all_args.overcooked_version == "old":
                    from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
                        SHAPED_INFOS,
                    )

                    shaped_info_keys = SHAPED_INFOS
                else:
                    from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
                        SHAPED_INFOS,
                    )

                    shaped_info_keys = SHAPED_INFOS
                for e, info in enumerate(infos):
                    agent0_trainer = self.trainer.map_ea2t[(e, 0)]
                    agent1_trainer = self.trainer.map_ea2t[(e, 1)]
                    for log_name in [
                        f"{agent0_trainer}-{agent1_trainer}",
                    ]:
                        episode_env_infos[f"{log_name}-ep_sparse_r"].append(info["episode"]["ep_sparse_r"])
                        episode_env_infos[f"{log_name}-ep_shaped_r"].append(info["episode"]["ep_shaped_r"])
                        for a in range(self.num_agents):
                            if getattr(self.all_args, "stage", 1) == 1 or not self.all_args.use_wandb:
                                for i, k in enumerate(shaped_info_keys):
                                    episode_env_infos[f"{log_name}-ep_{k}_by_agent{a}"].append(
                                        info["episode"]["ep_category_r_by_agent"][a][i]
                                    )
                            episode_env_infos[f"{log_name}-ep_sparse_r_by_agent{a}"].append(
                                info["episode"]["ep_sparse_r_by_agent"][a]
                            )
                            episode_env_infos[f"{log_name}-ep_shaped_r_by_agent{a}"].append(
                                info["episode"]["ep_shaped_r_by_agent"][a]
                            )
                    for k in ["ep_sparse_r", "ep_shaped_r"]:
                        for log_name in [
                            f"either-{agent0_trainer}-{k}",
                            f"either-{agent0_trainer}-{k}-as_agent_0",
                            f"either-{agent1_trainer}-{k}",
                            f"either-{agent1_trainer}-{k}-as_agent_1",
                        ]:
                            episode_env_infos[log_name].append(info["episode"][k])
                    if agent0_trainer != self.trainer.agent_name:
                        # suitable for both stage 1 and stage 2
                        ep_returns_per_trainer[agent1_trainer][1].append(info["episode"]["ep_sparse_r"])
                        e2ta[e] = (agent1_trainer, 1)
                    elif agent1_trainer != self.trainer.agent_name:
                        ep_returns_per_trainer[agent0_trainer][0].append(info["episode"]["ep_sparse_r"])
                        e2ta[e] = (agent0_trainer, 0)
                env_infos.update(episode_env_infos)
            max_ep_sparse_r_dict = defaultdict(lambda: [0, 0])

            self.env_info.update(env_infos)
            e_time = time.time()
            logger.trace(f"Rollout time: {e_time - s_time:.3f}s")

            # compute return and update network
            s_time = time.time()
            if self.all_args.stage == 1:
                self.trainer.adapt_entropy_coef(total_num_steps // self.population_size)
            else:
                self.trainer.adapt_entropy_coef(total_num_steps)

            train_infos = self.trainer.train(sp_size=getattr(self, "n_repeats", 0) * self.num_agents)
            e_time = time.time()
            logger.trace(f"Update models time: {e_time - s_time:.3f}s")

            s_time = time.time()
            if self.all_args.stage == 2:
                # update advantage moving average, used in stage2
                if self.all_args.use_advantage_prioritized_sampling:
                    if not hasattr(self, "avg_adv"):
                        self.avg_adv = defaultdict(float)
                    adv = self.trainer.compute_advantages()
                    for (agent0, agent1, a), vs in adv.items():
                        agent_pair = (agent0, agent1)
                        for v in vs:
                            if agent_pair not in self.avg_adv.keys():
                                self.avg_adv[agent_pair] = v
                            else:
                                self.avg_adv[agent_pair] = self.avg_adv[agent_pair] * 0.99 + v * 0.01

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode < 50:
                if episode % 2 == 0:
                    self.trainer.save(total_num_steps, save_dir=self.save_dir)
                    # self.trainer.save(episode, save_dir=self.save_dir)
            elif episode < 100:
                if episode % 5 == 0:
                    self.trainer.save(total_num_steps, save_dir=self.save_dir)
                    # self.trainer.save(episode, save_dir=self.save_dir)
            else:
                if episode % self.save_interval == 0 or episode == episodes - 1:
                    self.trainer.save(total_num_steps, save_dir=self.save_dir)
                    # self.trainer.save(episode, save_dir=self.save_dir)

            self.trainer.update_best_r(
                {
                    trainer_name: np.mean(self.env_info.get(f"either-{trainer_name}-ep_sparse_r", -1e9))
                    for trainer_name in self.trainer.active_trainers
                },
                save_dir=self.save_dir,
            )

            # log information
            if episode % self.log_interval == 0 or episode == episodes - 1:
                end = time.time()
                eta_t = eta(start, end, self.num_env_steps, total_num_steps)
                logger.info(
                    "Layout {} Algo {} Exp {} Seed {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}, ETA {}.".format(
                        self.all_args.layout_name,
                        self.algorithm_name,
                        self.experiment_name,
                        self.all_args.seed,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                        eta_t,
                    )
                )
                average_ep_rew_dict = {
                    k[: k.rfind("-")]: f"{np.mean(train_infos[k]):.3f}"
                    for k in train_infos.keys()
                    if "average_episode_rewards" in k and "either" not in k
                }
                logger.info(f"average episode rewards is\n{pprint.pformat(average_ep_rew_dict, width=600)}")
                average_ep_sparse_rew_dict = {
                    k[: k.rfind("-")]: f"{np.mean(env_infos[k]):.3f}"
                    for k in env_infos.keys()
                    if k.endswith("ep_sparse_r") and "either" not in k
                }
                logger.info(
                    f"average sparse episode rewards is\n{pprint.pformat(average_ep_sparse_rew_dict, width=600, compact=True)}"
                )
                if self.all_args.algorithm_name == "traj":
                    if self.all_args.stage == 1:
                        logger.debug(f'jsd is {train_infos["average_jsd"]}')
                        logger.debug(f'jsd loss is {train_infos["average_jsd_loss"]}')

                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                if self.use_wandb:
                    wandb.log({"train/ETA": eta_t}, step=total_num_steps)

            # eval
            if episode > 0 and episode % self.eval_interval == 0 and self.use_eval or episode == episodes - 1:
                if reset_map_ea2p_fn is not None:
                    map_ea2p = reset_map_ea2p_fn(episode)
                    self.policy.set_map_ea2p(map_ea2p, load_unused_to_cpu=True)
                eval_info = self.evaluate_with_multi_policy()
                # logger.debug("eval_info: {}".format(pprint.pformat(eval_info)))
                self.log_env(eval_info, total_num_steps)
                self.eval_info.update(eval_info)

            e_time = time.time()
            logger.trace(f"Post update models time: {e_time - s_time:.3f}s")

    def train_fcp(self):
        raise NotImplementedError

    def train_mep(self):
        assert self.all_args.population_size == len(self.trainer.population)
        self.population_size = self.all_args.population_size
        self.population = sorted(
            self.trainer.population.keys()
        )  # Note index and trainer name would not match when there are >= 10 agents

        logger.info(f"population_size: {self.all_args.population_size}, {self.population}")

        if self.all_args.stage == 1:
            # Stage 1: train a maximum entropy population
            if self.use_eval:
                assert self.n_eval_rollout_threads % self.population_size == 0
                self.all_args.eval_episodes *= self.population_size
                map_ea2p = {
                    (e, a): self.population[e % self.population_size]
                    for e in range(self.n_eval_rollout_threads)
                    for a in range(self.num_agents)
                }
                self.policy.set_map_ea2p(map_ea2p)

            def pbt_reset_map_ea2t_fn(episode):
                # Round robin trainer
                trainer_name = self.population[episode % self.population_size]
                map_ea2t = {(e, a): trainer_name for e in range(self.n_rollout_threads) for a in range(self.num_agents)}
                return map_ea2t

            # MARK: *self.population_size
            self.num_env_steps *= self.population_size
            self.save_interval *= self.population_size
            self.log_interval *= self.population_size
            self.eval_interval *= self.population_size

            self.naive_train_with_multi_policy(reset_map_ea2t_fn=pbt_reset_map_ea2t_fn)

            if self.use_eval:
                self.all_args.eval_episodes /= self.population_size
            self.num_env_steps /= self.population_size
            self.save_interval /= self.population_size
            self.log_interval /= self.population_size
            self.eval_interval /= self.population_size
        else:
            # Stage 2: train an agent against population with prioritized sampling
            agent_name = self.trainer.agent_name
            assert self.use_eval
            assert (
                self.n_eval_rollout_threads % self.all_args.eval_env_batch == 0
                and self.all_args.eval_episodes % self.all_args.eval_env_batch == 0
            )
            assert self.n_rollout_threads % self.all_args.train_env_batch == 0
            self.all_args.eval_episodes = (
                self.all_args.eval_episodes * self.population_size // self.all_args.eval_env_batch
            )
            self.eval_idx = 0
            all_agent_pairs = list(itertools.product(self.population, [agent_name])) + list(
                itertools.product([agent_name], self.population)
            )
            logger.info(f"all agent pairs: {all_agent_pairs}")

            running_avg_r = -np.ones((self.population_size * 2,), dtype=np.float32) * 1e9

            def mep_reset_map_ea2t_fn(episode):
                # Randomly select agents from population to be trained
                # 1) consistent with MEP to train against one agent each episode 2) sample different agents to train against
                sampling_prob_np = np.ones((self.population_size * 2,)) / self.population_size / 2
                if self.all_args.use_advantage_prioritized_sampling:
                    # logger.debug("use advantage prioritized sampling")
                    if episode > 0:
                        metric_np = np.array([self.avg_adv[agent_pair] for agent_pair in all_agent_pairs])
                        # TODO: retry this
                        sampling_rank_np = rankdata(metric_np, method="dense")
                        sampling_prob_np = sampling_rank_np / sampling_rank_np.sum()
                        sampling_prob_np /= sampling_prob_np.sum()
                        maxv = 1.0 / (self.population_size * 2) * 10
                        while sampling_prob_np.max() > maxv + 1e-6:
                            sampling_prob_np = sampling_prob_np.clip(max=maxv)
                            sampling_prob_np /= sampling_prob_np.sum()
                elif self.all_args.mep_use_prioritized_sampling:
                    metric_np = np.zeros((self.population_size * 2,))
                    for i, agent_pair in enumerate(all_agent_pairs):
                        train_r = np.mean(self.env_info.get(f"{agent_pair[0]}-{agent_pair[1]}-ep_sparse_r", -1e9))
                        eval_r = np.mean(
                            self.eval_info.get(
                                f"{agent_pair[0]}-{agent_pair[1]}-eval_ep_sparse_r",
                                -1e9,
                            )
                        )

                        avg_r = 0.0
                        cnt_r = 0
                        if train_r > -1e9:
                            avg_r += train_r * (self.n_rollout_threads // self.all_args.train_env_batch)
                            cnt_r += self.n_rollout_threads // self.all_args.train_env_batch
                        if eval_r > -1e9:
                            avg_r += eval_r * (
                                self.all_args.eval_episodes
                                // (self.n_eval_rollout_threads // self.all_args.eval_env_batch)
                            )
                            cnt_r += self.all_args.eval_episodes // (
                                self.n_eval_rollout_threads // self.all_args.eval_env_batch
                            )
                        if cnt_r > 0:
                            avg_r /= cnt_r
                        else:
                            avg_r = -1e9
                        if running_avg_r[i] == -1e9:
                            running_avg_r[i] = avg_r
                        else:
                            # running average
                            running_avg_r[i] = running_avg_r[i] * 0.95 + avg_r * 0.05
                        metric_np[i] = running_avg_r[i]
                    running_avg_r_dict = {}
                    for i, agent_pair in enumerate(all_agent_pairs):
                        running_avg_r_dict[f"running_average_return/{agent_pair[0]}-{agent_pair[1]}"] = np.mean(
                            running_avg_r[i]
                        )
                    if self.use_wandb:
                        for k, v in running_avg_r_dict.items():
                            if v > -1e9:
                                wandb.log({k: v}, step=self.total_num_steps)
                    running_avg_r_dict = {
                        f"running_average_return/{agent_pair[0]}-{agent_pair[1]}": f"{running_avg_r[i]:.3f}"
                        for i, agent_pair in enumerate(all_agent_pairs)
                    }
                    logger.trace(f"running avg_r\n{pprint.pformat(running_avg_r_dict, width=600, compact=True)}")
                    if (metric_np > -1e9).astype(np.int32).sum() > 0:
                        avg_metric = metric_np[metric_np > -1e9].mean()
                    else:
                        # uniform
                        avg_metric = 1.0
                    metric_np[metric_np == -1e9] = avg_metric

                    # reversed return
                    sampling_rank_np = rankdata(1.0 / (metric_np + 1e-6), method="dense")
                    sampling_prob_np = sampling_rank_np / sampling_rank_np.sum()
                    sampling_prob_np = sampling_prob_np**self.all_args.mep_prioritized_alpha
                    sampling_prob_np /= sampling_prob_np.sum()
                assert abs(sampling_prob_np.sum() - 1) < 1e-3

                # log sampling prob
                sampling_prob_dict = {}
                for i, agent_pair in enumerate(all_agent_pairs):
                    sampling_prob_dict[f"sampling_prob/{agent_pair[0]}-{agent_pair[1]}"] = sampling_prob_np[i]
                if self.use_wandb:
                    wandb.log(sampling_prob_dict, step=self.total_num_steps)

                n_selected = self.n_rollout_threads // self.all_args.train_env_batch
                pair_idx = np.random.choice(2 * self.population_size, size=(n_selected,), p=sampling_prob_np)
                if self.all_args.uniform_sampling_repeat > 0:
                    assert n_selected >= 2 * self.population_size * self.all_args.uniform_sampling_repeat
                    i = 0
                    for r in range(self.all_args.uniform_sampling_repeat):
                        for x in range(2 * self.population_size):
                            pair_idx[i] = x
                            i += 1
                map_ea2t = {
                    (e, a): all_agent_pairs[pair_idx[e % n_selected]][a]
                    for e, a in itertools.product(range(self.n_rollout_threads), range(self.num_agents))
                }

                return map_ea2t

            def mep_reset_map_ea2p_fn(episode):
                if self.all_args.eval_policy != "":
                    map_ea2p = {
                        (e, a): [self.all_args.eval_policy, agent_name][(e + a) % 2]
                        for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))
                    }
                else:
                    map_ea2p = {
                        (e, a): all_agent_pairs[
                            (self.eval_idx + e // self.all_args.eval_env_batch) % (self.population_size * 2)
                        ][a]
                        for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))
                    }
                    self.eval_idx += self.n_eval_rollout_threads // self.all_args.eval_env_batch
                    self.eval_idx %= self.population_size * 2
                featurize_type = [
                    [self.policy.featurize_type[map_ea2p[(e, a)]] for a in range(self.num_agents)]
                    for e in range(self.n_eval_rollout_threads)
                ]
                self.eval_envs.reset_featurize_type(featurize_type)
                return map_ea2p

            self.naive_train_with_multi_policy(
                reset_map_ea2t_fn=mep_reset_map_ea2t_fn,
                reset_map_ea2p_fn=mep_reset_map_ea2p_fn,
            )

    def train_traj(self):
        assert self.all_args.population_size == len(self.trainer.population)
        self.population_size = self.all_args.population_size
        self.population = sorted(
            self.trainer.population.keys()
        )  # Note index and trainer name would not match when there are >= 10 agents

        logger.info(f"population_size: {self.all_args.population_size}, {self.population}")

        assert self.n_rollout_threads % self.all_args.train_env_batch == 0
        assert self.all_args.stage == 1
        if self.all_args.stage == 1:
            if self.use_eval:
                assert self.n_eval_rollout_threads % self.population_size == 0
                self.all_args.eval_episodes *= self.population_size
                map_ea2p = {
                    (e, a): self.population[e % self.population_size]
                    for e in range(self.n_eval_rollout_threads)
                    for a in range(self.num_agents)
                }
                self.policy.set_map_ea2p(map_ea2p)

            def pbt_reset_map_ea2t_fn(episode):
                # Round robin trainer
                map_ea2t = {
                    (e, a): self.population[(e + episode * self.n_rollout_threads) % self.population_size]
                    for e in range(self.n_rollout_threads)
                    for a in range(self.num_agents)
                }
                return map_ea2t

            # MARK: *self.population_size
            self.num_env_steps *= self.population_size
            self.save_interval *= self.population_size
            self.log_interval *= self.population_size
            self.eval_interval *= self.population_size

            self.naive_train_with_multi_policy(reset_map_ea2t_fn=pbt_reset_map_ea2t_fn)

            if self.use_eval:
                self.all_args.eval_episodes /= self.population_size
            self.num_env_steps /= self.population_size
            self.save_interval /= self.population_size
            self.log_interval /= self.population_size
            self.eval_interval /= self.population_size

    def train_cole(self):
        assert self.all_args.stage == 2
        assert self.use_eval
        assert (
            self.n_eval_rollout_threads % self.all_args.eval_env_batch == 0
            and self.all_args.eval_episodes % self.all_args.eval_env_batch == 0
        )
        assert self.n_rollout_threads % self.all_args.train_env_batch == 0
        self.all_args._eval_episodes = self.all_args.eval_episodes

        """
           p1 p2 p3 ...
        p1
        p2
        p3
        ...
        agent_name
        """
        self.u_matrix = defaultdict(dict)
        self.generation_interval = self.all_args.generation_interval
        self.num_generation = self.all_args.num_generation
        self.population_play_ratio = self.all_args.population_play_ratio
        assert self.all_args.population_size == len(self.trainer.population)
        self.max_population_size = self.all_args.population_size
        self.population = list(
            self.trainer.population.keys()
        )  # Note index and trainer name would not match when there are >= 10 agents
        logger.info(f"total population {self.population}")
        # self.population_size = self.population_play_ratio
        self.generated_population_names = self.trainer.generated_population_names
        self.population_size = len(self.generated_population_names)
        self.generation = self.population_size
        logger.info(f"population {self.generated_population_names}")

        self.eval_idx = 0
        self.n_generation_try = 0

        # self.all_args.eval_episodes = (
        #     self.all_args.eval_episodes
        #     * self.population_size
        #     // self.all_args.eval_env_batch
        # )

        agent_name = self.trainer.agent_name

        # init u_matrix
        for p in self.generated_population_names:
            for o_p in self.generated_population_names:
                self.u_matrix[p][o_p] = 0.0
            self.u_matrix[agent_name][p] = 0.0

        def cole_reset_map_ea2t_fn(episode):
            if episode > 0:
                # update u_matrix
                for p in self.generated_population_names:
                    eval_r = []
                    for log_name in [f"{p}-{agent_name}", f"{agent_name}-{p}"]:
                        if f"{log_name}-ep_sparse_r" in self.env_info:
                            eval_r.append(np.mean(self.env_info[f"{log_name}-ep_sparse_r"]))
                    if len(eval_r) > 0:
                        self.u_matrix[agent_name][p] = (self.u_matrix[agent_name][p] + np.mean(eval_r)) / 2

            if episode > self.generation_interval and episode % self.generation_interval == 1:
                # generate a new partner
                model_path = self.trainer.save_actor(agent_name, self.generation + 1)
                self.generation += 1
                available_population = list(set(self.population).difference(set(self.generated_population_names)))
                if len(available_population) > 0:
                    percent = 0.9
                else:
                    percent = 0.8
                metric_np = [
                    np.mean([v for _, v in self.u_matrix[p_name].items()]) for p_name in self.generated_population_names
                ] + [np.mean([v for _, v in self.u_matrix[agent_name].items()])]
                rank = np.argsort(np.argsort(metric_np))[-1]
                if self.use_wandb:
                    wandb.log({"rank": rank}, step=self.total_num_steps)
                threshold = np.ceil(len(self.generated_population_names) * percent)
                if rank >= threshold or self.n_generation_try >= 2:
                    if len(available_population) > 0:
                        p_name = available_population[0]
                        self.trainer.policy_pool.update_policy(p_name, False, model_path={"actor": model_path})
                        logger.success(
                            f"add {model_path} with rank {rank}/{len(self.generated_population_names)} as {p_name}"
                        )
                        self.generated_population_names.append(p_name)
                        self.population_size += 1
                    else:
                        # replace old policy
                        p_name = np.random.choice(self.generated_population_names[:10])
                        self.trainer.policy_pool.update_policy(p_name, False, model_path={"actor": model_path})
                        logger.success(
                            f"replace {model_path} with rank {rank}/{len(self.generated_population_names)} as {p_name}"
                        )
                    # update u_matrix
                    self.u_matrix[p_name] = copy.deepcopy(self.u_matrix[agent_name])
                    for p, v in self.u_matrix[p_name].items():
                        self.u_matrix[p][p_name] = v
                    sp_v = np.mean(self.env_info[f"{agent_name}-{agent_name}-ep_sparse_r"])
                    self.u_matrix[p_name][p_name] = sp_v
                    self.u_matrix[agent_name][p_name] = sp_v

                    self.n_generation_try = 0
                    population_str = zip(
                        self.generated_population_names,
                        [
                            osp.basename(self.trainer.policy_pool.policy_info[a_n][1]["model_path"]["actor"])
                            for a_n in self.generated_population_names
                        ],
                    )
                    logger.success(f"population: size {len(self.generated_population_names)}, {list(population_str)}")

                    metric_np = [
                        np.mean([v for _, v in self.u_matrix[p_name].items()])
                        for p_name in self.generated_population_names
                    ] + [np.mean([v for _, v in self.u_matrix[agent_name].items()])]
                    ranks = np.argsort(np.argsort(metric_np)) + 1
                    logger.success(f"utility matrix sum\n{[round(m, 3) for m in metric_np]}")
                    logger.success(f"ranks\n{ranks}")
                else:
                    self.n_generation_try += 1
                    logger.warning(f"Failed to generate a new partner, try {self.n_generation_try} / 3 times")
                    logger.warning(
                        f"""population metric: {[round(m,3) for m in metric_np[:-1]]}, ego agent metric: {round(metric_np[-1], 3)} rank {rank}/{len(self.generated_population_names)}, need to rank >= {threshold}"""
                    )

            all_agent_pairs = list(itertools.product(self.generated_population_names, [agent_name])) + list(
                itertools.product([agent_name], self.generated_population_names)
            )
            rollout_block_size = self.n_rollout_threads // (self.population_play_ratio + 1)
            map_ea2t = {
                (e, a): agent_name for e, a in itertools.product(range(rollout_block_size), range(self.num_agents))
            }
            metric_np = []
            for p_name in self.generated_population_names:
                metric = np.mean([v for _, v in self.u_matrix[p_name].items()])
                metric_np.append(metric)
            metric_np = np.array(metric_np)
            if metric_np.sum() > 0:
                metric_np = metric_np / metric_np.sum()
            metric_np = 1 - metric_np
            metric_np /= metric_np.sum()

            sampling_prob_np = metric_np

            sampling_prob_np = rankdata(sampling_prob_np, method="dense")
            sampling_prob_np = sampling_prob_np / sampling_prob_np.sum()
            sampling_prob_np = sampling_prob_np**self.all_args.prioritized_alpha
            sampling_prob_np /= sampling_prob_np.sum()
            # logger.info(f"cole sampling prob {sampling_prob_np}")
            # log sampling prob
            sampling_prob_dict = {}
            for i, p_name in enumerate(self.generated_population_names):
                sampling_prob_dict[f"sampling_prob/{p_name}"] = sampling_prob_np[i]
            if self.use_wandb:
                wandb.log(sampling_prob_dict, step=self.total_num_steps)

            sampling_prob_np = np.repeat(sampling_prob_np, 2) / 2
            n_selected = self.n_rollout_threads - rollout_block_size
            pair_idx = np.random.choice(2 * self.population_size, size=(n_selected,), p=sampling_prob_np)
            for i in range(rollout_block_size, self.n_rollout_threads):
                map_ea2t[(i, 0)] = all_agent_pairs[pair_idx[i - rollout_block_size]][0]
                map_ea2t[(i, 1)] = all_agent_pairs[pair_idx[i - rollout_block_size]][1]
            return map_ea2t

        def cole_reset_map_ea2p_fn(episode):
            self.all_args.eval_episodes = (
                self.all_args._eval_episodes * (self.population_size * 2 + 1) // self.all_args.eval_env_batch
            )
            all_agent_pairs = (
                list(itertools.product(self.generated_population_names, [agent_name]))
                + list(itertools.product([agent_name], self.generated_population_names))
                + [(agent_name, agent_name)]
            )
            map_ea2p = {
                (e, a): all_agent_pairs[(self.eval_idx + e) % (self.population_size * 2 + 1)][a]
                for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))
            }
            self.eval_idx += self.n_eval_rollout_threads
            self.eval_idx %= self.population_size * 2 + 1

            featurize_type = [
                [self.policy.featurize_type[map_ea2p[(e, a)]] for a in range(self.num_agents)]
                for e in range(self.n_eval_rollout_threads)
            ]
            self.eval_envs.reset_featurize_type(featurize_type)
            return map_ea2p

        self.naive_train_with_multi_policy(
            reset_map_ea2t_fn=cole_reset_map_ea2t_fn,
            reset_map_ea2p_fn=cole_reset_map_ea2p_fn,
        )
