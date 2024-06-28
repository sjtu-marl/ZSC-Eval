import time
from collections import defaultdict

import numpy as np
import torch
import wandb
from icecream import ic
from loguru import logger

from zsceval.envs.grf.grf_env import SHAPED_INFOS
from zsceval.runner.separated.base_runner import Runner
from zsceval.utils.log_util import eta, get_table_str


def _t2n(x):
    return x.detach().cpu().numpy()


class GRFRunner(Runner):
    """
    A wrapper to start the RL agent training algorithm.
    """

    def __init__(self, config):
        super().__init__(config)

    def run(self):
        # train sp
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0

        shaped_info_keys = SHAPED_INFOS
        env_infos = defaultdict(list)

        for episode in range(episodes):
            # s_time = time.time()
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

                for _d, _info in zip(dones, infos):
                    if np.all(np.array(_d)):
                        for a in range(self.num_agents):
                            env_infos[f"ep_sparse_r_by_agent{a}"].append(_info["episode"]["score"][a])
                            env_infos[f"ep_shaped_r_by_agent{a}"].append(_info["episode"]["shaped_return"][a])
                            for i, k in enumerate(shaped_info_keys):
                                env_infos[f"ep_{k}_by_agent{a}"].append(_info["episode"][k][a])
                        env_infos["ep_sparse_r"].append(int(sum(_info["episode"]["score"]) > 0))
                        env_infos["ep_shaped_r"].append(sum(_info["episode"]["shaped_return"]))
                        env_infos["ep_length"].append(_info["episode"]["length"])

            # e_time = time.time()
            # logger.trace(f"Rollout time: {e_time - s_time:.3f}s")

            # compute return and update network
            # s_time = time.time()
            self.compute()
            train_infos = self.train(total_num_steps)
            # e_time = time.time()
            # logger.trace(f"Update models time: {e_time - s_time:.3f}s")

            # post process
            # s_time = time.time()
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode < 50:
                if episode % 2 == 0:
                    self.save(total_num_steps)
            elif episode < 100:
                if episode % 5 == 0:
                    self.save(total_num_steps)
            else:
                if episode % self.save_interval == 0 or episode == episodes - 1:
                    self.save(total_num_steps)

            # log information
            if episode % self.log_interval == 0 or episode == episodes - 1:
                end = time.time()
                eta_t = eta(start, end, self.num_env_steps, total_num_steps)

                # shaped reward
                log_data = []
                for a in range(self.num_agents):
                    train_infos[a]["average_episode_rewards"] = np.mean(self.buffer[a].rewards) * self.episode_length
                    log_data.append(
                        (
                            f"Average Return Agent {a}",
                            f'{train_infos[a]["average_episode_rewards"]:.3f}',
                        )
                    )

                get_table_str(log_data)

                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                if self.use_wandb:
                    wandb.log({"train/ETA": eta_t}, step=total_num_steps)

                if len(env_infos["ep_length"]) > 0:
                    average_len = np.mean(env_infos["ep_length"])
                    average_sparse_r = np.mean(env_infos["ep_sparse_r"])
                else:
                    average_len = self.episode_length
                    average_sparse_r = 0
                log_data = [
                    ("scenario", self.all_args.scenario_name),
                    ("Algorithm", self.algorithm_name),
                    ("Experiment", self.experiment_name),
                    ("Seed", self.all_args.seed),
                    ("Episodes", episode),
                    ("Total Episodes", episodes),
                    ("Timesteps", total_num_steps),
                    ("Total Timesteps", self.num_env_steps),
                    ("FPS", int(total_num_steps / (end - start))),
                    ("ETA", eta_t),
                    ("Average Return", f'{np.mean(env_infos["ep_shaped_r"]):.3f}'),
                    ("Average Sparse Return", f"{average_sparse_r:.3f}"),
                    ("Average Episode Length", f"{average_len:.3f}"),
                ]
                logger.info("training process:\n" + get_table_str(log_data))

                env_infos = defaultdict(list)
            # eval
            if episode % self.eval_interval == 0 and self.use_eval or episode == episodes - 1:
                self.eval(total_num_steps)
            # e_time = time.time()
            # logger.trace(f"Post update models time: {e_time - s_time:.3f}s")

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        obs = np.stack(obs)

        # replay buffer
        if self.use_centralized_V:
            share_obs = share_obs
        else:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            self.buffer[agent_id].available_actions[0] = available_actions[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[agent_id].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
                self.buffer[agent_id].available_actions[step],
            )
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)

            actions.append(action)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

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
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info["bad_transition"] else [1.0]] * self.num_agents for info in infos])

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = obs

            self.buffer[agent_id].insert(
                share_obs[:, agent_id],
                obs[:, agent_id],
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
                bad_masks=bad_masks[:, agent_id],
                available_actions=available_actions[:, agent_id],
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_average_episode_rewards = []
        eval_obs, _, eval_available_actions = self.eval_envs.reset()
        eval_obs = np.stack(eval_obs)

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        n_done = 0
        eval_env_infos = defaultdict(list)
        shaped_info_keys = SHAPED_INFOS
        unfinished_threads = np.ones(self.n_eval_rollout_threads, dtype=bool)

        for _ in range(self.episode_length):
            eval_actions = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id],
                    deterministic=not self.all_args.eval_stochastic,
                )

                eval_action = _t2n(eval_action)
                eval_actions.append(eval_action)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            eval_actions = np.stack(eval_actions).transpose(1, 0, 2)
            # logger.debug(f"eval_actions {eval_actions.shape}")
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

            for e, (_d, _info) in enumerate(zip(eval_dones, eval_infos)):
                if np.all(np.array(_d)) and unfinished_threads[e]:
                    unfinished_threads[e] = False
                    for a in range(self.num_agents):
                        eval_env_infos[f"eval_ep_sparse_r_by_agent{a}"].append(_info["episode"]["score"][a])
                        eval_env_infos[f"eval_ep_shaped_r_by_agent{a}"].append(_info["episode"]["shaped_return"][a])
                        for i, k in enumerate(shaped_info_keys):
                            eval_env_infos[f"eval_ep_{k}_by_agent{a}"].append(_info["episode"][k][a])
                    eval_env_infos["eval_ep_sparse_r"].append(int(sum(_info["episode"]["score"]) > 0))
                    eval_env_infos["eval_ep_shaped_r"].append(sum(_info["episode"]["shaped_return"]))
                    eval_env_infos["eval_ep_length"].append(_info["episode"]["length"])
                    n_done += 1

        eval_env_infos["eval_average_episode_rewards"] = np.sum(eval_average_episode_rewards)

        if n_done > 0:
            average_len = np.mean(eval_env_infos["eval_ep_length"])
            average_sparse_r = np.mean(eval_env_infos["eval_ep_sparse_r"])
        else:
            average_len = self.episode_length
            average_sparse_r = 0

        log_data = [
            ("Eval Average Sparse Return", f"{average_sparse_r:.3f}"),
            ("Eval Average Episode Length", f"{average_len:.3f}"),
            ("Eval Episode Num", n_done),
            ("Timesteps", f"{total_num_steps}/{self.num_env_steps}"),
        ]
        logger.info("evaluation:\n" + get_table_str(log_data))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        envs = self.envs
        obs, share_obs, available_actions = envs.reset()
        obs = np.stack(obs)

        for episode in range(self.all_args.render_episodes):
            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []
            for step in range(self.episode_length):
                time.time()

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
                obs, share_obs, rewards, dones, infos, available_actions = envs.step(actions)
                obs = np.stack(obs)

                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

            for info in infos:
                ic(info["episode"]["ep_sparse_r_by_agent"][0])
                ic(info["episode"]["ep_sparse_r_by_agent"][1])
                ic(info["episode"]["ep_shaped_r_by_agent"][0])
                ic(info["episode"]["ep_shaped_r_by_agent"][1])
                ic(info["episode"]["ep_sparse_r"])
                ic(info["episode"]["ep_shaped_r"])

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

    def save(self, step, save_critic: bool = False):
        # logger.info(f"save sp periodic_{step}.pt")
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                torch.save(
                    policy_model.state_dict(),
                    str(self.save_dir) + f"/model_agent{agent_id}_periodic_{step}.pt",
                )
            else:
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(
                    policy_actor.state_dict(),
                    str(self.save_dir) + f"/actor_agent{agent_id}_periodic_{step}.pt",
                )
                if save_critic:
                    policy_critic = self.trainer[agent_id].policy.critic
                    torch.save(
                        policy_critic.state_dict(),
                        str(self.save_dir) + f"/critic_agent{agent_id}_periodic_{step}.pt",
                    )
