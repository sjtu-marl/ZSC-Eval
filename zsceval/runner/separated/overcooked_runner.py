import time
from collections import defaultdict

import numpy as np
import torch
import wandb
from icecream import ic
from loguru import logger

from zsceval.runner.separated.base_runner import Runner
from zsceval.utils.log_util import eta


def _t2n(x):
    return x.detach().cpu().numpy()


class OvercookedRunner(Runner):
    def __init__(self, config):
        super(OvercookedRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0

        for episode in range(episodes):
            time.time()
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

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
                    # self.save(episode)
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

                for a in range(self.num_agents):
                    train_infos[a]["average_episode_rewards"] = np.mean(self.buffer[a].rewards) * self.episode_length
                    logger.info(
                        "agent {} average episode rewards is {}".format(a, train_infos[a]["average_episode_rewards"])
                    )

                env_infos = defaultdict(list)
                if self.use_wandb:
                    wandb.log({"train/ETA": eta_t}, step=total_num_steps)
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
                            if "ep_hidden_r_by_agent" in info["episode"]:
                                env_infos[f"ep_hidden_r_by_agent{a}"].append(info["episode"]["ep_hidden_r_by_agent"][a])
                            for i, k in enumerate(shaped_info_keys):
                                env_infos[f"ep_{k}_by_agent{a}"].append(info["episode"]["ep_category_r_by_agent"][a][i])
                        env_infos["ep_sparse_r"].append(info["episode"]["ep_sparse_r"])
                        env_infos["ep_shaped_r"].append(info["episode"]["ep_shaped_r"])

                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                logger.info(f'average sparse rewards is {np.mean(env_infos["ep_sparse_r"]):.3f}')

            # eval
            if episode % self.eval_interval == 0 and self.use_eval or episode == episodes - 1:
                self.eval(total_num_steps)
            # e_time = time.time()
            # logger.trace(f"Post update models time: {e_time - s_time:.3f}s")

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        obs = np.stack(obs)

        if not self.use_centralized_V:
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
        eval_episode_rewards = []
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
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
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

        eval_env_infos["eval_average_episode_rewards"] = np.sum(eval_episode_rewards, axis=0)
        logger.success(
            f'eval average sparse rewards {np.mean(eval_env_infos["eval_ep_sparse_r"]):.3f} {len(eval_env_infos["eval_ep_sparse_r"])} episodes, total num timesteps {total_num_steps}/{self.num_env_steps}'
        )

        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        envs = self.envs
        obs, share_obs, available_actions = envs.reset()
        obs = np.stack(obs)

        for episode in range(self.all_args.render_episodes):
            episode_rewards = []

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

            for step in range(self.episode_length):
                time.time()
                actions = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(np.array(obs)[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(
                        np.array(obs)[:, agent_id],
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True,
                    )

                    action = action.detach().cpu().numpy()
                    actions.append(action[0])
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                # Obser reward and next obs
                print("action:", actions)
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step([actions])
                obs = np.stack(obs)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

            for info in infos:
                for a in range(self.num_agents):
                    ic(info["episode"]["ep_sparse_r_by_agent"][a])
                    ic(info["episode"]["ep_shaped_r_by_agent"][a])
                ic(info["episode"]["ep_sparse_r"])
                ic(info["episode"]["ep_shaped_r"])

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
            # print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

    def save(self, step, save_critic: bool = False):
        # logger.info(f"save hsp periodic_{step}.pt")
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
