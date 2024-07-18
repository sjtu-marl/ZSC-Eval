import copy

import numpy as np
import torch
from loguru import logger

from zsceval.algorithms.population.policy_pool import PolicyPool
from zsceval.algorithms.population.trainer_pool import TrainerPool
from zsceval.algorithms.population.utils import _t2n


class MEP_Trainer(TrainerPool):
    def __init__(self, args, policy_pool: PolicyPool, device=torch.device("cpu")):
        super().__init__(args, policy_pool, device)

        self.stage = args.stage
        if self.stage == 1:
            self.entropy_alpha = args.mep_entropy_alpha
        self.num_mini_batch = args.num_mini_batch
        self.share_policy = args.share_policy
        self.eval_policy = args.eval_policy
        self.num_agents = args.num_agents
        self.args = args

    def init_population(self):
        super().init_population()

        self.agent_name = self.all_args.adaptive_agent_name
        self.population = {
            trainer_name: self.trainer_pool[trainer_name]
            for trainer_name in self.trainer_pool.keys()
            if self.agent_name not in trainer_name
        }
        if self.eval_policy != "":
            self.population = {
                trainer_name: self.population[trainer_name]
                for trainer_name in self.population.keys()
                if self.eval_policy not in trainer_name
            }
        self.population_size = self.all_args.population_size

        # print(self.population.keys(), self.population_size, self.stage)
        logger.info(f"population keys {self.population.keys()}, size {self.population_size}, stage {self.stage}")

        if self.share_policy:
            assert len(self.population) == self.population_size, len(self.population)
        else:
            assert len(self.population) == self.population_size * self.num_agents, (
                len(self.population),
                self.population_size,
                self.num_agents,
            )
            all_trainer_names = self.trainer_pool.keys()
            all_trainer_names = [x[: x.rfind("_")] for x in all_trainer_names]
            for a in range(self.num_agents):
                for x in all_trainer_names:
                    assert f"{x}_{a}" in self.trainer_pool.keys()

    def reward_shaping_steps(self):
        reward_shaping_steps = super().reward_shaping_steps()
        if self.stage == 1:
            return [x // 2 for x in reward_shaping_steps]
        return reward_shaping_steps

    def save_steps(self):
        steps = super().save_steps()
        # logger.info(f"steps {steps}")
        if self.stage == 1:
            steps = {trainer_name: v // 2 for trainer_name, v in steps.items()}
        return steps

    def train(self, **kwargs):
        if self.stage == 1:
            # pbt stage, add population entropy into rewards
            if self.share_policy:
                assert (
                    len(self.active_trainers) == 1 * self.all_args.n_rollout_threads // self.all_args.train_env_batch
                ), (
                    len(self.active_trainers),
                    self.all_args.n_rollout_threads,
                    self.all_args.train_env_batch,
                )
            else:
                assert (
                    len(self.active_trainers)
                    == self.num_agents * self.all_args.n_rollout_threads // self.all_args.train_env_batch
                ), (
                    len(self.active_trainers),
                    self.all_args.n_rollout_threads,
                    self.all_args.train_env_batch,
                )
            for trainer in self.population.values():
                trainer.to(self.device)

            # WARNING: not supported for recurrent policy
            # Qs: Why

            nlog_pop_act_prob = {}
            for active_trainer_name in self.active_trainers:
                buffer = self.buffer_pool[active_trainer_name]
                num_traj = (
                    self.all_args.train_env_batch * self.num_agents
                    if self.share_policy
                    else self.all_args.train_env_batch
                )
                population_action_probs = np.zeros((buffer.episode_length, num_traj, 1), dtype=np.float32)
                actions = (
                    buffer.actions[: buffer.episode_length]
                    .reshape(buffer.episode_length, num_traj, *buffer.actions.shape[3:])
                    .astype(np.int32)
                )

                # data for recurrent policy
                if any(
                    self.policy_config(trainer_name)[0].use_recurrent_policy for trainer_name in self.population.keys()
                ):
                    obs_recurrent = buffer.obs[: buffer.episode_length].reshape(
                        buffer.episode_length, num_traj, *buffer.obs.shape[3:]
                    )
                    rnn_states_recurrent = np.zeros_like(buffer.rnn_states[0]).reshape(-1, *buffer.rnn_states.shape[3:])
                    masks_recurrent = buffer.masks[: buffer.episode_length].reshape(buffer.episode_length, num_traj, 1)
                    if self.args.env_name == "Overcooked":
                        available_actions_recurrent = buffer.available_actions[: buffer.episode_length].reshape(
                            buffer.episode_length,
                            num_traj,
                            *buffer.available_actions.shape[3:],
                        )
                    else:
                        available_actions_recurrent = None
                    actions_recurrent = actions

                # data for mlp policy
                if any(
                    not self.policy_config(trainer_name)[0].use_recurrent_policy
                    for trainer_name in self.population.keys()
                ):
                    obs_mlp = buffer.obs[: buffer.episode_length].reshape(-1, *buffer.obs.shape[3:])
                    rnn_states_mlp = buffer.rnn_states[: buffer.episode_length].reshape(
                        -1, *buffer.rnn_states.shape[3:]
                    )
                    masks_mlp = buffer.masks[: buffer.episode_length].reshape(-1, *buffer.masks.shape[3:])
                    actions_mlp = actions.reshape(-1, actions.shape[-1])
                    if self.args.env_name == "Overcooked":
                        available_actions_mlp = buffer.available_actions[: buffer.episode_length].reshape(
                            -1,
                            *buffer.available_actions.shape[3:],
                        )
                    else:
                        available_actions_mlp = None

                with torch.no_grad():
                    for trainer_name, trainer in self.population.items():
                        if (
                            not self.share_policy and trainer_name.split("_")[-1] != active_trainer_name.split("_")[-1]
                        ):  # only the data from the same side should be used
                            continue
                        if self.policy_config(trainer_name)[0].use_recurrent_policy:
                            # print(trainer_name, "use recurrent policy")
                            action_probs = np.zeros((buffer.episode_length, num_traj, 1), dtype=np.float32)

                            for t in range(buffer.episode_length):
                                (
                                    t_action_log_probs,
                                    rnn_states,
                                ) = trainer.policy.get_action_log_probs(
                                    obs_recurrent[t],
                                    rnn_states_recurrent,
                                    actions_recurrent[t],
                                    masks_recurrent[t],
                                    available_actions_recurrent[t] if available_actions_recurrent is not None else None,
                                )
                                action_probs[t] = _t2n(t_action_log_probs.exp()).reshape(num_traj, 1)

                            action_probs = action_probs.reshape(buffer.episode_length, num_traj, 1)
                            population_action_probs += action_probs
                        else:
                            # print(trainer_name, "mlp")
                            action_probs = np.zeros((buffer.episode_length * num_traj, 1), dtype=np.float32)
                            mini_batch_size = buffer.episode_length * num_traj // self.num_mini_batch
                            for l in range(0, buffer.episode_length * num_traj, mini_batch_size):
                                r = min(
                                    buffer.episode_length * num_traj,
                                    l + mini_batch_size,
                                )
                                (
                                    t_action_log_probs,
                                    _,
                                ) = trainer.policy.get_action_log_probs(
                                    obs_mlp[l:r],
                                    rnn_states_mlp[l:r],
                                    actions_mlp[l:r],
                                    masks_mlp[l:r],
                                    available_actions_mlp[l:r] if available_actions_mlp is not None else None,
                                )
                                action_probs[l:r] = _t2n(t_action_log_probs.exp())

                            action_probs = action_probs.reshape(buffer.episode_length, num_traj, 1)
                            population_action_probs += action_probs
                population_action_probs /= len(self.population)

                nlog_pop_act_prob[active_trainer_name] = -np.log(np.maximum(population_action_probs, 1e-5)).reshape(
                    buffer.episode_length, num_traj, 1, 1
                )

                buffer.rewards[: buffer.episode_length] += nlog_pop_act_prob[active_trainer_name] * self.entropy_alpha

        super().train()

        if self.stage == 1:
            # subtract population entropy from reward
            # correct log info

            for active_trainer_name in self.active_trainers:
                buffer = self.buffer_pool[active_trainer_name]
                buffer.rewards[: buffer.episode_length] -= nlog_pop_act_prob[active_trainer_name] * self.entropy_alpha
                self.train_infos.update(
                    {
                        f"{active_trainer_name}-average_nlog_pop_act_prob": np.mean(
                            nlog_pop_act_prob[active_trainer_name]
                        )
                    }
                )
                self.train_infos.update(
                    {f"{active_trainer_name}-average_episode_rewards": np.mean(buffer.rewards) * buffer.episode_length}
                )

        return copy.deepcopy(self.train_infos)
