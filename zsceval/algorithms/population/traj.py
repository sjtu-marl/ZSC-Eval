import copy
import random
from typing import Dict, List

import numpy as np
import torch

from zsceval.algorithms.population.policy_pool import PolicyPool
from zsceval.algorithms.population.trainer_pool import TrainerPool
from zsceval.algorithms.r_mappo.r_mappo import R_MAPPO


class Traj_Trainer(TrainerPool):
    def __init__(self, args, policy_pool: PolicyPool, device=torch.device("cpu")):
        super().__init__(args, policy_pool, device)

        self.seed = args.seed
        self.stage = args.stage
        if self.stage == 1:
            self.entropy_alpha = args.traj_entropy_alpha
            self.gamma = args.traj_gamma
            self.use_full_jsd = args.traj_use_full_jsd
            self.use_co_update = args.traj_use_co_update
        self.args = args

    def init_population(self):
        super().init_population()

        self.agent_name = self.all_args.adaptive_agent_name
        self.population = {
            trainer_name: self.trainer_pool[trainer_name]
            for trainer_name in self.trainer_pool.keys()
            if not trainer_name.startswith(self.agent_name)
        }
        self.population_size = self.all_args.population_size

        assert len(self.population) % self.population_size == 0

    def reward_shaping_steps(self):
        reward_shaping_steps = super().reward_shaping_steps()
        if self.stage == 1:
            return [x // 2 for x in reward_shaping_steps]
        return reward_shaping_steps

    def save_steps(self):
        steps = super().save_steps()
        if self.stage == 1:
            steps = {trainer_name: v // 2 for trainer_name, v in steps.items()}
        return steps

    def reset(
        self,
        map_ea2t,
        n_rollout_threads,
        num_agents,
        load_unused_to_cpu=False,
    ):
        super().reset(map_ea2t, n_rollout_threads, num_agents, load_unused_to_cpu)

    def traj_jsd(self, all_trainer: List[R_MAPPO], trainer: R_MAPPO, traj: Dict[str, np.ndarray]):
        # (episode_length, n_threads, n_agents, *)
        share_obs = traj["share_obs"]
        obs = traj["obs"]
        rnn_states = traj["rnn_states"]
        rnn_states_critic = traj["rnn_states_critic"]
        actions = traj["actions"]
        masks = traj["masks"]
        available_actions = traj["available_actions"]

        share_obs_shape = share_obs.shape[3:]
        obs_shape = obs.shape[3:]

        episode_length = self.all_args.episode_length
        assert len(share_obs) == episode_length, (len(share_obs), episode_length)

        gamma = self.gamma
        gamma_metric = torch.diag(gamma * torch.ones(episode_length))
        for i in range(episode_length - 1):
            gamma *= self.gamma
            gamma_metric += torch.diag(gamma * torch.ones(episode_length - 1 - i), i + 1)
            gamma_metric += torch.diag(gamma * torch.ones(episode_length - 1 - i), -(i + 1))
        gamma_metric = gamma_metric.to(self.device)

        other_action_log_prob = []
        if self.use_full_jsd:
            trainer_idxs = list(range(self.population_size))
        else:
            trainer_idxs = [random.randint(0, self.population_size - 1)]
        for idx in trainer_idxs:
            t = all_trainer[idx]
            _other_action_log_prob = []
            _rnn_states = np.expand_dims(rnn_states[0], 0)
            _rnn_states_critic = np.expand_dims(rnn_states_critic[0], 0)
            for i in range(episode_length):
                # reshape to (n_threads*n_agents, *)
                _, action_log_prob, _, _, _rnn_states = t.policy.evaluate_transitions(
                    share_obs=share_obs[i].reshape(-1, *share_obs_shape),
                    obs=obs[i].reshape(-1, *obs_shape),
                    rnn_states_actor=_rnn_states.reshape(-1, *rnn_states.shape[-2:]),
                    rnn_states_critic=_rnn_states_critic.reshape(-1, *rnn_states_critic.shape[-2:]),
                    action=actions[i].reshape(-1, *actions.shape[-1:]),
                    masks=masks[i].reshape(-1, *masks.shape[-1:]),
                    available_actions=(
                        available_actions[i].reshape(-1, *available_actions.shape[-1:])
                        if available_actions is not None
                        else None
                    ),
                )
                _other_action_log_prob.append(action_log_prob.unsqueeze(0))
            # shape: (1, episode_length, n_threads*n_agents, 1)
            _other_action_log_prob = torch.cat(_other_action_log_prob, dim=0).unsqueeze(0)
            other_action_log_prob.append(_other_action_log_prob)
        # shape: (1, episode_length, n_threads*n_agents, 1)
        other_action_log_prob = torch.cat(other_action_log_prob, dim=0).mean(dim=0, keepdim=True)

        # reshape to (episode_length*n_threds*n_agents, *)
        _, log_prob, _, _ = trainer.policy.evaluate_actions(
            share_obs.reshape(-1, *share_obs_shape),
            obs.reshape(-1, *obs_shape),
            rnn_states.reshape(-1, *rnn_states.shape[-2:]),
            rnn_states_critic.reshape(-1, *rnn_states_critic.shape[-2:]),
            actions.reshape(-1, *actions.shape[-1:]),
            masks.reshape(-1, *masks.shape[-1:]),
            available_actions=(
                available_actions.reshape(-1, *available_actions.shape[-1:]) if available_actions is not None else None
            ),
        )
        # shape: (1, episode_length, n_threads*n_agents, 1)
        log_prob = log_prob.reshape(self.all_args.episode_length, -1).unsqueeze(-1).unsqueeze(0)

        if self.use_full_jsd:
            all_action_log_prob = []
            all_action_log_prob.append(other_action_log_prob)
        else:
            all_action_log_prob = []
            all_action_log_prob.append(other_action_log_prob)
            all_action_log_prob.append(log_prob)
        # shape: (n_threads*n_agents, 1, episode_length, 1)
        other_action_log_prob = other_action_log_prob.permute(2, 0, 1, 3)

        # shape: (n_threads*n_agents, 1, episode_length, 1)
        log_prob = log_prob.permute(2, 0, 1, 3)

        # shape: (n_threads*n_agents, 1, episode_length, 1)
        gamma_prob = (log_prob * gamma_metric).sum(dim=-1, keepdim=True)

        # [num_trainer, episode_length, num_traj, 1] -> [num_traj, num_trainer, episode_length, 1]
        all_action_log_prob = torch.cat(all_action_log_prob, dim=0).permute(2, 0, 1, 3)

        # [num_traj, num_trainer, steps, steps]
        all_gamma_prob = all_action_log_prob * gamma_metric
        # [num_traj, num_trainer, steps, 1]
        all_gamma_prob = all_gamma_prob.sum(dim=-1, keepdim=True)
        # [num_traj, 1, steps, 1]
        all_gamma_prob = all_gamma_prob.exp().mean(dim=1, keepdim=True).log()
        # print(all_gamma_prob)

        other_gamma_prob = (other_action_log_prob * gamma_metric).sum(dim=-1, keepdim=True)

        # compute first term
        traj_ratio = (
            other_action_log_prob.detach().sum(dim=2, keepdim=True) - log_prob.detach().sum(dim=2, keepdim=True)
        ).exp()
        first_item = traj_ratio * gamma_prob.detach().exp() * gamma_prob
        first_item = first_item.mean()

        # compute second term
        factor = other_gamma_prob.detach().exp() - gamma_prob.detach() / self.population_size
        second_item = factor * log_prob.sum(dim=2, keepdim=True)
        second_item = second_item.mean()

        # [num_traj, 1, steps, 1]
        delta = all_gamma_prob - gamma_prob
        delta = -delta.mean()
        return self.entropy_alpha * (first_item + second_item), delta

    def train(self, sp_size):
        # Calculate JSD loss

        sp_size = self.n_rollout_threads * 2
        if self.stage == 1:
            all_trainer = []
            for trainer_name in self.active_trainers:
                if trainer_name != self.agent_name:
                    trainer = self.trainer_pool[trainer_name]
                    trainer.prep_training()
                    all_trainer.append(trainer)

            losses = []
            loss = 0
            jsd = 0
            for trainer_name in self.active_trainers:
                if not trainer_name.startswith(self.agent_name):
                    trainer = self.trainer_pool[trainer_name]
                    buffer = self.buffer_pool[trainer_name]
                    (
                        share_obs,
                        obs,
                        rnn_states,
                        rnn_states_critic,
                        actions,
                        _,
                        _,
                        _,
                        masks,
                        _,
                        _,
                        available_actions,
                    ) = buffer.all_traj()
                    traj = {
                        "share_obs": share_obs[:, :sp_size],
                        "obs": obs[:, :sp_size],
                        "rnn_states": rnn_states[:, :sp_size],
                        "rnn_states_critic": rnn_states_critic[:, :sp_size],
                        "actions": actions[:, :sp_size],
                        "masks": masks[:, :sp_size],
                        "available_actions": available_actions[:, :sp_size] if available_actions is not None else None,
                    }

                    _loss, _jsd = self.traj_jsd(all_trainer, trainer, traj)
                    losses.append(_loss)
                    jsd += _jsd
            jsd = jsd.detach().cpu().numpy() / len(self.population)
            for l in losses:
                loss += l
                l.backward()
            if not self.use_co_update:
                for trainer_name in self.active_trainers:
                    trainer = self.trainer_pool[trainer_name]
                    trainer.update_actor()

        # PPO training
        if self.stage == 1 and self.use_co_update:
            super().train(actor_zero_grad=False)
        else:
            super().train()

        if self.stage == 1:
            loss = loss.detach().cpu().numpy() / len(self.population)
            self.train_infos.update({f"average_jsd": jsd, "average_jsd_loss": loss})
        return copy.deepcopy(self.train_infos)
