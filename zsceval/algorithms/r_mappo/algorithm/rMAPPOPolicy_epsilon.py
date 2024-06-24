import torch

from zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy


class R_MAPPOPolicy_Epsilon(R_MAPPOPolicy):
    def __init__(
        self,
        args,
        obs_space,
        share_obs_space,
        act_space,
        epsilon: float = 0.0,
        device=torch.device("cpu"),
    ):
        super().__init__(args, obs_space, share_obs_space, act_space, device)
        self.epsilon = epsilon

    def get_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False,
        task_id=None,
        **kwargs,
    ):
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        values, rnn_states_critic = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        if self.epsilon > 0:
            is_random_actions = torch.rand(actions.shape, device=actions.device)
            random_actions = torch.tensor(
                [self.act_space.sample() for _ in range(actions.numel())],
                device=actions.device,
            ).reshape(actions.shape)
            # logger.info(f"random action factor: {is_random_actions.mean()}")
            # logger.info(f"devices {random_actions.device} {actions.device}")
            actions = torch.where(is_random_actions < self.epsilon, random_actions, actions)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
