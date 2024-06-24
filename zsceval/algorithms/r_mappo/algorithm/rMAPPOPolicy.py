import torch
from loguru import logger

from zsceval.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from zsceval.utils.util import update_linear_schedule


class ExDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class R_MAPPOPolicy:
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.data_parallel = getattr(args, "data_parallel", False)

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def to_parallel(self):
        if self.data_parallel:
            logger.warning(
                f"Use Data Parallel for Forwarding in devices {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
            )
            for name, children in self.actor.named_children():
                setattr(self.actor, name, ExDataParallel(children))
            for name, children in self.critic.named_children():
                setattr(self.critic, name, ExDataParallel(children))

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

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
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, share_obs, rnn_states_critic, masks, task_id=None):
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values

    def evaluate_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        task_id=None,
    ):
        (
            action_log_probs,
            dist_entropy,
            policy_values,
        ) = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values, action_log_probs, dist_entropy, policy_values

    def evaluate_transitions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        task_id=None,
    ):
        (
            action_log_probs,
            dist_entropy,
            policy_values,
            rnn_states_actor,
        ) = self.actor.evaluate_transitions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values, action_log_probs, dist_entropy, policy_values, rnn_states_actor

    def act(
        self,
        obs,
        rnn_states_actor,
        masks,
        available_actions=None,
        deterministic=False,
        **kwargs,
    ):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

    def get_probs(self, obs, rnn_states_actor, masks, available_actions=None):
        action_probs, rnn_states_actor = self.actor.get_probs(
            obs, rnn_states_actor, masks, available_actions=available_actions
        )
        return action_probs, rnn_states_actor

    def get_action_log_probs(
        self,
        obs,
        rnn_states_actor,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        action_log_probs, _, _, rnn_states_actor = self.actor.get_action_log_probs(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        return action_log_probs, rnn_states_actor

    def load_checkpoint(self, ckpt_path):
        if "actor" in ckpt_path:
            self.actor.load_state_dict(torch.load(ckpt_path["actor"], map_location=self.device))
        if "critic" in ckpt_path:
            self.critic.load_state_dict(torch.load(ckpt_path["critic"], map_location=self.device))

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()
