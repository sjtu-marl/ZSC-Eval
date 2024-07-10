import numpy as np
import torch


def _t2n(x):
    if not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().numpy()


class EvalPolicy:
    """A policy for evaluation.
    It maintains hidden states on its own.
    For usage, 'reset' before every eval episode, 'register_control_agents' to indicate agents controlled by this policy and 'step' means an env step.
    """

    def __init__(self, args, policy):
        self.args = args
        self.policy = policy
        self._control_agents = []
        self._map_a2id = dict()

        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_N = args.recurrent_N
        self.hidden_size = args.hidden_size

    @property
    def default_hidden_state(self):
        return np.zeros((self.recurrent_N, self.hidden_size), dtype=np.float32)

    @property
    def control_agents(self):
        return self._control_agents

    def reset(self, num_envs, num_agents):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self._control_agents = []
        self._map_a2id = dict()
        self._rnn_states = dict()

    def reset_state(self, e, a):
        assert (e, a) in self._control_agents
        self._rnn_states[(e, a)] = self.default_hidden_state

    def register_control_agent(self, e, a):
        if (e, a) not in self._control_agents:
            self._control_agents.append((e, a))
            self._map_a2id[(e, a)] = len(self._control_agents)
            self._rnn_states[(e, a)] = self.default_hidden_state

    def step(self, obs, agents, deterministic=False, masks=None, **kwargs):
        num = len(agents)
        assert obs.shape[0] == num
        rnn_states = [self._rnn_states[ea] for ea in agents]
        if masks is None:
            masks = np.ones((num, 1), dtype=np.float32)
        action, rnn_states = self.policy.act(
            obs, np.stack(rnn_states, axis=0), masks, deterministic=deterministic, **kwargs
        )
        for ea, rnn_state in zip(agents, _t2n(rnn_states)):
            self._rnn_states[ea] = rnn_state
        return _t2n(action)

    def to(self, device):
        self.policy.to(device)

    def prep_rollout(self):
        self.policy.prep_rollout()
