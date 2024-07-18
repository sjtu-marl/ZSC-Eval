import os
import pickle
import pprint

import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from loguru import logger

from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
    BASE_REW_SHAPING_PARAMS,
    SHAPED_INFOS,
    OvercookedGridworld,
)
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_trajectory import (
    DEFAULT_TRAJ_KEYS,
    TIMESTEP_TRAJ_KEYS,
)
from zsceval.envs.overcooked.overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MediumLevelPlanner,
)
from zsceval.envs.overcooked.overcooked_ai_py.utils import mean_and_std_err
from zsceval.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import (
    StateVisualizer,
)
from zsceval.envs.overcooked.script_agent.script_agent import SCRIPT_AGENTS
from zsceval.utils.train_util import setup_seed

DEFAULT_ENV_PARAMS = {"horizon": 400}

MAX_HORIZON = 1e10


class OvercookedEnv:
    """An environment wrapper for the OvercookedGridworld Markov Decision Process.

    The environment keeps track of the current state of the agent, updates
    it as the agent takes actions, and provides rewards to the agent.
    """

    def __init__(
        self,
        mdp,
        start_state_fn=None,
        horizon=MAX_HORIZON,
        debug=False,
        evaluation: bool = False,
        use_random_player_pos: bool = False,
        use_random_terrain_state: bool = False,
        num_initial_state: int = 5,
        replay_return_threshold: float = 0.75,
    ):
        """
        mdp (OvercookedGridworld or function): either an instance of the MDP or a function that returns MDP instances
        start_state_fn (OvercookedState): function that returns start state for the MDP, called at each environment reset
        horizon (float): number of steps before the environment returns done=True
        """
        if isinstance(mdp, OvercookedGridworld):
            self.mdp_generator_fn = lambda: mdp
        elif callable(mdp) and isinstance(mdp(), OvercookedGridworld):
            self.mdp_generator_fn = mdp
        else:
            raise ValueError("Mdp should be either OvercookedGridworld instance or a generating function")

        self.horizon = horizon
        self.start_state_fn = start_state_fn
        self.evaluation = evaluation
        self.use_random_player_pos = use_random_player_pos
        self.use_random_terrain_state = use_random_terrain_state
        self.reset()

        if self.horizon >= MAX_HORIZON and self.state.order_list is None and debug:
            print("Environment has (near-)infinite horizon and no terminal states")

    def __repr__(self):
        """Standard way to view the state of an environment programatically
        is just to print the Env object"""
        return self.mdp.state_string(self.state)

    def print_state_transition(self, a_t, r_t, info):
        print(
            "Timestep: {}\nJoint action taken: {} \t Reward: {} + shape * {} \n{}\n".format(
                self.t,
                tuple(Action.ACTION_TO_CHAR[a] for a in a_t),
                r_t,
                info["shaped_r"],
                self,
            )
        )

    @property
    def env_params(self):
        return {"start_state_fn": self.start_state_fn, "horizon": self.horizon}

    def display_states(self, *states):
        old_state = self.state
        for s in states:
            self.state = s
            print(self)
        self.state = old_state

    @staticmethod
    def print_state(mdp, s):
        e = OvercookedEnv(mdp, s)
        print(e)

    def copy(self):
        return OvercookedEnv(
            mdp=self.mdp.copy(),
            start_state_fn=self.start_state_fn,
            horizon=self.horizon,
        )

    def step(self, joint_action):
        """Performs a joint action, updating the environment state
        and providing a reward.

        On being done, stats about the episode are added to info:
            ep_sparse_r: the environment sparse reward, given only at soup delivery
            ep_shaped_r: the component of the reward that is due to reward shaped (excluding sparse rewards)
            ep_length: length of rollout
        """
        assert not self.is_done()
        self.t += 1
        next_state, mdp_infos = self.mdp.get_state_transition(self.state, joint_action)

        # Update game_stats
        self._update_game_stats(mdp_infos)

        # Update state and done
        self.state = next_state
        done = self.is_done()
        env_info = self._prepare_info_dict([{}, {}], mdp_infos)

        if done:
            self._add_episode_info(env_info)

        timestep_sparse_reward = sum(mdp_infos["sparse_reward_by_agent"])

        return next_state, timestep_sparse_reward, done, env_info

    def reset(self):
        """Resets the environment. Does NOT reset the agent."""
        self.mdp = self.mdp_generator_fn()
        if self.start_state_fn is None:
            self.state = self.mdp.get_standard_start_state()
        elif type(self.start_state_fn) in [float, int]:
            # p = np.random.uniform(0, 1)
            p = np.random.uniform(0, 1)
            if p <= self.start_state_fn and not self.evaluation:
                # logger.error("Random start state")
                self.state = self.mdp.get_random_start_state(self.use_random_terrain_state, self.use_random_player_pos)
            else:
                self.state = self.mdp.get_standard_start_state()
        else:
            self.state = self.start_state_fn()
        # assert self.mdp.start_player_positions == list(self.state.player_positions)
        self.cumulative_sparse_rewards = 0
        self.cumulative_shaped_rewards = 0
        self.t = 0

        rewards_dict = {
            "cumulative_sparse_rewards_by_agent": np.array([0] * self.mdp.num_players),
            "cumulative_shaped_rewards_by_agent": np.array([0] * self.mdp.num_players),
            "cumulative_category_rewards_by_agent": np.zeros((self.mdp.num_players, len(SHAPED_INFOS))),
        }

        self.game_stats = {**rewards_dict}

    def is_done(self):
        """Whether the episode is over."""
        return self.t >= self.horizon or self.mdp.is_terminal(self.state)

    def _prepare_info_dict(self, joint_agent_action_info, mdp_infos):
        """
        The normal timestep info dict will contain infos specifc to each agent's action taken,
        and reward shaping information.
        """
        # Get the agent action info, that could contain info about action probs, or other
        # custom user defined information
        env_info = {"agent_infos": [joint_agent_action_info[agent_idx] for agent_idx in range(self.mdp.num_players)]}
        # TODO: This can be further simplified by having all the mdp_infos copied over to the env_infos automatically
        env_info["sparse_r_by_agent"] = mdp_infos["sparse_reward_by_agent"]
        env_info["shaped_r_by_agent"] = mdp_infos["shaped_reward_by_agent"]
        env_info["shaped_info_by_agent"] = mdp_infos["shaped_info_by_agent"]
        env_info["phi_s"] = mdp_infos["phi_s"] if "phi_s" in mdp_infos else None
        env_info["phi_s_prime"] = mdp_infos["phi_s_prime"] if "phi_s_prime" in mdp_infos else None
        return env_info

    # MARK: info
    def _add_episode_info(self, env_info):
        env_info["episode"] = {
            "ep_game_stats": self.game_stats,
            "ep_sparse_r": sum(self.game_stats["cumulative_sparse_rewards_by_agent"]),
            "ep_shaped_r": sum(self.game_stats["cumulative_shaped_rewards_by_agent"]),
            "ep_sparse_r_by_agent": self.game_stats["cumulative_sparse_rewards_by_agent"],
            "ep_shaped_r_by_agent": self.game_stats["cumulative_shaped_rewards_by_agent"],
            "ep_category_r_by_agent": self.game_stats["cumulative_category_rewards_by_agent"],
            "ep_length": self.t,
        }
        return env_info

    def vectorize_shaped_info(self, shaped_info_by_agent):
        from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
            SHAPED_INFOS,
        )

        def vectorize(d: dict):
            # return np.array([v for k, v in d.items()])
            return np.array([d[k] for k in SHAPED_INFOS])

        shaped_info_by_agent = np.stack([vectorize(shaped_info) for shaped_info in shaped_info_by_agent])
        return shaped_info_by_agent

    def _update_game_stats(self, infos):
        """
        Update the game stats dict based on the events of the current step
        NOTE: the timer ticks after events are logged, so there can be events from time 0 to time self.horizon - 1
        """
        self.game_stats["cumulative_sparse_rewards_by_agent"] += np.array(infos["sparse_reward_by_agent"])
        self.game_stats["cumulative_shaped_rewards_by_agent"] += np.array(infos["shaped_reward_by_agent"])
        self.game_stats["cumulative_category_rewards_by_agent"] += self.vectorize_shaped_info(
            infos["shaped_info_by_agent"]
        )

        """for event_type, bool_list_by_agent in infos["event_infos"].items():
            # For each event type, store the timestep if it occurred
            event_occurred_by_idx = [int(x) for x in bool_list_by_agent]
            for idx, event_by_agent in enumerate(event_occurred_by_idx):
                if event_by_agent:
                    self.game_stats[event_type][idx].append(self.state.timestep)"""

    def execute_plan(self, start_state, joint_action_plan, display=False):
        """Executes action_plan (a list of joint actions) from a start
        state in the mdp and returns the resulting state."""
        self.state = start_state
        done = False
        if display:
            print(f"Starting state\n{self}")
        for joint_action in joint_action_plan:
            self.step(joint_action)
            done = self.is_done()
            if display:
                print(self)
            if done:
                break
        successor_state = self.state
        self.reset()
        return successor_state, done

    def run_agents(self, agent_pair, include_final_state=False, display=False, display_until=np.Inf):
        """
        Trajectory returned will a list of state-action pairs (s_t, joint_a_t, r_t, done_t).
        """
        assert (
            self.cumulative_sparse_rewards == self.cumulative_shaped_rewards == 0
        ), "Did not reset environment before running agents"
        trajectory = []
        done = False

        if display:
            print(self)
        while not done:
            s_t = self.state
            a_t = agent_pair.joint_action(s_t)

            # Break if either agent is out of actions
            if any([a is None for a in a_t]):
                break

            s_tp1, r_t, done, info = self.step(a_t)
            trajectory.append((s_t, a_t, r_t, done))

            if display and self.t < display_until:
                self.print_state_transition(a_t, r_t, info)

        assert len(trajectory) == self.t, f"{len(trajectory)} vs {self.t}"

        # Add final state
        if include_final_state:
            trajectory.append((s_tp1, (None, None), 0, True))

        return (
            np.array(trajectory),
            self.t,
            self.cumulative_sparse_rewards,
            self.cumulative_shaped_rewards,
        )

    def get_rollouts(
        self,
        agent_pair,
        num_games,
        display=False,
        final_state=False,
        agent_idx=0,
        reward_shaping=0.0,
        display_until=np.Inf,
        info=True,
    ):
        """
        Simulate `num_games` number rollouts with the current agent_pair and returns processed
        trajectories.

        Only returns the trajectories for one of the agents (the actions _that_ agent took),
        namely the one indicated by `agent_idx`.

        Returning excessive information to be able to convert trajectories to any required format
        (baselines, stable_baselines, etc)

        NOTE: standard trajectories format used throughout the codebase
        """
        trajectories = {
            # With shape (n_timesteps, game_len), where game_len might vary across games:
            "ep_observations": [],
            "ep_actions": [],
            "ep_rewards": [],  # Individual dense (= sparse + shaped * rew_shaping) reward values
            "ep_dones": [],  # Individual done values
            # With shape (n_episodes, ):
            "ep_returns": [],  # Sum of dense and sparse rewards across each episode
            "ep_returns_sparse": [],  # Sum of sparse rewards across each episode
            "ep_lengths": [],  # Lengths of each episode
            "mdp_params": [],  # Custom MDP params to for each episode
            "env_params": [],  # Custom Env params for each episode
        }

        for _ in tqdm.trange(num_games):
            agent_pair.set_mdp(self.mdp)

            trajectory, time_taken, tot_rews_sparse, tot_rews_shaped = self.run_agents(
                agent_pair,
                display=display,
                include_final_state=final_state,
                display_until=display_until,
            )
            obs, actions, rews, dones = (
                trajectory.T[0],
                trajectory.T[1],
                trajectory.T[2],
                trajectory.T[3],
            )
            trajectories["ep_observations"].append(obs)
            trajectories["ep_actions"].append(actions)
            trajectories["ep_rewards"].append(rews)
            trajectories["ep_dones"].append(dones)
            trajectories["ep_returns"].append(tot_rews_sparse + tot_rews_shaped * reward_shaping)
            trajectories["ep_returns_sparse"].append(tot_rews_sparse)
            trajectories["ep_lengths"].append(time_taken)
            trajectories["mdp_params"].append(self.mdp.mdp_params)
            trajectories["env_params"].append(self.env_params)

            self.reset()
            agent_pair.reset()

        mu, se = mean_and_std_err(trajectories["ep_returns"])
        if info:
            print(
                "Avg reward {:.2f} (std: {:.2f}, se: {:.2f}) over {} games of avg length {}".format(
                    mu,
                    np.std(trajectories["ep_returns"]),
                    se,
                    num_games,
                    np.mean(trajectories["ep_lengths"]),
                )
            )

        # Converting to numpy arrays
        trajectories = {k: np.array(v) for k, v in trajectories.items()}
        return trajectories


class Overcooked(gym.Env):
    """
    Wrapper for the Env class above that is SOMEWHAT compatible with the standard gym API.

    NOTE: Observations returned are in a dictionary format with various information that is
    necessary to be able to handle the multi-agent nature of the environment. There are probably
    better ways to handle this, but we found this to work with minor modifications to OpenAI Baselines.

    NOTE: The index of the main agent in the mdp is randomized at each reset of the environment, and
    is kept track of by the self.agent_idx attribute. This means that it is necessary to pass on this
    information in the output to know for which agent index featurizations should be made for other agents.

    For example, say one is training A0 paired with A1, and A1 takes a custom state featurization.
    Then in the runner.py loop in OpenAI Baselines, we will get the lossless encodings of the state,
    and the true Overcooked state. When we encode the true state to feed to A1, we also need to know
    what agent index it has in the environment (as encodings will be index dependent).
    """

    env_name = "Overcooked-v0"

    def __init__(
        self,
        all_args,
        run_dir,
        baselines_reproducible=True,
        featurize_type=("ppo", "ppo"),
        stuck_time=4,
        rank=None,
        evaluation=False,
    ):
        """
        base_env: OvercookedEnv
        featurize_fn(mdp, state): fn used to featurize states returned in the 'both_agent_obs' field
        """
        if baselines_reproducible:
            # NOTE:
            # This will cause all agent indices to be chosen in sync across simulation
            # envs (for each update, all envs will have index 0 or index 1).
            # This is to prevent the randomness of choosing agent indexes
            # from leaking when using subprocess-vec-env in baselines (which
            # seeding does not reach) i.e. having different results for different
            # runs with the same seed.
            # The effect of this should be negligible, as all other randomness is
            # controlled by the actual run seeds
            np.random.seed(0)
        self.all_args = all_args
        self.agent_idx = 0
        self._initial_reward_shaping_factor = all_args.initial_reward_shaping_factor
        self.reward_shaping_factor = all_args.reward_shaping_factor
        self.reward_shaping_horizon = all_args.reward_shaping_horizon
        self.use_phi = all_args.use_phi
        self.use_hsp = all_args.use_hsp
        self.store_traj = getattr(all_args, "store_traj", False)
        self.rank = rank
        self.random_index = all_args.random_index
        if self.use_hsp:
            self.w0 = self.string2array(all_args.w0)
            self.w1 = self.string2array(all_args.w1)
            w_dict = {"w0": f"{self.w0}", "w1": f"{self.w1}"}
            logger.debug("hsp weights:\n" + pprint.pformat(w_dict, compact=True, width=120))
            self.cumulative_hidden_reward = np.zeros(2)
        self.use_available_actions = getattr(all_args, "use_available_actions", True)
        self.use_render = all_args.use_render
        self.num_agents = all_args.num_agents
        self.layout_name = all_args.layout_name
        self.episode_length = all_args.episode_length
        self.random_start_prob = getattr(all_args, "random_start_prob", 0.0)
        self.stuck_time = stuck_time
        self.history_sa = []
        self.traj_num = 0
        self.step_count = 0
        self.run_dir = run_dir
        mdp_params = {"layout_name": all_args.layout_name, "start_order_list": None}
        # MARK: use reward shaping
        mdp_params.update({"rew_shaping_params": BASE_REW_SHAPING_PARAMS})
        env_params = {
            "horizon": all_args.episode_length,
            "evaluation": evaluation,
            "use_random_player_pos": all_args.use_random_player_pos,
            "use_random_terrain_state": all_args.use_random_terrain_state,
            "num_initial_state": all_args.num_initial_state,
            "replay_return_threshold": all_args.replay_return_threshold,
        }
        self.mdp_fn = lambda: OvercookedGridworld.from_layout_name(**mdp_params)
        self.base_mdp = self.mdp_fn()
        self.base_env = OvercookedEnv(
            self.mdp_fn,
            start_state_fn=self.random_start_prob if self.random_start_prob > 0 else None,
            **env_params,
        )
        self.mlp = MediumLevelPlanner.from_pickle_or_compute(
            mdp=self.base_mdp, mlp_params=NO_COUNTERS_PARAMS, force_compute=False
        )
        self.use_agent_policy_id = dict(all_args._get_kwargs()).get(
            "use_agent_policy_id", False
        )  # Add policy id for loaded policy
        self.agent_policy_id = [-1.0 for _ in range(self.num_agents)]
        self.use_timestep_feature = all_args.use_timestep_feature
        self.featurize_fn_ppo = lambda state: self.base_mdp.lossless_state_encoding(
            state,
            add_timestep=self.use_timestep_feature,
            horizon=self.episode_length,
            add_identity=all_args.use_identity_feature,
        )  # Encoding obs for PPO
        self.featurize_fn_bc = lambda state: self.base_mdp.featurize_state(state)  # Encoding obs for BC
        self.featurize_fn_mapping = {
            "ppo": self.featurize_fn_ppo,
            "bc": self.featurize_fn_bc,
        }
        self.reset_featurize_type(featurize_type=featurize_type)  # default agents are both ppo

        if self.all_args.algorithm_name == "population":
            assert not self.random_index
            self.script_agent = [None, None]
            for player_idx, policy_name in enumerate([all_args.agent0_policy_name, all_args.agent1_policy_name]):
                if policy_name.startswith("script:"):
                    self.script_agent[player_idx] = SCRIPT_AGENTS[policy_name[7:]]()
                    self.script_agent[player_idx].reset(self.base_env.mdp, self.base_env.state, player_idx)
        else:
            self.script_agent = [None, None]

    def reset_featurize_type(self, featurize_type=("ppo", "ppo")):
        assert len(featurize_type) == 2
        self.featurize_type = featurize_type
        self.featurize_fn = lambda state: [
            self.featurize_fn_mapping[f](state)[i] * (255 if f == "ppo" else 1)
            for i, f in enumerate(self.featurize_type)
        ]

        # reset observation_space, share_observation_space and action_space
        self.observation_space = []
        self.share_observation_space = []
        self.action_space = []
        self._setup_observation_space()
        for i in range(2):
            self.observation_space.append(self._observation_space(featurize_type[i]))
            self.action_space.append(gym.spaces.Discrete(len(Action.ALL_ACTIONS)))
            self.share_observation_space.append(self._setup_share_observation_space())

    def _anneal(self, start_v, curr_t, end_t, end_v=0, start_t=0):
        if end_t == 0:
            # No annealing if horizon is zero
            return start_v
        else:
            off_t = curr_t - start_t
            # Calculate the new value based on linear annealing formula
            fraction = max(1 - float(off_t) / (end_t - start_t), 0)
            return fraction * start_v + (1 - fraction) * end_v

    def onehot2idx(self, onehot):
        idx = []
        for a in onehot:
            idx.append(np.argmax(a))
        return idx

    def string2array(self, weight):
        w = []
        for s in weight.split(","):
            w.append(float(s))
        return np.array(w)

    def _action_convertor(self, action):
        return [a[0] for a in list(action)]

    def _observation_space(self, featurize_type):
        return {"ppo": self.ppo_observation_space, "bc": self.bc_observation_space}[featurize_type]

    def _setup_observation_space(self):
        dummy_state = self.base_env.mdp.get_standard_start_state()

        # ppo observation
        # featurize_fn_ppo = lambda state: self.base_mdp.lossless_state_encoding(state)
        featurize_fn_ppo = self.featurize_fn_ppo
        obs_shape = featurize_fn_ppo(dummy_state)[0].shape
        high = np.ones(obs_shape) * float("inf")
        low = np.ones(obs_shape) * 0
        self.ppo_observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)

        # bc observation
        featurize_fn_bc = lambda state: self.base_mdp.featurize_state(state, self.mlp)
        obs_shape = featurize_fn_bc(dummy_state)[0].shape
        high = np.ones(obs_shape) * 100
        low = np.ones(obs_shape) * -100
        self.bc_observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)

    def _setup_share_observation_space(self):
        dummy_state = self.base_env.mdp.get_standard_start_state()
        share_obs_shape = self.featurize_fn_ppo(dummy_state)[0].shape
        if self.use_agent_policy_id:
            share_obs_shape = [
                share_obs_shape[0],
                share_obs_shape[1],
                share_obs_shape[2] + 1,
            ]
        share_obs_shape = [
            share_obs_shape[0],
            share_obs_shape[1],
            share_obs_shape[2] * self.num_agents,
        ]
        high = np.ones(share_obs_shape) * float("inf")
        low = np.ones(share_obs_shape) * 0

        return gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)

    def _set_agent_policy_id(self, agent_policy_id):
        self.agent_policy_id = agent_policy_id

    def _gen_share_observation(self, state):
        share_obs = list(self.featurize_fn_ppo(state))
        if self.agent_idx == 1:
            share_obs = [share_obs[1], share_obs[0]]
        if self.use_agent_policy_id:
            for a in range(self.num_agents):
                share_obs[a] = np.concatenate(
                    [
                        share_obs[a],
                        np.ones((*share_obs[a].shape[:2], 1), dtype=np.float32) * self.agent_policy_id[a],
                    ],
                    axis=-1,
                )
        share_obs0 = np.concatenate([share_obs[0], share_obs[1]], axis=-1) * 255
        share_obs1 = np.concatenate([share_obs[1], share_obs[0]], axis=-1) * 255
        return np.stack([share_obs0, share_obs1], axis=0)  # shape (2, *obs_shape)

    def _get_available_actions(self):
        available_actions = np.ones((self.num_agents, len(Action.ALL_ACTIONS)), dtype=np.uint8)
        if self.use_available_actions:
            state = self.base_env.state
            interact_index = Action.ACTION_TO_INDEX["interact"]
            for agent_idx in range(self.num_agents):
                player = state.players[agent_idx]
                pos = player.position
                o = player.orientation
                for move_i, move in enumerate(Direction.ALL_DIRECTIONS):
                    new_pos = Action.move_in_direction(pos, move)
                    if new_pos not in self.base_mdp.get_valid_player_positions() and o == move:
                        available_actions[agent_idx, move_i] = 0

                i_pos = Action.move_in_direction(pos, o)
                terrain_type = self.base_mdp.get_terrain_type_at_pos(i_pos)

                if (
                    terrain_type == " "
                    or (
                        terrain_type == "X"
                        and (
                            (not player.has_object() and not state.has_object(i_pos))
                            or (player.has_object() and state.has_object(i_pos))
                        )
                    )
                    or (terrain_type in ["O", "T", "D"] and player.has_object())
                    or (
                        terrain_type == "P"
                        and (not player.has_object() or player.get_object().name not in ["dish", "onion", "tomato"])
                    )
                    or (terrain_type == "S" and (not player.has_object() or player.get_object().name not in ["soup"]))
                ):
                    available_actions[agent_idx, interact_index] = 0
                # assert available_actions[agent_idx].sum() > 0
        return available_actions

    def step(self, action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format

        returns:
            observation: formatted to be standard input for self.agent_idx's policy

        main_agent_index:
            While existing other agent like planning or human model, use an index to fix the main RL-policy agent.
            Default False for multi-agent training.
        """
        self.step_count += 1
        action = self._action_convertor(action)
        assert all(self.action_space[0].contains(a) for a in action), "{!r} ({}) invalid".format(
            action,
            type(action),
        )

        agent_action, other_agent_action = (Action.INDEX_TO_ACTION[a] for a in action)

        joint_action = [agent_action, other_agent_action]

        for a in range(self.num_agents):
            if self.script_agent[a] is not None:
                joint_action[a] = self.script_agent[a].step(self.base_env.mdp, self.base_env.state, a)
        joint_action = tuple(joint_action)

        if self.agent_idx == 1:
            joint_action = (other_agent_action, agent_action)

        if self.stuck_time > 0:
            self.history_sa[-1][1] = joint_action

        if self.store_traj:
            self.traj_to_store.append(joint_action)

        if self.use_phi:
            raise NotImplementedError
            next_state, sparse_reward, done, info = self.base_env.step(joint_action, display_phi=True)
            potential = info["phi_s_prime"] - info["phi_s"]
            dense_reward = (potential, potential)
            shaped_reward_p0 = sparse_reward + self.reward_shaping_factor * dense_reward[0]
            shaped_reward_p1 = sparse_reward + self.reward_shaping_factor * dense_reward[1]
        else:
            next_state, sparse_reward, done, info = self.base_env.step(joint_action)

            if self.use_hsp:
                from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
                    SHAPED_INFOS,
                )

                shaped_info = info["shaped_info_by_agent"]
                # vec_shaped_info = np.array(
                #     [[v for k, v in agent_info.items()] for agent_info in shaped_info]
                # ).astype(np.float32)
                vec_shaped_info = np.array(
                    [[agent_info[k] for k in SHAPED_INFOS] for agent_info in shaped_info]
                ).astype(np.float32)
                assert (
                    len(self.w0) == len(self.w1) == len(SHAPED_INFOS) + 1
                ), f"the weight shape must match {pprint.pformat(SHAPED_INFOS, compact=True)} with len {len(SHAPED_INFOS)}"
                dense_reward = info["shaped_r_by_agent"]
                #! no default reward shaping for hidden-utility agent
                if self.agent_idx == 0:
                    hidden_reward = (
                        np.dot(self.w0[:-1], vec_shaped_info[0]) + sparse_reward * self.w0[-1],
                        np.dot(self.w1[:-1], vec_shaped_info[1]) + sparse_reward * self.w1[-1],
                    )
                    shaped_reward_p0 = hidden_reward[0]
                    shaped_reward_p1 = hidden_reward[1] + self.reward_shaping_factor * dense_reward[1]
                else:
                    hidden_reward = (
                        np.dot(self.w1[:-1], vec_shaped_info[0]) + sparse_reward * self.w1[-1],
                        np.dot(self.w0[:-1], vec_shaped_info[1]) + sparse_reward * self.w0[-1],
                    )
                    shaped_reward_p0 = hidden_reward[0] + self.reward_shaping_factor * dense_reward[0]
                    shaped_reward_p1 = hidden_reward[1]
                self.cumulative_hidden_reward += hidden_reward
                #! BUG: reward_shaping_factor
                # logger.debug(f"reward shaping factor {self.reward_shaping_factor}")
                # shaped_reward_p0 = (
                #     hidden_reward[0] + self.reward_shaping_factor * dense_reward[0]
                # )
                # shaped_reward_p1 = (
                #     hidden_reward[1] + self.reward_shaping_factor * dense_reward[1]
                # )
            else:
                dense_reward = info["shaped_r_by_agent"]
                shaped_reward_p0 = sparse_reward + self.reward_shaping_factor * dense_reward[0]
                shaped_reward_p1 = sparse_reward + self.reward_shaping_factor * dense_reward[1]
        # TODO: log returned reward
        if self.store_traj:
            self.traj_to_store.append(info["shaped_info_by_agnet"])
            self.traj_to_store.append(self.base_env.state.to_dict())

        reward = [[shaped_reward_p0], [shaped_reward_p1]]

        if self.agent_idx == 1:
            reward = [[shaped_reward_p1], [shaped_reward_p0]]

        self.history_sa = self.history_sa[1:] + [
            [next_state, None],
        ]

        # vec_shaped_info_by_agent = self.base_env.vectorize_shaped_info(
        #     info["shaped_info_by_agent"]
        # )
        # vec_shaped_info_by_agent = np.concatenate(
        #     [
        #         vec_shaped_info_by_agent,
        #         (np.array(info["sparse_r_by_agent"]) > 0)
        #         .astype(np.int32)
        #         .reshape(2, 1),
        #     ],
        #     axis=-1,
        # )
        # if self.agent_idx == 1:
        #     vec_shaped_info_by_agent = np.stack(
        #         [vec_shaped_info_by_agent[1], vec_shaped_info_by_agent[0]]
        #     )
        # info["vec_shaped_info_by_agent"] = vec_shaped_info_by_agent

        # stuck
        stuck_info = []
        for agent_id in range(2):
            stuck, history_a = self.is_stuck(agent_id)
            if stuck:
                assert any([a not in history_a for a in Direction.ALL_DIRECTIONS]), history_a
                history_a_idxes = [Action.ACTION_TO_INDEX[a] for a in history_a]
                stuck_info.append([True, history_a_idxes])
            else:
                stuck_info.append([False, []])

        info["stuck"] = stuck_info

        if self.use_render:
            state = self.base_env.state
            self.traj["ep_states"][0].append(state)
            self.traj["ep_actions"][0].append(joint_action)
            self.traj["ep_rewards"][0].append(sparse_reward)
            self.traj["ep_dones"][0].append(done)
            self.traj["ep_infos"][0].append(info)
            if done:
                self.traj["ep_returns"].append(info["episode"]["ep_sparse_r"])
                self.traj["mdp_params"].append(self.base_mdp.mdp_params)
                self.traj["env_params"].append(self.base_env.env_params)
                self.render()
            # self.fake_render()

        if done:
            if self.store_traj:
                self._store_trajectory()
            if self.use_hsp:
                info["episode"]["ep_hidden_r_by_agent"] = self.cumulative_hidden_reward
            info["bad_transition"] = True
        else:
            info["bad_transition"] = False

        ob_p0, ob_p1 = self.featurize_fn(next_state)

        both_agents_ob = (ob_p0, ob_p1)
        if self.agent_idx == 1:
            both_agents_ob = (ob_p1, ob_p0)

        share_obs = self._gen_share_observation(self.base_env.state)
        done = [done, done]
        available_actions = self._get_available_actions()
        if self.agent_idx == 1:
            available_actions = np.stack([available_actions[1], available_actions[0]])

        return both_agents_ob, share_obs, reward, done, info, available_actions

    def anneal_reward_shaping_factor(self, timesteps):
        """
        Set the current reward shaping factor such that we anneal linearly until self.reward_shaping_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        new_factor = self._anneal(self._initial_reward_shaping_factor, timesteps, self.reward_shaping_horizon)
        self.set_reward_shaping_factor(new_factor)

    def set_reward_shaping_factor(self, factor):
        self.reward_shaping_factor = factor

    def reset(self, reset_choose=True):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """

        if reset_choose:
            self.traj_num += 1
            self.step_count = 0
            self.base_env.reset()

        if self.random_index:
            self.agent_idx = np.random.choice([0, 1])

        for a in range(self.num_agents):
            if self.script_agent[a] is not None:
                self.script_agent[a].reset(self.base_env.mdp, self.base_env.state, a)

        self.mdp = self.base_env.mdp
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        if self.stuck_time > 0:
            self.history_sa = [None for _ in range(self.stuck_time - 1)] + [[self.base_env.state, None]]

        both_agents_ob = (ob_p0, ob_p1)
        if self.agent_idx == 1:
            both_agents_ob = (ob_p1, ob_p0)

        if self.use_render:
            self.init_traj()
            # self.fake_render()

        if self.store_traj:
            self.traj_to_store = []
            self.traj_to_store.append(self.base_env.state.to_dict())

        if self.use_hsp:
            self.cumulative_hidden_reward = np.zeros(2)

        share_obs = self._gen_share_observation(self.base_env.state)
        available_actions = self._get_available_actions()
        if self.agent_idx == 1:
            available_actions = np.stack([available_actions[1], available_actions[0]])

        return both_agents_ob, share_obs, available_actions

    def is_stuck(self, agent_id):
        if self.stuck_time == 0 or None in self.history_sa:
            return False, []
        history_s = [sa[0] for sa in self.history_sa]
        history_a = [sa[1][agent_id] for sa in self.history_sa[:-1]]  # last action is None
        player_s = [s.players[agent_id] for s in history_s]
        pos_and_ors = [p.pos_and_or for p in player_s]
        cur_po = pos_and_ors[-1]
        if all([po[0] == cur_po[0] and po[1] == cur_po[1] for po in pos_and_ors]):
            return True, history_a
        return False, []

    def init_traj(self):
        # return
        self.traj = {k: [] for k in DEFAULT_TRAJ_KEYS}
        for key in TIMESTEP_TRAJ_KEYS:
            self.traj[key].append([])

    def render(self):
        # raise NotImplementedError
        # try:
        save_dir = f"{self.run_dir}/gifs/{self.layout_name}/traj_num_{self.traj_num}"
        save_dir = os.path.expanduser(save_dir)
        StateVisualizer().display_rendered_trajectory(self.traj, img_directory_path=save_dir, ipython_display=False)
        for img_path in os.listdir(save_dir):
            img_path = save_dir + "/" + img_path
        imgs = []
        imgs_dir = os.listdir(save_dir)
        imgs_dir = sorted(imgs_dir, key=lambda x: int(x.split(".")[0]))
        for img_path in imgs_dir:
            img_path = save_dir + "/" + img_path
            imgs.append(imageio.imread(img_path))
        imageio.mimsave(save_dir + f'/reward_{self.traj["ep_returns"][0]}.gif', imgs, duration=0.05)
        imgs_dir = os.listdir(save_dir)
        for img_path in imgs_dir:
            img_path = save_dir + "/" + img_path
            if "png" in img_path:
                os.remove(img_path)
        # except Exception as e:
        #    print('failed to render traj: ', e)

    def fake_render(self):
        state = self.base_env.state
        mdp = self.base_mdp
        """String representation of the current state"""
        players_dict = {player.position: player for player in state.players}

        plt.cla()
        plt.clf()
        plt.axis([0, len(mdp.terrain_mtx[0]), 0, len(mdp.terrain_mtx)])
        grid_string = ""
        for y, terrain_row in enumerate(mdp.terrain_mtx):
            for x, element in enumerate(terrain_row):
                plt_x = x + 0.5
                plt_y = len(mdp.terrain_mtx) - y - 0.5
                plt_str = ""
                if (x, y) in players_dict.keys():
                    player = players_dict[(x, y)]
                    orientation = player.orientation
                    assert orientation in Direction.ALL_DIRECTIONS

                    grid_string += Action.ACTION_TO_CHAR[orientation]
                    plt_str = Action.ACTION_TO_CHAR[orientation]
                    player_object = player.held_object
                    if player_object:
                        grid_string += player_object.name[:1]
                        plt_str += player_object.name[:1]
                    else:
                        player_idx_lst = [i for i, p in enumerate(state.players) if p.position == player.position]
                        assert len(player_idx_lst) == 1
                        grid_string += str(player_idx_lst[0])
                        plt_str += str(player_idx_lst[0])
                else:
                    if element == "X" and state.has_object((x, y)):
                        state_obj = state.get_object((x, y))
                        grid_string = grid_string + element + state_obj.name[:1]
                        plt_str += element + state_obj.name[:1]

                    elif element == "P" and state.has_object((x, y)):
                        soup_obj = state.get_object((x, y))
                        soup_type, num_items, cook_time = soup_obj.state
                        if soup_type == "onion":
                            grid_string += "ø"
                            plt_str += "ø"
                        elif soup_type == "tomato":
                            grid_string += "†"
                            plt_str += "†"
                        else:
                            raise ValueError()

                        if num_items == mdp.num_items_for_soup:
                            grid_string += str(cook_time)
                            plt_str += str(cook_time)

                        # NOTE: do not currently have terminal graphics
                        # support for cooking times greater than 3.
                        elif num_items == 2:
                            grid_string += "="
                            plt_str += "="
                        else:
                            grid_string += "-"
                            plt_str += "-"
                    else:
                        grid_string += element + " "
                        plt_str = element
                plt.text(plt_x, plt_y, plt_str, ha="center", fontsize=30, alpha=0.4)
            grid_string += "\n"

        if state.order_list is not None:
            grid_string += "Current orders: {}/{} are any's\n".format(
                len(state.order_list),
                len([order == "any" for order in state.order_list]),
            )
        save_dir = f"{self.run_dir}/gifs/{self.layout_name}/traj_num_{self.traj_num}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"{save_dir}/step_{self.step_count}.png")
        if self.step_count == self.episode_length:
            imgs = []
            for s in range(0, self.episode_length + 1):
                img = imageio.imread(f"{save_dir}/step_{s}.png")
                imgs.append(img)
            imageio.mimsave(f"{save_dir}/traj.gif", imgs, duration=0.1)
            imgs_dir = os.listdir(save_dir)
            for img_path in imgs_dir:
                img_path = save_dir + "/" + img_path
                if "png" in img_path:
                    os.remove(img_path)

        return grid_string

    def _store_trajectory(self):
        if not os.path.exists(f"{self.run_dir}/trajs/{self.layout_name}/"):
            os.makedirs(f"{self.run_dir}/trajs/{self.layout_name}/")
        save_dir = f"{self.run_dir}/trajs/{self.layout_name}/traj_{self.rank}_{self.traj_num}.pkl"
        pickle.dump(self.traj_to_store, open(save_dir, "wb"))

    def seed(self, seed):
        setup_seed(seed)
        super().seed(seed)
