from typing import Dict, List, Tuple, Union

import gfootball.env as football_env
import numpy as np
from gym import spaces

SHAPED_INFOS = [
    "pass",
    "actual_pass",
    "shot",
    "slide",
    "catch",
    "assist",
    "possession",
    "score",
]

from .multiagentenv import MultiAgentEnv
from .raw_feature_process import FeatureEncoder
from .reward_process import Rewarder
from .stats_process import StatsObserver


class FootballEnv(MultiAgentEnv):
    """Wrapper to make Google Research Football environment compatible"""

    def __init__(self, args, evaluation: bool = False, seed: int = None):
        assert args.representation in ["simple115v2", "simple115v2_custom"]
        self.num_agents = args.num_agents
        self.scenario_name = args.scenario_name
        # self.episode_length = args.episode_length

        self.representation = args.representation
        self.obs_last_action = args.obs_last_action

        self.share_reward = args.share_reward
        self.reward_shaping = args.reward_shaping
        self.reward_config = []
        self.use_hsp = args.use_hsp
        for a_i in range(self.num_agents):
            if self.use_hsp and a_i == 0:
                assert isinstance(args.w0, dict) and len(args.w0) > 0
                self.reward_config.append(args.w0)
            else:
                assert isinstance(args.reward_config, dict) and len(args.reward_config) > 0
                self.reward_config.append(args.reward_config)

        # logger.debug(f"GRF env: reward config {self.reward_config}")

        self.use_available_actions = getattr(args, "use_available_actions", True)

        self.random_index = args.random_index  # agent id 0 is the bf agent
        self.agent_idxs = np.arange(self.num_agents)

        self.use_agent_policy_id = dict(args._get_kwargs()).get(
            "use_agent_policy_id", False
        )  # Add policy id for loaded policy
        self.agent_policy_id = [-1.0 for _ in range(self.num_agents)]

        assert not self.use_hsp or (self.random_index and not self.share_reward)

        assert not "academy" in args.scenario_name or args.action_set == "default"
        assert "academy" not in args.scenario_name or "win_reward" not in args.reward_config

        representation_trans = {
            "simple115v2": "simple115v2",
            "simple115v2_custom": "simple115v2",
        }

        # make env
        if evaluation:
            self.env = football_env.create_environment(
                env_name=args.scenario_name,
                representation=representation_trans[args.representation],
                rewards=args.rewards,
                number_of_left_players_agent_controls=args.num_agents,
                number_of_right_players_agent_controls=0,
                # setting seed, but actually it will be random.randint if not provided
                # if 'game_engine_random_seed' not in self._config._values:
                #    self._config.set_scenario_value('game_engine_random_seed',
                #    random.randint(0, 2000000000))
                other_config_options={
                    "action_set": args.action_set,
                    "game_engine_random_seed": seed,
                },
                # render=True,
                # write_video=True,
                # write_full_episode_dumps=True,
                # dump_frequency=1,
                # logdir=Path(args.exp_dir) / "replays",
            )
            # print("Evaluation mode GRF")
        else:
            self.env = football_env.create_environment(
                env_name=args.scenario_name,
                representation=representation_trans[args.representation],
                rewards=args.rewards,
                number_of_left_players_agent_controls=args.num_agents,
                number_of_right_players_agent_controls=0,
                other_config_options={
                    "action_set": args.action_set,
                    "game_engine_random_seed": seed,
                },
            )

        self.action_n = self.env.action_space[0].n
        self.action_set = args.action_set
        self.num_left_agents = len(list(self.env.unwrapped._env._env.config.left_team))
        self.num_right_agents = len(list(self.env.unwrapped._env._env.config.right_team))
        assert self.num_agents <= self.num_left_agents

        self.max_steps = self.env.unwrapped.observation()[0]["steps_left"]

        self.feature_encoder = FeatureEncoder(
            self.num_agents,
            self.num_left_agents,
            self.num_right_agents,
            self.action_n,
            self.representation,
            self.env.observation_space,
            self.use_agent_policy_id,
        )

        self.reward_encoder = Rewarder(self.reward_config)

        self.env.reset()
        self.player_ids = [obs["active"] for obs in self.env.unwrapped.observation()]
        self.statsobserver = StatsObserver(self.num_agents, self.player_ids)

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        if self.num_agents == 1:
            self.action_space.append(self.env.action_space)
            self.observation_space = self.feature_encoder.observation_space
            self.share_observation_space = self.feature_encoder.share_observation_space
        else:
            for idx in range(self.num_agents):
                self.action_space.append(spaces.Discrete(n=self.env.action_space[idx].n))
            self.observation_space = self.feature_encoder.observation_space
            self.share_observation_space = self.feature_encoder.share_observation_space
        self.prev_obs_dict_list: List[Dict] = None
        if self.obs_last_action:
            self.last_action = np.zeros((self.num_agents, self.action_n))

        self.shaped_returns = [0 for _ in range(self.num_agents)]
        self.step_i = 0

    def encode_obs(
        self, obs_list: List[np.array], obs_dict_list: List[Dict]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        new_obs_list = []
        avail_list = []
        for _i, (_obs, _obs_dict) in enumerate(zip(obs_list, obs_dict_list)):
            _obs, _avail = self.feature_encoder.encode_each(_obs, _obs_dict)
            if self.obs_last_action:
                _obs = np.concatenate((_obs, self.last_action[_i]), axis=-1)
            new_obs_list.append(_obs)
            avail_list.append(_avail)
        return new_obs_list, avail_list

    def reset(self, flag: bool = True):
        obs_list = self.env.reset()
        obs_dict_list = self.env.unwrapped.observation()
        self.prev_obs_dict_list = obs_dict_list.copy()
        if self.use_agent_policy_id:
            state_list = self.feature_encoder.encode_state(obs_dict_list, self.agent_policy_id)
        else:
            state_list = self.feature_encoder.encode_state(obs_dict_list)

        # logger.debug(obs_list)
        obs_list, avail_list = self.encode_obs(obs_list, obs_dict_list)
        self.statsobserver.reset()
        if self.random_index:
            self.agent_idxs = np.random.permutation(self.num_agents)
            # logger.debug(self.agent_idxs)

        # logger.debug(obs_list)
        obs_list = np.stack(obs_list, axis=0)[self.agent_idxs]
        obs = tuple(_obs for _obs in obs_list)

        state = np.stack(state_list, axis=0)[self.agent_idxs]

        if self.use_available_actions:
            avail = np.stack(avail_list, axis=0)[self.agent_idxs]
        else:
            avail = np.ones(self.num_agents, self.action_n)

        self.shaped_returns = [0 for _ in range(self.num_agents)]
        self.step_i = 0

        return obs, state, avail

    def step(self, action: Union[np.ndarray, List[int]]):
        self.step_i += 1

        if isinstance(action, list):
            action = np.array(action)
        if action.ndim == 2:
            action = action.squeeze(-1)
        action = action[np.argsort(self.agent_idxs)]
        if self.obs_last_action:
            self.last_action = np.eye(self.action_n)[np.array(action)]
        obs_list, reward, done, info = self.env.step(action)
        obs_list = self._obs_wrapper(obs_list)
        obs_dict_list = self.env.unwrapped.observation()
        reward = reward.reshape(self.num_agents, 1)
        one_step_stats = self.statsobserver.observe(action, self.prev_obs_dict_list, obs_dict_list)
        self.prev_obs_dict_list = obs_dict_list.copy()
        if self.use_agent_policy_id:
            state_list = self.feature_encoder.encode_state(obs_dict_list, self.agent_policy_id)
        else:
            state_list = self.feature_encoder.encode_state(obs_dict_list)
        obs_list, avail_list = self.encode_obs(obs_list, obs_dict_list)
        # if obs_dict_list[0]["score"][0] > obs_dict_list[0]["score"][1]:
        #     logger.success(f"win!, done: {done} info: {info}")
        # logger.debug(one_step_stats)
        info = self._info_wrapper(info, one_step_stats, done)
        for idx in range(self.num_agents):
            reward[idx] = self.reward_encoder.calc_reward(
                reward[idx], info, self.prev_obs_dict_list[idx], obs_list[idx], idx
            )
            self.shaped_returns[idx] += reward[idx]

        if self.share_reward:
            global_reward = np.mean(reward)
            reward = [[global_reward]] * self.num_agents
        # if done or self.step_i >= self.episode_length:
        info["episode"]["shaped_return"] = self.shaped_returns

        obs_list = np.stack(obs_list, axis=0)[self.agent_idxs]
        obs = tuple(_obs for _obs in obs_list)

        state = np.stack(state_list, axis=0)[self.agent_idxs]

        reward = np.array(reward)[self.agent_idxs]

        done = np.array([done] * self.num_agents)

        if self.use_available_actions:
            avail = np.stack(avail_list, axis=0)[self.agent_idxs]
        else:
            avail = np.ones(self.num_agents, self.action_n)

        return obs, state, reward, done, info, avail

    def _obs_wrapper(self, obs: Union[np.ndarray, List[np.ndarray]]):
        if self.num_agents == 1:
            return obs[np.newaxis, :]
        else:
            return obs

    def _info_wrapper(self, info: dict, stats: Dict[str, List[int]], done: bool) -> Dict:
        obs_dict_list = self.env.unwrapped.observation()
        obs_dict = obs_dict_list[0]
        bad_transition = False
        if done:
            if obs_dict["steps_left"] == 0:
                if (obs_dict["score"][0] <= obs_dict["score"][1]) and (
                    all([self.reward_config[a_i].get("lose_reward", 0) != 0 for a_i in range(self.num_agents)])
                    and self.reward_shaping
                ):
                    bad_transition = False
                elif (obs_dict["score"][0] > obs_dict["score"][1]) and (
                    "academy" in self.scenario_name
                    or all([self.reward_config[a_i].get("win_reward", 0) != 0 for a_i in range(self.num_agents)])
                    and self.reward_shaping
                ):  # academy, win the game
                    bad_transition = False
                else:
                    bad_transition = True

        info["bad_transition"] = bad_transition
        info.update(stats)
        # if done or self.step_i >= self.episode_length:
        info["episode"] = self.statsobserver.get_stats()

        if done:
            info["episode"]["length"] = self.step_i

        return info

    def _set_agent_policy_id(self, agent_policy_id):
        self.agent_policy_id = agent_policy_id

    def close(self):
        self.env.close()
