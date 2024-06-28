from typing import Dict, List, Tuple

import numpy as np
from gym.spaces import Box


def do_flatten(obj):
    """Run flatten on either python list or numpy array."""
    if type(obj) == list:
        return np.array(obj).flatten()
    return obj.flatten()


class FeatureEncoder:
    """
    Process the feature and available actions
    """

    def __init__(
        self,
        num_controlled_agents: int,
        num_left_agents: int,
        num_right_agents: int,
        action_n: int,
        representation: str,
        origin_obs_space: Box,
        use_agent_policy_id: bool = False,
    ):
        self.action_n = action_n
        self.num_controlled_agents = num_controlled_agents
        self.num_left_agents = num_left_agents
        self.num_right_agents = num_right_agents
        self.representation = representation
        self.origin_obs_space = origin_obs_space
        self.use_agent_policy_id = use_agent_policy_id

    @property
    def observation_space(self) -> List[Box]:
        if "simple" in self.representation:
            if "custom" in self.representation:
                return [
                    Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(7 * self.num_left_agents + 6 * self.num_right_agents + 18,),
                        dtype=self.origin_obs_space.dtype,
                    )
                    for _ in range(self.num_controlled_agents)
                ]
            else:
                return [
                    Box(
                        low=self.origin_obs_space.low[idx],
                        high=self.origin_obs_space.high[idx],
                        shape=self.origin_obs_space.shape[1:],
                        dtype=self.origin_obs_space.dtype,
                    )
                    for idx in range(self.num_controlled_agents)
                ]
        else:
            raise NotImplementedError("Only support `simple115v2`-based")

    @property
    def share_observation_space(self) -> List[Box]:
        if "custom" in self.representation:
            share_obs_size = (
                4 * (self.num_left_agents + self.num_right_agents)
                + self.num_controlled_agents * self.num_left_agents
                + 16
            )
        else:
            share_obs_size = 104 + self.num_controlled_agents * 11
        if self.use_agent_policy_id:
            share_obs_size += self.num_controlled_agents
        return [
            Box(
                low=-np.inf,
                high=np.inf,
                shape=(share_obs_size,),
                dtype=self.origin_obs_space.dtype,
            )
            for _ in range(self.num_controlled_agents)
        ]

    def encode(self, obs_list: np.ndarray, obs_dict_list: List[Dict]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        feats = []
        avails = []
        assert len(obs_list) == len(obs_dict_list)
        for obs, obs_dict in zip(obs_list, obs_dict_list):
            feat, avail = self.encode_each(obs, obs_dict)
            feats.append(feat)
            avails.append(avail)
        return feats, avails

    def encode_state(self, obs_dict_list: List[Dict], agent_policy_id: List[float] = None) -> np.ndarray:
        state_list = []
        player_ids = np.array([obs_dict["active"] for obs_dict in obs_dict_list])
        for a_i in range(self.num_controlled_agents):
            obs_dict = obs_dict_list[a_i]
            if "custom" in self.representation:
                state = []
                # n1 * 2 - (x,y) coordinates of left team players
                # n1 * 2 - (x,y) direction of left team players
                # n2 * 2 - (x,y) coordinates of right team players
                # n2 * 2 - (x,y) direction of right team players

                # 3 - (x,y,z) - ball position
                # 3 - ball direction
                # 3 - one hot encoding of ball ownership (noone, left, right)

                # 7 - one hot encoding of `game_mode`
                # n0 * n1 - one hot encoding of which player is active  (controlled agent ids)
                # dim: 4*(n1+n2) + n0*n1 + 16
                state.extend(do_flatten(obs_dict["left_team"]))
                state.extend(do_flatten(obs_dict["left_team_direction"]))
                state.extend(do_flatten(obs_dict["right_team"]))
                state.extend(do_flatten(obs_dict["right_team_direction"]))

                state.extend(do_flatten(obs_dict["ball"]))
                state.extend(do_flatten(obs_dict["ball_direction"]))
                if obs_dict["ball_owned_team"] == -1:
                    state.extend([1, 0, 0])
                if obs_dict["ball_owned_team"] == 0:
                    state.extend([0, 1, 0])
                if obs_dict["ball_owned_team"] == 1:
                    state.extend([0, 0, 1])

                state.extend(do_flatten(np.eye(7)[obs_dict["game_mode"]]))

                for p_i in player_ids:
                    state.extend(do_flatten(np.eye(self.num_left_agents)[p_i]))
            else:
                state = []
                for i, name in enumerate(
                    [
                        "left_team",
                        "left_team_direction",
                        "right_team",
                        "right_team_direction",
                    ]
                ):
                    state.extend(do_flatten(obs_dict[name]))
                    # If there were less than 11vs11 players we backfill missing values
                    # with -1.
                    if len(state) < (i + 1) * 22:
                        state.extend([-1] * ((i + 1) * 22 - len(state)))

                state.extend(obs_dict["ball"])
                state.extend(obs_dict["ball_direction"])
                if obs_dict["ball_owned_team"] == -1:
                    state.extend([1, 0, 0])
                if obs_dict["ball_owned_team"] == 0:
                    state.extend([0, 1, 0])
                if obs_dict["ball_owned_team"] == 1:
                    state.extend([0, 0, 1])

                game_mode = [0] * 7
                game_mode[obs_dict["game_mode"]] = 1
                state.extend(game_mode)

                for p_i in player_ids:
                    active = [0] * 11
                    active[p_i] = 1
                    state.extend(active)

            if self.use_agent_policy_id:
                assert agent_policy_id is not None
                state.extend(do_flatten(np.ones(self.num_controlled_agents, dtype=np.float32) * agent_policy_id[a_i]))

            state_list.append(np.array(state, dtype=np.float32))

        return state_list

    def encode_each(self, obs: np.ndarray, obs_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
        if "simple" in self.representation:
            if "custom" in self.representation:
                # 22 - (x,y) coordinates of left team players
                # 22 - (x,y) direction of left team players
                # 22 - (x,y) coordinates of right team players
                # 22 - (x,y) direction of right team players
                # 3 - (x,y,z) - ball position
                # 3 - ball direction
                # 3 - one hot encoding of ball ownership (noone, left, right)
                # 11 - one hot encoding of which player is active  (agent id)
                # 7 - one hot encoding of `game_mode`
                # dim: 115
                # ==>

                # 2 - (x,y) coordinate of current player
                # 2 - (x,y) direction of current player
                # 2 - (is_sprinting, is_dribbling) agent status
                # (n1-1) * 2 - (Δx,Δy) relative coordinates of other left team players
                # n2 * 2 - (Δx,Δy) relative coordinates of right team players
                # 2 - (Δx,Δy) relative coordinate of current player to the ball

                # (n1-1) * 2 - (x,y) coordinates of other left team players
                # (n1-1) * 2 - (x,y) direction of other left team players
                # n2 * 2 - (x,y) coordinates of right team players
                # n2 * 2 - (x,y) direction of right team players

                # 3 - (x,y,z) - ball position
                # 3 - ball direction
                # 3 - one hot encoding of ball ownership (noone, left, right)

                # 7 - one hot encoding of `game_mode`
                # n1 - one hot encoding of which player is active  (agent id)
                # dim: 4 * 2 + (n1-1) * 2 * 3 + n2 * 2 * 3 + 3 + 3 + 3 + n1 + 7 = 7 * n1 + 6 * n2 + 18

                p_i = obs_dict["active"]
                feat = []
                feat.extend(do_flatten(obs_dict["left_team"][p_i]))
                feat.extend(do_flatten(obs_dict["left_team_direction"][p_i]))
                feat.extend([obs_dict["sticky_actions"][8], obs_dict["sticky_actions"][9]])
                feat.extend(do_flatten(np.delete(obs_dict["left_team"], p_i, axis=0) - obs_dict["left_team"][p_i]))
                feat.extend(do_flatten(obs_dict["right_team"] - obs_dict["left_team"][p_i]))
                feat.extend(do_flatten(obs_dict["ball"][:2] - obs_dict["left_team"][p_i]))

                feat.extend(do_flatten(np.delete(obs_dict["left_team"], p_i, axis=0)))
                feat.extend(do_flatten(np.delete(obs_dict["left_team_direction"], p_i, axis=0)))
                feat.extend(do_flatten(obs_dict["right_team"]))
                feat.extend(do_flatten(obs_dict["right_team_direction"]))

                feat.extend(do_flatten(obs_dict["ball"]))
                feat.extend(do_flatten(obs_dict["ball_direction"]))
                if obs_dict["ball_owned_team"] == -1:
                    feat.extend([1, 0, 0])
                if obs_dict["ball_owned_team"] == 0:
                    feat.extend([0, 1, 0])
                if obs_dict["ball_owned_team"] == 1:
                    feat.extend([0, 0, 1])

                feat.extend(do_flatten(np.eye(7)[obs_dict["game_mode"]]))
                feat.extend(do_flatten(np.eye(self.num_left_agents)[p_i]))
                feat = np.array(feat, dtype=np.float32)
            else:
                feat = obs
        else:
            raise NotImplementedError("Only support `simple115v2`-based")

        avail = self.get_available_actions(
            obs_dict,
        )

        return feat, avail

    def get_available_actions(self, obs_dict: Dict) -> np.ndarray:
        assert self.action_n == 19 or self.action_n == 20  # we don't support full action set
        avail = [1] * self.action_n

        a_i = obs_dict["active"]
        ball_distance = np.linalg.norm(obs_dict["left_team"][a_i] - obs_dict["ball"][:2])

        (
            NO_OP,
            LEFT,
            TOP_LEFT,
            TOP,
            TOP_RIGHT,
            RIGHT,
            BOTTOM_RIGHT,
            BOTTOM,
            BOTTOM_LEFT,
            LONG_PASS,
            HIGH_PASS,
            SHORT_PASS,
            SHOT,
            SPRINT,
            RELEASE_MOVE,
            RELEASE_SPRINT,
            SLIDE,
            DRIBBLE,
            RELEASE_DRIBBLE,
        ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

        if self.action_n == 20:
            pass

        # if obs_dict["ball_owned_team"] == 1:  # opponents owning ball
        #     (
        #         avail[LONG_PASS],
        #         avail[HIGH_PASS],
        #         avail[SHORT_PASS],
        #         avail[SHOT],
        #         avail[DRIBBLE],
        #     ) = (0, 0, 0, 0, 0)
        #     if ball_distance > 0.03:
        #         avail[SLIDE] = 0
        # elif (
        #     obs_dict["ball_owned_team"] == -1
        #     and ball_distance > 0.03
        #     and obs_dict["game_mode"] == 0
        # ):  # Ground ball and far from me
        #     (
        #         avail[LONG_PASS],
        #         avail[HIGH_PASS],
        #         avail[SHORT_PASS],
        #         avail[SHOT],
        #         avail[DRIBBLE],
        #         avail[SLIDE],
        #     ) = (0, 0, 0, 0, 0, 0)
        # elif obs_dict["ball_owned_team"] == 0:  # my team owning ball
        #     avail[SLIDE] = 0
        #     if ball_distance > 0.03:
        #         (
        #             avail[LONG_PASS],
        #             avail[HIGH_PASS],
        #             avail[SHORT_PASS],
        #             avail[SHOT],
        #             avail[DRIBBLE],
        #         ) = (0, 0, 0, 0, 0)
        if obs_dict["ball_owned_team"] == 0:  # my team owning ball
            avail[SLIDE] = 0

        # Dealing with sticky actions
        sticky_actions = obs_dict["sticky_actions"]
        if sticky_actions[8] == 0:  # sprinting
            avail[RELEASE_SPRINT] = 0

        if sticky_actions[9] == 1:  # dribbling
            avail[SLIDE] = 0
        else:
            avail[RELEASE_DRIBBLE] = 0

        if np.sum(sticky_actions[:8]) == 0:
            avail[RELEASE_MOVE] = 0

        ball_x, ball_y, _ = obs_dict["ball"]

        # if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
        #     # if too far, no shot
        #     avail[SHOT] = 0
        # elif (0.64 <= ball_x and ball_x <= 1.0) and (-0.27 <= ball_y and ball_y <= 0.27):
        #     # if in restricted area, no high pass
        #     avail[HIGH_PASS], avail[LONG_PASS] = 0, 0

        if obs_dict["game_mode"] == 2 and ball_x < -0.7:  # Our GoalKick
            avail = [1] + [0] * (self.action_n - 1)
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs_dict["game_mode"] == 4 and ball_x > 0.9:  # Our CornerKick
            avail = [1] + [0] * (self.action_n - 1)
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs_dict["game_mode"] == 6 and ball_x > 0.6:  # Our PenaltyKick
            avail = [1] + [0] * (self.action_n - 1)
            avail[SHOT] = 1
            return np.array(avail)

        return np.array(avail)
