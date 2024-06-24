from typing import Dict, List, Tuple, Union

import numpy as np

from .grf_env import SHAPED_INFOS


class StatsObserver:
    def __init__(self, num_agents: int, player_ids: List[int]):
        self.num_agents = num_agents
        self.player_ids = player_ids
        self.player_ids_2_agent_ids = dict(zip(self.player_ids, range(self.num_agents)))

        self.ball_tracer: List[Tuple[int, int, int, int]] = []  # team, player, game_mode, my_score
        self.stats = {info: [0 for _ in range(self.num_agents)] for info in SHAPED_INFOS}

    def observe(
        self,
        action: Union[np.ndarray, List[int]],
        obs_dict_list: List[Dict],
        next_obs_dict_list: List[Dict],
    ) -> Dict[str, List[int]]:
        one_step_stats = {info: [0 for _ in range(self.num_agents)] for info in SHAPED_INFOS}

        LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT, SLIDE = 9, 10, 11, 12, 16
        KickOff, GoalKick, Corner, ThrowIn = 1, 2, 4, 5

        BALL_NO_OWNER, BALL_OWNED_BY_US, BALL_OWNED_BY_OPP = -1, 0, 1

        ball_own_team = obs_dict_list[0]["ball_owned_team"]
        ball_own_player = obs_dict_list[0]["ball_owned_player"]
        game_mode = obs_dict_list[0]["game_mode"]
        my_score, opp_score = obs_dict_list[0]["score"]

        next_ball_own_team = next_obs_dict_list[0]["ball_owned_team"]
        next_ball_own_player = next_obs_dict_list[0]["ball_owned_player"]
        next_obs_dict_list[0]["game_mode"]
        next_my_score, next_opp_score = next_obs_dict_list[0]["score"]

        if ball_own_team != BALL_NO_OWNER:
            self.ball_tracer.append((ball_own_team, ball_own_player, game_mode, my_score))

            if ball_own_team == BALL_OWNED_BY_US:
                if ball_own_player in self.player_ids:
                    ball_own_agent_id = self.player_ids_2_agent_ids[ball_own_player]
                    if action[ball_own_agent_id] in [
                        LONG_PASS,
                        HIGH_PASS,
                        SHORT_PASS,
                    ]:
                        one_step_stats["pass"][ball_own_agent_id] += 1
                    elif action[ball_own_agent_id] in [
                        SHOT,
                    ]:
                        one_step_stats["shot"][ball_own_agent_id] += 1
                    elif action[ball_own_agent_id] in [
                        SLIDE,
                    ]:
                        one_step_stats["slide"][ball_own_agent_id] += 1
                    one_step_stats["possession"][ball_own_agent_id] += 1

        if next_ball_own_team == BALL_OWNED_BY_US:
            if len(self.ball_tracer) >= 1:
                (
                    last_ball_own_team,
                    last_ball_own_player,
                    last_game_mode,
                    last_my_score,
                ) = self.ball_tracer[-1]

                # last ball is owned by teammate
                if (
                    last_ball_own_team == BALL_OWNED_BY_US
                    and last_ball_own_player != next_ball_own_player
                    and last_ball_own_player in self.player_ids
                ):
                    if next_ball_own_player in self.player_ids:
                        ball_own_agent_id = self.player_ids_2_agent_ids[next_ball_own_player]
                        one_step_stats["catch"][ball_own_agent_id] += 1
                        last_ball_own_agent_id = self.player_ids_2_agent_ids[last_ball_own_player]
                        one_step_stats["actual_pass"][last_ball_own_agent_id] += 1  # only count in-team stats

        # elif next_ball_own_team == BALL_NO_OWNER:  # ball may be not owned when goaled
        if len(self.ball_tracer) >= 1:
            (
                last_ball_own_team,
                last_ball_own_player,
                last_game_mode,
                last_my_score,
            ) = self.ball_tracer[-1]

            if (
                next_my_score > last_my_score
                and last_ball_own_team == BALL_OWNED_BY_US
                and last_ball_own_player in self.player_ids
            ):
                last_ball_own_agent_id = self.player_ids_2_agent_ids[last_ball_own_player]
                one_step_stats["score"][last_ball_own_agent_id] += 1

                if len(self.ball_tracer) > 1:
                    shot_player = last_ball_own_player
                    step_i = len(self.ball_tracer) - 2
                    while (
                        step_i >= 0
                        and self.ball_tracer[step_i][0] == BALL_OWNED_BY_US
                        and self.ball_tracer[step_i][1] == shot_player
                    ):
                        step_i -= 1
                    if (
                        step_i >= 0
                        and self.ball_tracer[step_i][0] == BALL_OWNED_BY_US
                        and self.ball_tracer[step_i][1] in self.player_ids
                    ):
                        one_step_stats["assist"][self.player_ids_2_agent_ids[self.ball_tracer[step_i][1]]] += 1

        for k, v in one_step_stats.items():
            self.stats[k] = [x + y for x, y in zip(self.stats[k], v)]

        return one_step_stats

    def get_stats(self) -> Dict[str, List]:
        return self.stats

    def reset(self):
        self.stats = {info: [0 for _ in range(self.num_agents)] for info in SHAPED_INFOS}
