from typing import Dict


class Rewarder:
    def __init__(self, reward_config: Dict = {}) -> None:
        self.reward_config = reward_config

    def calc_reward(self, rew: float, info: Dict, prev_obs_dict: Dict, obs_dict: Dict, a_i: int):
        """
        rew: original reward
        info: info["score_reward"] records the score (0, 1, -1)
        """
        reward = 0
        checkpoints_reward = rew - info["score_reward"]  # checkpoints reward

        sub_reward_to_func = {
            "win_reward": get_win_reward,
        }

        if len(self.reward_config[a_i]) > 0:
            for sub_reward, weight in self.reward_config[a_i].items():
                if sub_reward == "checkpoints":
                    reward += weight * checkpoints_reward
                elif sub_reward in info:
                    # event-based reward in info["stats"]
                    reward += weight * info[sub_reward][a_i]
                else:
                    reward += weight * sub_reward_to_func[sub_reward](prev_obs_dict, obs_dict)
        else:
            reward = rew
        return reward


def get_win_reward(prev_obs_dict: Dict, obs_dict: Dict):
    win_reward = 0.0
    if obs_dict["steps_left"] == 0:
        [my_score, opponent_score] = obs_dict["score"]
        if my_score > opponent_score:
            win_reward = my_score - opponent_score
    return win_reward
