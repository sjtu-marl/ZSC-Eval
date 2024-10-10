import os
import pickle
import random
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import tqdm
import yaml
from loguru import logger

from zsceval.algorithms.population.policy_pool import add_path_prefix
from zsceval.algorithms.population.utils import EvalPolicy
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action, Direction
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    SoupState,
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
)
from zsceval.runner.shared.base_runner import make_trainer_policy_cls


class AgentPool:
    def get_agent(self) -> Callable[[Dict, int], int]:
        pass

    def _get_action(self, policy: EvalPolicy, state: np.ndarray, available_actions: np.ndarray = None) -> int:
        pass

    def _process_state(self, state: Dict, featurize_type: str, pos: int) -> Tuple[np.ndarray, np.ndarray]:
        pass


class ZSCEvalAgentPool(AgentPool):
    """
    Policy used for a layout
    """

    POLICY_POOL_PATH = os.environ["POLICY_POOL"]

    def __init__(self, population_yaml_path: str, layout_name: str, deterministic: bool = True, epsilon: float = 0.5):
        population_config = yaml.load(open(population_yaml_path, "r", encoding="utf-8"), yaml.Loader)
        self.n_agents = len(population_config)
        self.policy_pool: Dict[str, List[Tuple]] = defaultdict(list)
        """ 
        MEP:
            - 0
                - policy_name
                - policy_args
                - policy
                - featurize_type: str
        """
        self.layout_name = layout_name
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.deterministic = deterministic
        self.epsilon = epsilon

        for policy_name in tqdm.tqdm(population_config, desc="Loading models..."):
            try:
                policy_config_path = os.path.join(
                    self.POLICY_POOL_PATH,
                    population_config[policy_name]["policy_config_path"],
                )
                policy_config = list(
                    pickle.load(open(policy_config_path, "rb"))
                )  # args, obs_shape, share_obs_shape, act_shape
                policy_args = policy_config[0]
                _, policy_cls = make_trainer_policy_cls(policy_args.algorithm_name)
                policy = policy_cls(*policy_config, device=torch.device("cpu"))
                if population_config[policy_name].get("model_path", None):
                    model_path = add_path_prefix(self.POLICY_POOL_PATH, population_config[policy_name]["model_path"])
                    policy.load_checkpoint(model_path)
                featurize_type = population_config[policy_name]["featurize_type"]
                self.policy_pool[population_config[policy_name]["algo"]].append(
                    (policy_name, policy_args, policy, featurize_type)
                )
            except Exception as e:
                logger.error(f"Error loading policy {policy_name}: {e}")
                raise e

    @property
    def agent_names(self) -> Dict[str, List[str]]:
        policy_name_pool = {
            algo: [policy_tuple[0] for policy_tuple in self.policy_pool[algo]] for algo in self.policy_pool
        }
        return policy_name_pool

    def get_agent(self, algo: str) -> Callable[[Dict, int], int]:
        policy_tuple = random.choice(self.policy_pool[algo])

        policy = EvalPolicy(policy_tuple[1], policy_tuple[2])
        policy.reset(1, 1)
        policy.register_control_agent(0, 0)

        def _agent_call(state: Dict, pos: int) -> int:
            state, available_actions = self._process_state(state, policy_tuple[3], pos)
            return self._get_action(policy, state, available_actions)

        return _agent_call

    def _process_state(self, state: Dict, featurize_type: str, pos: int) -> Tuple[np.ndarray, np.ndarray]:

        def object_from_dict(object_dict: Dict):
            return ObjectState(**object_dict)
        
        def soup_from_dict(object_dict: Dict):
            ingredients = []
            for i in object_dict["ingredients"]:
                ingredients.append(object_from_dict(i))
            return SoupState(position=object_dict["position"], ingredients=ingredients,
                             cooking_tick=object_dict["cooking_tick"], cooking_time=object_dict["cook_time"])

        def player_from_dict(player_dict: Dict):
            
            held_obj = player_dict.get("held_object")
            if held_obj is not None:
                if held_obj["name"] == "soup": 
                    player_dict["held_object"] = soup_from_dict(held_obj)
                else:
                    player_dict["held_object"] = object_from_dict(held_obj)
                    
            return PlayerState(**player_dict)

        def state_from_dict(state_dict: Dict):
            state_dict["players"] = [player_from_dict(p) for p in state_dict["players"]]
            object_list = []
            for _, o in state_dict["objects"].items():
                if o["name"] == "soup":
                    object_list.append(soup_from_dict(o))
                else:
                    object_list.append(object_from_dict(o))
            state_dict["objects"] = {ob.position: ob for ob in object_list}
            return OvercookedState(**state_dict)

        if featurize_type == "ppo":
            state_obj = state_from_dict(state.copy())
            return self.mdp.lossless_state_encoding(state_obj)[pos] * 255, self._get_available_actions(state_obj)[pos]

    def _get_available_actions(self, state: OvercookedState) -> np.ndarray:
        num_agents = len(state.players)
        available_actions = np.ones((num_agents, len(Action.ALL_ACTIONS)), dtype=np.uint8)
        interact_index = Action.ACTION_TO_INDEX["interact"]
        for agent_idx in range(num_agents):
            player = state.players[agent_idx]
            pos = player.position
            o = player.orientation
            for move_i, move in enumerate(Direction.ALL_DIRECTIONS):
                new_pos = Action.move_in_direction(pos, move)
                if new_pos not in self.mdp.get_valid_player_positions() and o == move:
                    available_actions[agent_idx, move_i] = 0

            i_pos = Action.move_in_direction(pos, o)
            terrain_type = self.mdp.get_terrain_type_at_pos(i_pos)

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
                    and (player.has_object() and player.get_object().name not in ["dish", "onion", "tomato"])
                )
                or (terrain_type == "S" and (not player.has_object() or player.get_object().name not in ["soup"]))
            ):
                available_actions[agent_idx, interact_index] = 0
        return available_actions

    @torch.no_grad()
    def _get_action(self, policy: EvalPolicy, state: np.ndarray, available_actions: np.ndarray = None) -> int:
        policy.prep_rollout()
        epsilon = random.random()
        # print(available_actions)
        if not self.deterministic or epsilon < self.epsilon:
            return policy.step(
                np.array([state]),
                [(0, 0)],
                deterministic=False,
                available_actions=np.array([available_actions]),
            )[0]
        else:
            return policy.step(
                np.array([state]),
                [(0, 0)],
                deterministic=True,
                available_actions=np.array([available_actions]),
            )[0]
