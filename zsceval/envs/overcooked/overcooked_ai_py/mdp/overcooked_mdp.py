import copy
import itertools
import random
from collections import defaultdict
from functools import reduce
from typing import List, Tuple

import numpy as np

from zsceval.envs.overcooked.overcooked_ai_py.data.layouts import read_layout_dict
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from zsceval.envs.overcooked.overcooked_ai_py.utils import load_from_json, pos_distance

SHAPED_INFOS = [
    "put_onion_on_X",
    "put_dish_on_X",
    "put_soup_on_X",
    "pickup_onion_from_X",
    "pickup_onion_from_O",
    "pickup_dish_from_X",
    "pickup_dish_from_D",
    "pickup_soup_from_X",
    "USEFUL_DISH_PICKUP",  # counted when #taken_dishes < #cooking_pots + #partially_full_pots and no dishes on the counter
    "SOUP_PICKUP",  # counted when soup in the pot is picked up (not a soup placed on the table)
    "PLACEMENT_IN_POT",  # counted when some ingredient is put into pot
    "delivery",
    "STAY",
    "MOVEMENT",
    "IDLE_MOVEMENT",
    "IDLE_INTERACT_X",
    "IDLE_INTERACT_EMPTY",
]


class ObjectState:
    """
    State of an object in OvercookedGridworld.
    """

    SOUP_TYPES = ["onion", "tomato"]

    def __init__(self, name, position, state=None):
        """
        name (str): The name of the object
        position (int, int): Tuple for the current location of the object.
        state (tuple or None):
            Extra information about the object. Is None for all objects
            except soups, for which `state` is a tuple:
            (soup_type, num_items, cook_time)
            where cook_time is how long the soup has been cooking for.
        """
        self.name = name
        self.position = tuple(position)
        if name == "soup":
            assert len(state) == 3
        self.state = None if state is None else tuple(state)

    def is_valid(self):
        if self.name in ["onion", "tomato", "dish"]:
            return self.state is None
        elif self.name == "soup":
            soup_type, num_items, cook_time = self.state
            valid_soup_type = soup_type in self.SOUP_TYPES
            valid_item_num = 1 <= num_items <= 3
            valid_cook_time = 0 <= cook_time
            return valid_soup_type and valid_item_num and valid_cook_time
        # Unrecognized object
        return False

    def deepcopy(self):
        return ObjectState(self.name, self.position, self.state)

    def __eq__(self, other):
        return (
            isinstance(other, ObjectState)
            and self.name == other.name
            and self.position == other.position
            and self.state == other.state
        )

    def __hash__(self):
        return hash((self.name, self.position, self.state))

    def __repr__(self):
        if self.state is None:
            return f"{self.name}@{self.position}"
        return f"{self.name}@{self.position} with state {str(self.state)}"

    def to_dict(self):
        return {"name": self.name, "position": self.position, "state": self.state}

    @staticmethod
    def from_dict(obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        return ObjectState(**obj_dict)


class PlayerState:
    """
    State of a player in OvercookedGridworld.

    position: (x, y) tuple representing the player's location.
    orientation: Direction.NORTH/SOUTH/EAST/WEST representing orientation.
    held_object: ObjectState representing the object held by the player, or
                 None if there is no such object.
    """

    def __init__(self, position, orientation, held_object=None):
        self.position = tuple(position)
        self.orientation = tuple(orientation)
        self.held_object = held_object

        assert self.orientation in Direction.ALL_DIRECTIONS
        if self.held_object is not None:
            assert isinstance(self.held_object, ObjectState)
            assert self.held_object.position == self.position

    @property
    def pos_and_or(self):
        return (self.position, self.orientation)

    def has_object(self):
        return self.held_object is not None

    def get_object(self):
        assert self.has_object()
        return self.held_object

    def set_object(self, obj):
        assert not self.has_object()
        obj.position = self.position
        self.held_object = obj

    def remove_object(self):
        assert self.has_object()
        obj = self.held_object
        self.held_object = None
        return obj

    def update_pos_and_or(self, new_position, new_orientation):
        self.position = new_position
        self.orientation = new_orientation
        if self.has_object():
            self.get_object().position = new_position

    def deepcopy(self):
        new_obj = None if self.held_object is None else self.held_object.deepcopy()
        return PlayerState(self.position, self.orientation, new_obj)

    def __eq__(self, other):
        return (
            isinstance(other, PlayerState)
            and self.position == other.position
            and self.orientation == other.orientation
            and self.held_object == other.held_object
        )

    def __hash__(self):
        return hash((self.position, self.orientation, self.held_object))

    def __repr__(self):
        return f"{self.position} facing {self.orientation} holding {str(self.held_object)}"

    def to_dict(self):
        return {
            "position": self.position,
            "orientation": self.orientation,
            "held_object": self.held_object.to_dict() if self.held_object is not None else None,
        }

    @staticmethod
    def from_dict(player_dict):
        player_dict = copy.deepcopy(player_dict)
        held_obj = player_dict["held_object"]
        if held_obj is not None:
            player_dict["held_object"] = ObjectState.from_dict(held_obj)
        return PlayerState(**player_dict)


class OvercookedState:
    """A state in OvercookedGridworld."""

    def __init__(self, players, objects, order_list, timestep=0):
        """
        players: List of PlayerStates (order corresponds to player indices).
        objects: Dictionary mapping positions (x, y) to ObjectStates.
                 NOTE: Does NOT include objects held by players (they are in
                 the PlayerState objects).
        order_list: Current orders to be delivered

        NOTE: Does not contain time left, which is handled from the environment side.
        """
        self.timestep = timestep
        for pos, obj in objects.items():
            assert obj.position == pos
        self.players = tuple(players)
        self.objects = objects
        if order_list is not None:
            assert all([o in OvercookedGridworld.ORDER_TYPES for o in order_list])
        self.order_list = order_list

    @property
    def player_positions(self):
        return tuple([player.position for player in self.players])

    @property
    def player_orientations(self):
        return tuple([player.orientation for player in self.players])

    @property
    def players_pos_and_or(self):
        """Returns a ((pos1, or1), (pos2, or2)) tuple"""
        return tuple(zip(*[self.player_positions, self.player_orientations]))

    @property
    def unowned_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects in the environment, NOT including
        ones held by players.
        """
        objects_by_type = defaultdict(list)
        for pos, obj in self.objects.items():
            objects_by_type[obj.name].append(obj)
        return objects_by_type

    @property
    def player_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects held by players.
        """
        player_objects = defaultdict(list)
        for player in self.players:
            if player.has_object():
                player_obj = player.get_object()
                player_objects[player_obj.name].append(player_obj)
        return player_objects

    @property
    def all_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects in the environment, including
        ones held by players.
        """
        all_objs_by_type = self.unowned_objects_by_type.copy()
        all_objs_by_type.update(self.player_objects_by_type)
        return all_objs_by_type

    @property
    def all_objects_list(self):
        all_objects_lists = list(self.all_objects_by_type.values()) + [[], []]
        return reduce(lambda x, y: x + y, all_objects_lists)

    @property
    def curr_order(self):
        return "any" if self.order_list is None else self.order_list[0]

    @property
    def next_order(self):
        return "any" if self.order_list is None else self.order_list[1]

    @property
    def num_orders_remaining(self):
        return np.Inf if self.order_list is None else len(self.order_list)

    def has_object(self, pos):
        return pos in self.objects

    def get_object(self, pos):
        assert self.has_object(pos)
        return self.objects[pos]

    def add_object(self, obj, pos=None):
        if pos is None:
            pos = obj.position

        assert not self.has_object(pos)
        obj.position = pos
        self.objects[pos] = obj

    def remove_object(self, pos):
        assert self.has_object(pos)
        obj = self.objects[pos]
        del self.objects[pos]
        return obj

    @staticmethod
    def from_players_pos_and_or(players_pos_and_or, order_list):
        """
        Make a dummy OvercookedState with no objects based on the passed in player
        positions and orientations and order list
        """
        return OvercookedState(
            [PlayerState(*player_pos_and_or) for player_pos_and_or in players_pos_and_or],
            objects={},
            order_list=order_list,
        )

    @staticmethod
    def from_player_positions(player_positions, order_list):
        """
        Make a dummy OvercookedState with no objects and with players facing
        North based on the passed in player positions and order list
        """
        dummy_pos_and_or = [(pos, Direction.NORTH) for pos in player_positions]
        return OvercookedState.from_players_pos_and_or(dummy_pos_and_or, order_list)

    def deepcopy(self):
        return OvercookedState(
            [player.deepcopy() for player in self.players],
            {pos: obj.deepcopy() for pos, obj in self.objects.items()},
            None if self.order_list is None else list(self.order_list),
            timestep=self.timestep,
        )

    def __eq__(self, other):
        order_list_equal = type(self.order_list) == type(other.order_list) and (
            (self.order_list is None and other.order_list is None)
            or (type(self.order_list) is list and np.array_equal(self.order_list, other.order_list))
        )

        return (
            isinstance(other, OvercookedState)
            and self.players == other.players
            and set(self.objects.items()) == set(other.objects.items())
            and order_list_equal
            and self.timestep == other.timestep
        )

    def __hash__(self):
        return hash((self.players, tuple(self.objects.values()), tuple(self.order_list)))

    def __str__(self):
        return "Players: {}, Objects: {}, Order list: {} Timestep: {}".format(
            str(self.players),
            str(list(self.objects.values())),
            str(self.order_list),
            str(self.timestep),
        )

    def to_dict(self):
        return {
            "players": [p.to_dict() for p in self.players],
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "order_list": self.order_list,
            "timestep": self.timestep,
        }

    @staticmethod
    def from_dict(state_dict):
        state_dict = copy.deepcopy(state_dict)
        state_dict["players"] = [PlayerState.from_dict(p) for p in state_dict["players"]]
        object_list = [ObjectState.from_dict(o) for o in state_dict["objects"]]
        state_dict["objects"] = {ob.position: ob for ob in object_list}
        return OvercookedState(**state_dict)

    @staticmethod
    def from_json(filename):
        return load_from_json(filename)


NO_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 0,
    "DISH_PICKUP_REWARD": 0,
    "SOUP_PICKUP_REWARD": 0,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}

BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}


class OvercookedGridworld:
    """An MDP grid world based off of the Overcooked game."""

    ORDER_TYPES = ObjectState.SOUP_TYPES + ["any"]

    def __init__(
        self,
        terrain,
        start_player_positions,
        start_order_list=None,
        cook_time=20,
        num_items_for_soup=3,
        delivery_reward=20,
        rew_shaping_params=None,
        layout_name="unnamed_layout",
    ):
        """
        terrain: a matrix of strings that encode the MDP layout
        layout_name: string identifier of the layout
        start_player_positions: tuple of positions for both players' starting positions
        start_order_list: either a tuple of orders or None if there is not specific list
        cook_time: amount of timesteps required for a soup to cook
        delivery_reward: amount of reward given per delivery
        rew_shaping_params: reward given for completion of specific subgoals
        """
        self.height = len(terrain)
        self.width = len(terrain[0])
        self.shape = (self.width, self.height)
        self.terrain_mtx = terrain
        self.terrain_pos_dict = self._get_terrain_type_pos_dict()
        self.start_player_positions = start_player_positions
        self.num_players = len(start_player_positions)
        self.start_order_list = start_order_list
        self.soup_cooking_time = cook_time
        self.num_items_for_soup = num_items_for_soup
        self.delivery_reward = delivery_reward
        self.reward_shaping_params = NO_REW_SHAPING_PARAMS if rew_shaping_params is None else rew_shaping_params
        self.layout_name = layout_name

    def __eq__(self, other):
        return (
            np.array_equal(self.terrain_mtx, other.terrain_mtx)
            and self.start_player_positions == other.start_player_positions
            and self.start_order_list == other.start_order_list
            and self.soup_cooking_time == other.soup_cooking_time
            and self.num_items_for_soup == other.num_items_for_soup
            and self.delivery_reward == other.delivery_reward
            and self.reward_shaping_params == other.reward_shaping_params
            and self.layout_name == other.layout_name
        )

    def copy(self):
        return OvercookedGridworld(
            terrain=self.terrain_mtx.copy(),
            start_player_positions=self.start_player_positions,
            start_order_list=None if self.start_order_list is None else list(self.start_order_list),
            cook_time=self.soup_cooking_time,
            num_items_for_soup=self.num_items_for_soup,
            delivery_reward=self.delivery_reward,
            rew_shaping_params=copy.deepcopy(self.reward_shaping_params),
            layout_name=self.layout_name,
        )

    @property
    def mdp_params(self):
        return {
            "layout_name": self.layout_name,
            "terrain": self.terrain_mtx,
            "start_player_positions": self.start_player_positions,
            "start_order_list": self.start_order_list,
            "cook_time": self.soup_cooking_time,
            "num_items_for_soup": self.num_items_for_soup,
            "delivery_reward": self.delivery_reward,
            "rew_shaping_params": copy.deepcopy(self.reward_shaping_params),
        }

    @staticmethod
    def from_layout_name(layout_name, **params_to_overwrite):
        """
        Generates a OvercookedGridworld instance from a layout file.

        One can overwrite the default mdp configuration using partial_mdp_config.
        """
        params_to_overwrite = params_to_overwrite.copy()
        base_layout_params = read_layout_dict(layout_name)

        grid = base_layout_params["grid"]
        del base_layout_params["grid"]
        base_layout_params["layout_name"] = layout_name

        # Clean grid
        grid = [layout_row.strip() for layout_row in grid.split("\n")]
        return OvercookedGridworld.from_grid(grid, base_layout_params, params_to_overwrite)

    @staticmethod
    def from_grid(layout_grid, base_layout_params={}, params_to_overwrite={}, debug=False):
        """
        Returns instance of OvercookedGridworld with terrain and starting
        positions derived from layout_grid.
        One can override default configuration parameters of the mdp in
        partial_mdp_config.
        """
        mdp_config = base_layout_params.copy()

        layout_grid = [[c for c in row] for row in layout_grid]
        OvercookedGridworld._assert_valid_grid(layout_grid)

        player_positions = [None] * 9
        for y, row in enumerate(layout_grid):
            for x, c in enumerate(row):
                if c in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    layout_grid[y][x] = " "

                    # -1 is to account for fact that player indexing starts from 1 rather than 0
                    assert player_positions[int(c) - 1] is None, "Duplicate player in grid"
                    player_positions[int(c) - 1] = (x, y)

        num_players = len([x for x in player_positions if x is not None])
        player_positions = player_positions[:num_players]

        # After removing player positions from grid we have a terrain mtx
        mdp_config["terrain"] = layout_grid
        mdp_config["start_player_positions"] = player_positions

        for k, v in params_to_overwrite.items():
            curr_val = mdp_config[k]
            if debug:
                print(f"Overwriting mdp layout standard config value {k}:{curr_val} -> {v}")
            mdp_config[k] = v

        return OvercookedGridworld(**mdp_config)

    def get_actions(self, state):
        """
        Returns the list of lists of valid actions for 'state'.

        The ith element of the list is the list of valid actions that player i
        can take.
        """
        self._check_valid_state(state)
        return [self._get_player_actions(state, i) for i in range(len(state.players))]

    def _get_player_actions(self, state, player_num):
        """All actions are allowed to all players in all states."""
        return Action.ALL_ACTIONS

    def _check_action(self, state, joint_action):
        for p_action, p_legal_actions in zip(joint_action, self.get_actions(state)):
            if p_action not in p_legal_actions:
                raise ValueError("Invalid action")

    def get_standard_start_state(self):
        start_state = OvercookedState.from_player_positions(
            self.start_player_positions, order_list=self.start_order_list
        )
        return start_state

    def get_random_start_state(self, random_terrain_state: bool = False, random_player_pos: bool = False):
        assert not (random_terrain_state or random_player_pos)
        state = self.get_standard_start_state()
        return state

    def _get_random_player_pos(self, valid_pos_lst: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        def neighbors(pos, pos_lst):
            deltas = Direction.ALL_DIRECTIONS
            neighs = [Action.move_in_direction(pos, d) for d in deltas]
            return [p for p in neighs if p in pos_lst]

        def dfs(pos, visited, pos_lst):
            stack = [pos]
            component = []
            while stack:
                vertex = stack.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    component.append(vertex)
                    stack.extend(neighbors(vertex, pos_lst))
            return component

        def connected_components(pos_lst):
            visited = set()
            components = []
            for pos in pos_lst:
                if pos not in visited:
                    component = dfs(pos, visited, pos_lst)
                    components.append(component)
            return components

        conns = connected_components(valid_pos_lst)
        assert len(conns) in [1, 2]
        random_pos_lst = []
        for p_i in range(self.num_players):
            for conn in conns:
                if self.start_player_positions[p_i] in conn:
                    candidates = list(set(conn).difference(random_pos_lst))
                    random_pos_lst.append(random.choice(candidates))
        assert len(random_pos_lst) == self.num_players
        return random_pos_lst

    def get_random_start_state_fn(self, random_start_pos=False, rnd_obj_prob_thresh=0.0):
        def start_state_fn():
            if random_start_pos:
                valid_positions = self.get_valid_joint_player_positions()
                start_pos = valid_positions[np.random.choice(len(valid_positions))]
            else:
                start_pos = self.start_player_positions

            start_state = OvercookedState.from_player_positions(start_pos, order_list=self.start_order_list)

            if rnd_obj_prob_thresh == 0:
                return start_state

            # Arbitrary hard-coding for randomization of objects
            # For each pot, add a random amount of onions with prob rnd_obj_prob_thresh
            pots = self.get_pot_states(start_state)["empty"]
            for pot_loc in pots:
                p = np.random.rand()
                if p < rnd_obj_prob_thresh:
                    n = int(np.random.randint(low=1, high=4))
                    start_state.objects[pot_loc] = ObjectState("soup", pot_loc, ("onion", n, 0))

            # For each player, add a random object with prob rnd_obj_prob_thresh
            for player in start_state.players:
                p = np.random.rand()
                if p < rnd_obj_prob_thresh:
                    # Different objects have different probabilities
                    obj = np.random.choice(["dish", "onion", "soup"], p=[0.2, 0.6, 0.2])
                    if obj == "soup":
                        player.set_object(
                            ObjectState(
                                obj,
                                player.position,
                                (
                                    "onion",
                                    self.num_items_for_soup,
                                    self.soup_cooking_time,
                                ),
                            )
                        )
                    else:
                        player.set_object(ObjectState(obj, player.position))
            return start_state

        return start_state_fn

    def is_terminal(self, state):
        # There is a finite horizon, handled by the environment.
        if state.order_list is None:
            return False
        return len(state.order_list) == 0

    def get_valid_player_positions(self):
        return self.terrain_pos_dict[" "]

    def get_valid_joint_player_positions(self):
        """Returns all valid tuples of the form (p0_pos, p1_pos, p2_pos, ...)"""
        valid_positions = self.get_valid_player_positions()
        all_joint_positions = list(itertools.product(valid_positions, repeat=self.num_players))
        valid_joint_positions = [j_pos for j_pos in all_joint_positions if not self.is_joint_position_collision(j_pos)]
        return valid_joint_positions

    def get_valid_player_positions_and_orientations(self):
        valid_states = []
        for pos in self.get_valid_player_positions():
            valid_states.extend([(pos, d) for d in Direction.ALL_DIRECTIONS])
        return valid_states

    def get_valid_joint_player_positions_and_orientations(self):
        """All joint player position and orientation pairs that are not
        overlapping and on empty terrain."""
        valid_player_states = self.get_valid_player_positions_and_orientations()

        valid_joint_player_states = []
        for players_pos_and_orientations in itertools.product(valid_player_states, repeat=self.num_players):
            joint_position = [plyer_pos_and_or[0] for plyer_pos_and_or in players_pos_and_orientations]
            if not self.is_joint_position_collision(joint_position):
                valid_joint_player_states.append(players_pos_and_orientations)

        return valid_joint_player_states

    def get_adjacent_features(self, player):
        adj_feats = []
        pos = player.position
        for d in Direction.ALL_DIRECTIONS:
            adj_pos = Action.move_in_direction(pos, d)
            adj_feats.append((pos, self.get_terrain_type_at_pos(adj_pos)))
        return adj_feats

    def get_terrain_type_at_pos(self, pos):
        x, y = pos
        return self.terrain_mtx[y][x]

    def get_dish_dispenser_locations(self):
        return list(self.terrain_pos_dict["D"])

    def get_onion_dispenser_locations(self):
        return list(self.terrain_pos_dict["O"])

    def get_tomato_dispenser_locations(self):
        return list(self.terrain_pos_dict["T"])

    def get_serving_locations(self):
        return list(self.terrain_pos_dict["S"])

    def get_pot_locations(self):
        return list(self.terrain_pos_dict["P"])

    def get_counter_locations(self):
        return list(self.terrain_pos_dict["X"])

    def get_pot_states(self, state):
        """Returns dict with structure:
        {
         empty: [ObjStates]
         onion: {
            'x_items': [soup objects with x items],
            'cooking': [ready soup objs]
            'ready': [ready soup objs],
            'partially_full': [all non-empty and non-full soups]
            }
         tomato: same dict structure as above
        }
        """
        pots_states_dict = {}
        pots_states_dict["empty"] = []
        pots_states_dict["onion"] = defaultdict(list)
        pots_states_dict["tomato"] = defaultdict(list)
        for pot_pos in self.get_pot_locations():
            if not state.has_object(pot_pos):
                pots_states_dict["empty"].append(pot_pos)
            else:
                soup_obj = state.get_object(pot_pos)
                soup_type, num_items, cook_time = soup_obj.state
                if 0 < num_items < self.num_items_for_soup:
                    pots_states_dict[soup_type][f"{num_items}_items"].append(pot_pos)
                elif num_items == self.num_items_for_soup:
                    assert cook_time <= self.soup_cooking_time
                    if cook_time == self.soup_cooking_time:
                        pots_states_dict[soup_type]["ready"].append(pot_pos)
                    else:
                        pots_states_dict[soup_type]["cooking"].append(pot_pos)
                else:
                    raise ValueError(f"Pot with more than {self.num_items_for_soup} items")

                if 0 < num_items < self.num_items_for_soup:
                    pots_states_dict[soup_type]["partially_full"].append(pot_pos)

        return pots_states_dict

    def get_counter_objects_dict(self, state, counter_subset=None):
        """Returns a dictionary of pos:objects on counters by type"""
        counters_considered = self.terrain_pos_dict["X"] if counter_subset is None else counter_subset
        counter_objects_dict = defaultdict(list)
        for obj in state.objects.values():
            if obj.position in counters_considered:
                counter_objects_dict[obj.name].append(obj.position)
        return counter_objects_dict

    def get_empty_counter_locations(self, state):
        counter_locations = self.get_counter_locations()
        return [pos for pos in counter_locations if not state.has_object(pos)]

    def get_state_transition(self, state, joint_action):
        """Gets information about possible transitions for the action.

        Returns the next state, sparse reward and reward shaping.
        Assumes all actions are deterministic.

        NOTE: Sparse reward is given only when soups are delivered,
        shaped reward is given only for completion of subgoals
        (not soup deliveries).
        """
        assert not self.is_terminal(state), f"Trying to find successor of a terminal state: {state}"
        for action, action_set in zip(joint_action, self.get_actions(state)):
            if action not in action_set:
                raise ValueError(f"Illegal action {action} in state {state}")

        new_state = state.deepcopy()

        # Resolve interacts first
        (
            sparse_reward_by_agent,
            shaped_reward_by_agent,
            shaped_info_by_agent,
        ) = self.resolve_interacts(new_state, joint_action)

        assert new_state.player_positions == state.player_positions
        assert new_state.player_orientations == state.player_orientations

        # Resolve player movements
        self.resolve_movement(new_state, joint_action, shaped_info_by_agent)

        # Finally, environment effects
        self.step_environment_effects(new_state)

        # Additional dense reward logic
        # shaped_reward += self.calculate_distance_based_shaped_reward(state, new_state)

        infos = {
            "sparse_reward_by_agent": sparse_reward_by_agent,
            "shaped_reward_by_agent": shaped_reward_by_agent,
            "shaped_info_by_agent": shaped_info_by_agent,
        }

        return new_state, infos

    def resolve_interacts(self, new_state, joint_action):
        """
        Resolve any INTERACT actions, if present.

        Currently if two players both interact with a terrain, we resolve player 1's interact
        first and then player 2's, without doing anything like collision checking.
        """
        pot_states = self.get_pot_states(new_state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = (
            cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"]["partially_full"]
        )

        sparse_reward, shaped_reward = [0] * self.num_players, [0] * self.num_players
        # MARK: shaped_info by agent
        shaped_info = [{k: 0 for k in SHAPED_INFOS} for _ in range(self.num_players)]
        for player_idx, player, action in zip(range(self.num_players), new_state.players, joint_action):
            if action != Action.INTERACT:
                if action in Direction.ALL_DIRECTIONS:
                    shaped_info[player_idx]["MOVEMENT"] += 1
                elif action == Action.STAY:
                    shaped_info[player_idx]["STAY"] += 1
                continue

            pos, o = player.position, player.orientation
            i_pos = Action.move_in_direction(pos, o)
            terrain_type = self.get_terrain_type_at_pos(i_pos)

            if terrain_type == "X":
                if player.has_object() and not new_state.has_object(i_pos):
                    shaped_info[player_idx][f"put_{player.get_object().name}_on_X"] += 1
                    new_state.add_object(player.remove_object(), i_pos)
                elif not player.has_object() and new_state.has_object(i_pos):
                    player.set_object(new_state.remove_object(i_pos))
                    shaped_info[player_idx][f"pickup_{player.get_object().name}_from_X"] += 1
                else:
                    shaped_info[player_idx]["IDLE_INTERACT_X"] += 1

            elif terrain_type == "O" and player.held_object is None:
                player.set_object(ObjectState("onion", pos))
                shaped_info[player_idx][f"pickup_{player.get_object().name}_from_O"] += 1
            elif terrain_type == "T" and player.held_object is None:
                # MARK: pick tomato
                player.set_object(ObjectState("tomato", pos))
                shaped_info[player_idx][f"pickup_{player.get_object().name}_from_T"] += 1
            elif terrain_type == "D" and player.held_object is None:
                dishes_already = len(new_state.player_objects_by_type["dish"])
                player.set_object(ObjectState("dish", pos))

                dishes_on_counters = self.get_counter_objects_dict(new_state)["dish"]
                if len(nearly_ready_pots) > dishes_already and len(dishes_on_counters) == 0:
                    shaped_reward[player_idx] += self.reward_shaping_params["DISH_PICKUP_REWARD"]
                    shaped_info[player_idx]["USEFUL_DISH_PICKUP"] += 1
                shaped_info[player_idx][f"pickup_{player.get_object().name}_from_D"] += 1

            elif terrain_type == "P" and player.has_object():
                if player.get_object().name == "dish" and new_state.has_object(i_pos):
                    obj = new_state.get_object(i_pos)
                    assert obj.name == "soup", "Object in pot was not soup"
                    _, num_items, cook_time = obj.state
                    if num_items == self.num_items_for_soup and cook_time >= self.soup_cooking_time:
                        player.remove_object()  # Turn the dish into the soup
                        player.set_object(new_state.remove_object(i_pos))
                        shaped_reward[player_idx] += self.reward_shaping_params["SOUP_PICKUP_REWARD"]
                        shaped_info[player_idx]["SOUP_PICKUP"] += 1

                elif player.get_object().name in ["onion", "tomato"]:
                    item_type = player.get_object().name

                    if not new_state.has_object(i_pos):
                        # Pot was empty
                        player.remove_object()
                        new_state.add_object(ObjectState("soup", i_pos, (item_type, 1, 0)), i_pos)
                        shaped_reward[player_idx] += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]
                        shaped_info[player_idx]["PLACEMENT_IN_POT"] += 1

                    else:
                        # Pot has already items in it
                        obj = new_state.get_object(i_pos)
                        assert obj.name == "soup", "Object in pot was not soup"
                        soup_type, num_items, cook_time = obj.state
                        if num_items < self.num_items_for_soup and soup_type == item_type:
                            player.remove_object()
                            obj.state = (soup_type, num_items + 1, 0)
                            shaped_reward[player_idx] += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]
                            shaped_info[player_idx]["PLACEMENT_IN_POT"] += 1
            elif terrain_type == "S" and player.has_object():
                obj = player.get_object()
                if obj.name == "soup":
                    new_state, delivery_rew = self.deliver_soup(new_state, player, obj)
                    sparse_reward[player_idx] += delivery_rew
                    shaped_info[player_idx]["delivery"] += 1

                    # If last soup necessary was delivered, stop resolving interacts
                    if new_state.order_list is not None and len(new_state.order_list) == 0:
                        break
            else:
                shaped_info[player_idx]["IDLE_INTERACT_EMPTY"] += 1

        return sparse_reward, shaped_reward, shaped_info

    def deliver_soup(self, state, player, soup_obj):
        """
        Deliver the soup, and get reward if there is no order list
        or if the type of the delivered soup matches the next order.
        """
        soup_type, num_items, cook_time = soup_obj.state
        assert soup_type in ObjectState.SOUP_TYPES
        assert num_items == self.num_items_for_soup
        assert cook_time >= self.soup_cooking_time, "Cook time {} mdp cook time {}".format(
            cook_time, self.soup_cooking_time
        )
        player.remove_object()

        if state.order_list is None:
            return state, self.delivery_reward

        # If the delivered soup is the one currently required
        assert not self.is_terminal(state)
        current_order = state.order_list[0]
        if current_order == "any" or soup_type == current_order:
            state.order_list = state.order_list[1:]
            return state, self.delivery_reward

        return state, 0

    def resolve_movement(self, state, joint_action, shaped_info_by_agent=None):
        """Resolve player movement and deal with possible collisions"""
        old_positions, old_orientations = [p.position for p in state.players], [p.orientation for p in state.players]
        new_positions, new_orientations = self.compute_new_positions_and_orientations(state.players, joint_action)
        if shaped_info_by_agent is not None:
            for player_idx, (old_pos, new_pos, old_o, new_o) in enumerate(
                zip(old_positions, new_positions, old_orientations, new_orientations)
            ):
                if joint_action[player_idx] in Direction.ALL_DIRECTIONS and old_pos == new_pos and old_o == new_o:
                    shaped_info_by_agent[player_idx]["IDLE_MOVEMENT"] += 1
        for player_state, new_pos, new_o in zip(state.players, new_positions, new_orientations):
            player_state.update_pos_and_or(new_pos, new_o)

    def compute_new_positions_and_orientations(self, old_player_states, joint_action):
        """Compute new positions and orientations ignoring collisions"""
        new_positions, new_orientations = list(
            zip(
                *[
                    self._move_if_direction(p.position, p.orientation, a)
                    for p, a in zip(old_player_states, joint_action)
                ]
            )
        )
        old_positions = tuple(p.position for p in old_player_states)
        new_positions = self._handle_collisions(old_positions, new_positions)
        return new_positions, new_orientations

    def is_transition_collision(self, old_positions, new_positions):
        # Checking for any players ending in same square
        if self.is_joint_position_collision(new_positions):
            return True
        # Check if any two players crossed paths
        for idx0, idx1 in itertools.combinations(range(self.num_players), 2):
            p1_old, p2_old = old_positions[idx0], old_positions[idx1]
            p1_new, p2_new = new_positions[idx0], new_positions[idx1]
            if p1_new == p2_old and p1_old == p2_new:
                return True
        return False

    def is_joint_position_collision(self, joint_position):
        return any(pos0 == pos1 for pos0, pos1 in itertools.combinations(joint_position, 2))

    def step_environment_effects(self, state):
        reward = 0
        state.timestep += 1
        for obj in state.objects.values():
            if obj.name == "soup":
                x, y = obj.position
                soup_type, num_items, cook_time = obj.state
                # NOTE: cook_time is capped at self.soup_cooking_time
                if (
                    self.terrain_mtx[y][x] == "P"
                    and num_items == self.num_items_for_soup
                    and cook_time < self.soup_cooking_time
                ):
                    obj.state = soup_type, num_items, cook_time + 1
        return reward

    def _handle_collisions(self, old_positions, new_positions):
        """If agents collide, they stay at their old locations"""
        if self.is_transition_collision(old_positions, new_positions):
            return old_positions
        return new_positions

    def _get_terrain_type_pos_dict(self):
        pos_dict = defaultdict(list)
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, terrain_type in enumerate(terrain_row):
                pos_dict[terrain_type].append((x, y))
        return pos_dict

    def _move_if_direction(self, position, orientation, action):
        """Returns position and orientation that would
        be obtained after executing action"""
        if action == Action.INTERACT:
            return position, orientation
        new_pos = Action.move_in_direction(position, action)
        new_orientation = orientation if action == Action.STAY else action
        if new_pos not in self.get_valid_player_positions():
            return position, new_orientation
        return new_pos, new_orientation

    def _check_valid_state(self, state):
        """Checks that the state is valid.

        Conditions checked:
        - Players are on free spaces, not terrain
        - Held objects have the same position as the player holding them
        - Non-held objects are on terrain
        - No two players or non-held objects occupy the same position
        - Objects have a valid state (eg. no pot with 4 onions)
        """
        all_objects = list(state.objects.values())
        for player_state in state.players:
            # Check that players are not on terrain
            pos = player_state.position
            assert pos in self.get_valid_player_positions()

            # Check that held objects have the same position
            if player_state.held_object is not None:
                all_objects.append(player_state.held_object)
                assert player_state.held_object.position == player_state.position

        for obj_pos, obj_state in state.objects.items():
            # Check that the hash key position agrees with the position stored
            # in the object state
            assert obj_state.position == obj_pos
            # Check that non-held objects are on terrain
            assert self.get_terrain_type_at_pos(obj_pos) != " "

        # Check that players and non-held objects don't overlap
        all_pos = [player_state.position for player_state in state.players]
        all_pos += [obj_state.position for obj_state in state.objects.values()]
        assert len(all_pos) == len(set(all_pos)), "Overlapping players or objects"

        # Check that objects have a valid state
        for obj_state in all_objects:
            assert obj_state.is_valid()

    @staticmethod
    def _assert_valid_grid(grid):
        """Raises an AssertionError if the grid is invalid.

        grid:  A sequence of sequences of spaces, representing a grid of a
        certain height and width. grid[y][x] is the space at row y and column
        x. A space must be either 'X' (representing a counter), ' ' (an empty
        space), 'O' (onion supply), 'P' (pot), 'D' (dish supply), 'S' (serving
        location), '1' (player 1) and '2' (player 2).
        """
        height = len(grid)
        width = len(grid[0])

        # Make sure the grid is not ragged
        assert all(len(row) == width for row in grid), "Ragged grid"

        # Borders must not be free spaces
        def is_not_free(c):
            return c in "XOPDST"

        for y in range(height):
            assert is_not_free(grid[y][0]), "Left border must not be free"
            assert is_not_free(grid[y][-1]), "Right border must not be free"
        for x in range(width):
            assert is_not_free(grid[0][x]), "Top border must not be free"
            assert is_not_free(grid[-1][x]), "Bottom border must not be free"

        all_elements = [element for row in grid for element in row]
        digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        layout_digits = [e for e in all_elements if e in digits]
        num_players = len(layout_digits)
        assert num_players > 0, "No players (digits) in grid"
        layout_digits = list(sorted(map(int, layout_digits)))
        assert layout_digits == list(range(1, num_players + 1)), "Some players were missing"

        assert all(c in "XOPDST123456789 " for c in all_elements), "Invalid character in grid"
        assert all_elements.count("1") == 1, "'1' must be present exactly once"
        assert all_elements.count("D") >= 1, "'D' must be present at least once"
        assert all_elements.count("S") >= 1, "'S' must be present at least once"
        assert all_elements.count("P") >= 1, "'P' must be present at least once"
        assert all_elements.count("O") >= 1 or all_elements.count("T") >= 1, "'O' or 'T' must be present at least once"

    #####################
    # TERMINAL GRAPHICS #
    #####################

    def state_string(self, state):
        """String representation of the current state"""
        players_dict = {player.position: player for player in state.players}

        grid_string = ""
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, element in enumerate(terrain_row):
                if (x, y) in players_dict.keys():
                    player = players_dict[(x, y)]
                    orientation = player.orientation
                    assert orientation in Direction.ALL_DIRECTIONS

                    grid_string += Action.ACTION_TO_CHAR[orientation]
                    player_object = player.held_object
                    if player_object:
                        grid_string += player_object.name[:1]
                    else:
                        player_idx_lst = [i for i, p in enumerate(state.players) if p.position == player.position]
                        assert len(player_idx_lst) == 1
                        grid_string += str(player_idx_lst[0])
                else:
                    if element == "X" and state.has_object((x, y)):
                        state_obj = state.get_object((x, y))
                        grid_string = grid_string + element + state_obj.name[:1]

                    elif element == "P" and state.has_object((x, y)):
                        soup_obj = state.get_object((x, y))
                        soup_type, num_items, cook_time = soup_obj.state
                        if soup_type == "onion":
                            grid_string += "ø"
                        elif soup_type == "tomato":
                            grid_string += "†"
                        else:
                            raise ValueError()

                        if num_items == self.num_items_for_soup:
                            grid_string += str(cook_time)

                        # NOTE: do not currently have terminal graphics
                        # support for cooking times greater than 3.
                        elif num_items == 2:
                            grid_string += "="
                        else:
                            grid_string += "-"
                    else:
                        grid_string += element + " "

            grid_string += "\n"

        if state.order_list is not None:
            grid_string += "Current orders: {}/{} are any's\n".format(
                len(state.order_list),
                len([order == "any" for order in state.order_list]),
            )
        return grid_string

    ###################
    # STATE ENCODINGS #
    ###################

    def lossless_state_encoding(
        self,
        overcooked_state: OvercookedState,
        debug=False,
        add_timestep: bool = False,
        add_identity: bool = False,
        horizon: int = 400,
    ):
        """Featurizes a OvercookedState object into a stack of boolean masks that are easily readable by a CNN"""
        assert type(debug) is bool
        assert not add_timestep or horizon
        base_map_features = [
            "pot_loc",
            "counter_loc",
            "onion_disp_loc",
            "dish_disp_loc",
            "serve_loc",
        ]
        variable_map_features = [
            "onions_in_pot",
            "onions_cook_time",
            "onion_soup_loc",
            "dishes",
            "onions",
        ]

        if add_timestep:
            timestep_features = ["timestep"]
        if add_identity:
            identity_features = ["player_id"]

        all_objects = overcooked_state.all_objects_list

        def make_layer(position, value):
            layer = np.zeros(self.shape)
            layer[position] = value
            return layer

        def process_for_player(primary_agent_idx):
            # Ensure that primary_agent_idx layers are ordered before other_agent_idx layers
            other_agent_idx = 1 - primary_agent_idx
            ordered_player_features = [
                f"player_{primary_agent_idx}_loc",
                f"player_{other_agent_idx}_loc",
            ] + [
                f"player_{i}_orientation_{Direction.DIRECTION_TO_INDEX[d]}"
                for i, d in itertools.product([primary_agent_idx, other_agent_idx], Direction.ALL_DIRECTIONS)
            ]

            LAYERS = ordered_player_features + base_map_features + variable_map_features
            if add_timestep:
                LAYERS += timestep_features
            if add_identity:
                LAYERS += identity_features

            state_mask_dict = {k: np.zeros(self.shape) for k in LAYERS}

            if add_timestep:
                state_mask_dict["timestep"] = np.ones(self.shape) * (horizon - overcooked_state.timestep) / horizon
            if add_identity:
                state_mask_dict["player_id"] = np.ones(self.shape) * primary_agent_idx

            # MAP LAYERS
            for loc in self.get_counter_locations():
                state_mask_dict["counter_loc"][loc] = 1

            for loc in self.get_pot_locations():
                state_mask_dict["pot_loc"][loc] = 1

            for loc in self.get_onion_dispenser_locations():
                state_mask_dict["onion_disp_loc"][loc] = 1

            for loc in self.get_dish_dispenser_locations():
                state_mask_dict["dish_disp_loc"][loc] = 1

            for loc in self.get_serving_locations():
                state_mask_dict["serve_loc"][loc] = 1

            # PLAYER LAYERS
            for i, player in enumerate(overcooked_state.players):
                player_orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
                state_mask_dict[f"player_{i}_loc"] = make_layer(player.position, 1)
                state_mask_dict[f"player_{i}_orientation_{player_orientation_idx}"] = make_layer(player.position, 1)

            # state_mask_dict["player_id"] = np.ones(self.shape) * primary_agent_idx

            # OBJECT & STATE LAYERS
            for obj in all_objects:
                if obj.name == "soup":
                    soup_type, num_onions, cook_time = obj.state
                    if soup_type == "onion":
                        if obj.position in self.get_pot_locations():
                            soup_type, num_onions, cook_time = obj.state
                            state_mask_dict["onions_in_pot"] += make_layer(obj.position, num_onions)
                            state_mask_dict["onions_cook_time"] += make_layer(obj.position, cook_time)
                        else:
                            # If player soup is not in a pot, put it in separate mask
                            state_mask_dict["onion_soup_loc"] += make_layer(obj.position, 1)
                    else:
                        raise ValueError("Unrecognized soup")

                elif obj.name == "dish":
                    state_mask_dict["dishes"] += make_layer(obj.position, 1)
                elif obj.name == "onion":
                    state_mask_dict["onions"] += make_layer(obj.position, 1)
                else:
                    raise ValueError("Unrecognized object")

            if debug:
                print(len(LAYERS))
                print(len(state_mask_dict))
                for k, v in state_mask_dict.items():
                    print(k)
                    print(np.transpose(v, (1, 0)))

            # Stack of all the state masks, order decided by order of LAYERS
            state_mask_stack = np.array([state_mask_dict[layer_id] for layer_id in LAYERS])
            state_mask_stack = np.transpose(state_mask_stack, (1, 2, 0))
            assert state_mask_stack.shape[:2] == self.shape
            assert state_mask_stack.shape[2] == len(LAYERS)
            # NOTE: currently not including time left or order_list in featurization
            return np.array(state_mask_stack).astype(float)
            # return np.array(state_mask_stack).astype(int)

        # NOTE: Currently not very efficient, a decent amount of computation repeated here
        num_players = len(overcooked_state.players)
        final_obs_for_players = tuple(process_for_player(i) for i in range(num_players))
        # print(f"timestep {overcooked_state.timestep}")
        # print(final_obs_for_players[0][:, :, -1])
        return final_obs_for_players

    def featurize_state(self, overcooked_state, mlp):
        """
        Encode state with some manually designed features.
        NOTE: currently works for just two players.
        """

        all_features = {}

        def make_closest_feature(idx, name, locations):
            "Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict"
            all_features[f"p{idx}_closest_{name}"] = self.get_deltas_to_closest_location(player, locations, mlp)

        IDX_TO_OBJ = ["onion", "soup", "dish"]
        OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(IDX_TO_OBJ)}

        counter_objects = self.get_counter_objects_dict(overcooked_state)
        pot_state = self.get_pot_states(overcooked_state)

        # Player Info
        for i, player in enumerate(overcooked_state.players):
            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            all_features[f"p{i}_orientation"] = np.eye(4)[orientation_idx]
            obj = player.held_object

            if obj is None:
                held_obj_name = "none"
                all_features[f"p{i}_objs"] = np.zeros(len(IDX_TO_OBJ))
            else:
                held_obj_name = obj.name
                obj_idx = OBJ_TO_IDX[held_obj_name]
                all_features[f"p{i}_objs"] = np.eye(len(IDX_TO_OBJ))[obj_idx]

            # Closest feature of each type
            if held_obj_name == "onion":
                all_features[f"p{i}_closest_onion"] = (0, 0)
            else:
                make_closest_feature(
                    i,
                    "onion",
                    self.get_onion_dispenser_locations() + counter_objects["onion"],
                )

            make_closest_feature(i, "empty_pot", pot_state["empty"])
            make_closest_feature(i, "one_onion_pot", pot_state["onion"]["one_onion"])
            make_closest_feature(i, "two_onion_pot", pot_state["onion"]["two_onion"])
            make_closest_feature(i, "cooking_pot", pot_state["onion"]["cooking"])
            make_closest_feature(i, "ready_pot", pot_state["onion"]["ready"])

            if held_obj_name == "dish":
                all_features[f"p{i}_closest_dish"] = (0, 0)
            else:
                make_closest_feature(
                    i,
                    "dish",
                    self.get_dish_dispenser_locations() + counter_objects["dish"],
                )

            if held_obj_name == "soup":
                all_features[f"p{i}_closest_soup"] = (0, 0)
            else:
                make_closest_feature(i, "soup", counter_objects["soup"])

            make_closest_feature(i, "serving", self.get_serving_locations())

            for direction, pos_and_feat in enumerate(self.get_adjacent_features(player)):
                adj_pos, feat = pos_and_feat

                if direction == player.orientation:
                    # Check if counter we are facing is empty
                    facing_counter = feat == "X" and adj_pos not in overcooked_state.objects.keys()
                    facing_counter_feature = [1] if facing_counter else [0]
                    all_features[f"p{i}_facing_empty_counter"] = facing_counter_feature

                all_features[f"p{i}_wall_{direction}"] = [0] if feat == " " else [1]

        features_np = {k: np.array(v) for k, v in all_features.items()}

        p0, p1 = overcooked_state.players
        p0_dict = {k: v for k, v in features_np.items() if k[:2] == "p0"}
        p1_dict = {k: v for k, v in features_np.items() if k[:2] == "p1"}
        p0_features = np.concatenate(list(p0_dict.values()))
        p1_features = np.concatenate(list(p1_dict.values()))

        p1_rel_to_p0 = np.array(pos_distance(p1.position, p0.position))
        abs_pos_p0 = np.array(p0.position)
        ordered_features_p0 = np.squeeze(np.concatenate([p0_features, p1_features, p1_rel_to_p0, abs_pos_p0]))

        p0_rel_to_p1 = np.array(pos_distance(p0.position, p1.position))
        abs_pos_p1 = np.array(p0.position)
        ordered_features_p1 = np.squeeze(np.concatenate([p1_features, p0_features, p0_rel_to_p1, abs_pos_p1]))
        return ordered_features_p0, ordered_features_p1

    def get_deltas_to_closest_location(self, player, locations, mlp):
        _, closest_loc = mlp.mp.min_cost_to_feature(player.pos_and_or, locations, with_argmin=True)
        if closest_loc is None:
            # "any object that does not exist or I am carrying is going to show up as a (0,0)
            # but I can disambiguate the two possibilities by looking at the features
            # for what kind of object I'm carrying"
            return (0, 0)
        dy_loc, dx_loc = pos_distance(closest_loc, player.position)
        return dy_loc, dx_loc

    ##############
    # DEPRECATED #
    ##############

    def calculate_distance_based_shaped_reward(self, state, new_state):
        """
        Adding reward shaping based on distance to certain features.
        """
        distance_based_shaped_reward = 0

        pot_states = self.get_pot_states(new_state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = (
            cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"]["partially_full"]
        )
        dishes_in_play = len(new_state.player_objects_by_type["dish"])
        for player_old, player_new in zip(state.players, new_state.players):
            # Linearly increase reward depending on vicinity to certain features, where distance of 10 achieves 0 reward
            max_dist = 8

            if (
                player_new.held_object is not None
                and player_new.held_object.name == "dish"
                and len(nearly_ready_pots) >= dishes_in_play
            ):
                min_dist_to_pot_new = np.inf
                min_dist_to_pot_old = np.inf
                for pot in nearly_ready_pots:
                    new_dist = np.linalg.norm(np.array(pot) - np.array(player_new.position))
                    old_dist = np.linalg.norm(np.array(pot) - np.array(player_old.position))
                    if new_dist < min_dist_to_pot_new:
                        min_dist_to_pot_new = new_dist
                    if old_dist < min_dist_to_pot_old:
                        min_dist_to_pot_old = old_dist
                if min_dist_to_pot_old > min_dist_to_pot_new:
                    distance_based_shaped_reward += self.reward_shaping_params["POT_DISTANCE_REW"] * (
                        1 - min(min_dist_to_pot_new / max_dist, 1)
                    )

            if player_new.held_object is None and len(cooking_pots) > 0 and dishes_in_play == 0:
                min_dist_to_d_new = np.inf
                min_dist_to_d_old = np.inf
                for serving_loc in self.terrain_pos_dict["D"]:
                    new_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_new.position))
                    old_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_old.position))
                    if new_dist < min_dist_to_d_new:
                        min_dist_to_d_new = new_dist
                    if old_dist < min_dist_to_d_old:
                        min_dist_to_d_old = old_dist

                if min_dist_to_d_old > min_dist_to_d_new:
                    distance_based_shaped_reward += self.reward_shaping_params["DISH_DISP_DISTANCE_REW"] * (
                        1 - min(min_dist_to_d_new / max_dist, 1)
                    )

            if player_new.held_object is not None and player_new.held_object.name == "soup":
                min_dist_to_s_new = np.inf
                min_dist_to_s_old = np.inf
                for serving_loc in self.terrain_pos_dict["S"]:
                    new_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_new.position))
                    old_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_old.position))
                    if new_dist < min_dist_to_s_new:
                        min_dist_to_s_new = new_dist

                    if old_dist < min_dist_to_s_old:
                        min_dist_to_s_old = old_dist

                if min_dist_to_s_old > min_dist_to_s_new:
                    distance_based_shaped_reward += self.reward_shaping_params["SOUP_DISTANCE_REW"] * (
                        1 - min(min_dist_to_s_new / max_dist, 1)
                    )

        return distance_based_shaped_reward
