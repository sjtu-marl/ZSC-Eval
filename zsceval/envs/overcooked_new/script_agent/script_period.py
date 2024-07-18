import random

import zsceval.envs.overcooked_new.script_agent.utils as utils
from zsceval.envs.overcooked_new.script_agent.base import BaseScriptPeriod


class Pickup_Object(BaseScriptPeriod):
    def __init__(self, obj, terrain_type="XPOTDS", random_put=True, random_pos=True):
        """Pickup some object at specific terrains
        obj: str
            "onion", "tomato" "dish", "soup"
        terrain_type: str
            example "XPOD"
        random_put: bool
            if True, put the irrelevant obj at random position
        random_pos: bool
            if True, find a random obj, otherwise the closest one
        """
        super().__init__(("random_" if random_pos else "") + "pickup_" + str(obj))

        self.__put_pos = None
        self.__obj_pos = None
        self.__random_pos = None

        self.random_put = random_put
        self.random_pos = random_pos
        self.target_obj = obj
        self.terrain_type = terrain_type

        if type(self.target_obj) != list:
            self.target_obj = [self.target_obj]

    def reset(self, mdp, state, player_idx):
        self.__put_pos = None
        self.__obj_pos = None
        self.__random_pos = None

    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if player.has_object() and player.get_object().name not in self.target_obj:
            # not target obj, place in random position
            action, self.__put_pos = utils.interact(
                mdp,
                state,
                player_idx,
                pre_goal=self.__put_pos,
                random=self.random_put,
                terrain_type="XOPDS",
                obj=["can_put"],
            )
            return action

        if not player.has_object():
            # find target obj
            action, self.__obj_pos = utils.interact(
                mdp,
                state,
                player_idx,
                pre_goal=self.__obj_pos,
                random=self.random_pos,
                terrain_type=self.terrain_type,
                obj=self.target_obj,
            )
            return action

        action, self.__random_pos = utils.random_move(mdp, state, player_idx, pre_goal=self.__random_pos)
        return action

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return player.has_object() and player.get_object().name in self.target_obj


class Put_Object(BaseScriptPeriod):
    def __init__(
        self,
        terrain_type="XPTODS",
        random_put=True,
        obj="can_put",
        pos_mask=None,
        move_mask=None,
    ):
        """Pickup some object at specific terrains
        terrain_type: str
            example "XPTODS"
        random_put: bool
            if True, put the irrelevant obj at random position
        """
        super().__init__(("random_" if random_put else "") + "put")

        self.__put_pos = None
        self.__random_pos = None
        self.__obj = obj

        self.random_put = random_put
        self.terrain_type = terrain_type
        self.pos_mask = pos_mask
        self.move_mask = move_mask

    def reset(self, mdp, state, player_idx):
        self.__put_pos = None
        self.__random_pos = None

    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if player.has_object():
            # not target obj, place in random position
            action, self.__put_pos = utils.interact(
                mdp,
                state,
                player_idx,
                pre_goal=self.__put_pos,
                random=self.random_put,
                terrain_type=self.terrain_type,
                obj=[self.__obj] if type(self.__obj) == str else self.__obj,
                pos_mask=self.pos_mask,
                move_mask=self.move_mask,
            )
            return action

        action, self.__random_pos = utils.random_move(
            mdp, state, player_idx, pre_goal=self.__random_pos, move_mask=self.move_mask
        )
        return action

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return not player.has_object()


class Pickup_Ingredient_and_Place_in_Pot(BaseScriptPeriod):
    def __init__(
        self,
        random_put=True,
        random_pot=True,
        random_ingredient=True,
        obj=["tomato", "onion"],
    ):
        """
        random_put: bool
            if True, place the object to random position when the player starts with
        random_pot: bool
            if True, find a random pot to place
        random_ingredient: bool
            if True, take a random ingredient
        """
        super().__init__(period_name="Pickup_Ingredient_and_Place_in_Pot")

        self.random_put = random_put
        self.random_pot = random_pot
        self.random_ingredient = random_ingredient
        self.target_obj = obj if type(obj) is list else [obj]

        self.__stage = 1
        self.__current_period = Pickup_Object(
            obj=self.target_obj,
            terrain_type="OTX",
            random_put=self.random_put,
            random_pos=self.random_ingredient,
        )

    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Object(
            obj=self.target_obj,
            terrain_type="OTX",
            random_put=self.random_put,
            random_pos=self.random_ingredient,
        )

    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                assert player.has_object() and player.get_object().name in self.target_obj
                self.__stage = 2
                self.__current_period = Put_Object(terrain_type="P", random_put=self.random_pot)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return self.__stage == 2 and not player.has_object()


class Pickup_Onion_and_Place_in_Pot(Pickup_Ingredient_and_Place_in_Pot):
    def __init__(self, random_put=True, random_pot=True, random_onion=True):
        super().__init__(
            obj="onion",
            random_put=random_put,
            random_pot=random_pot,
            random_ingredient=random_onion,
        )


class Pickup_Tomato_and_Place_in_Pot(Pickup_Ingredient_and_Place_in_Pot):
    def __init__(self, random_put=True, random_pot=True, random_tomato=True):
        super().__init__(
            obj="tomato",
            random_put=random_put,
            random_pot=random_pot,
            random_ingredient=random_tomato,
        )


class Pickup_Ingredient_and_Place_Mix(BaseScriptPeriod):
    def __init__(
        self,
        random_put=True,
        random_pot=True,
        random_ingredient=True,
        obj=["tomato", "onion"],
    ):
        """
        random_put: bool
            if True, place the object to random position when the player starts with
        random_pot: bool
            if True, find a random pot to place
        random_ingredient: bool
            if True, take a random ingredient
        """
        super().__init__(period_name="Pickup_Ingredient_and_Place_in_Pot")

        self.random_put = random_put
        self.random_pot = random_pot
        self.random_ingredient = random_ingredient
        self.target_obj = obj
        self.__put_pos = None
        self.__random_pos = None

        self.__stage = 1
        self.__current_period = Pickup_Object(
            obj=random.choice(self.target_obj),
            terrain_type="OTX",
            random_put=self.random_put,
            random_pos=self.random_ingredient,
        )

    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Object(
            obj=random.choice(self.target_obj),
            terrain_type="OTX",
            random_put=self.random_put,
            random_pos=self.random_ingredient,
        )

    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            # print(self.__current_period.target_obj, self.target_obj)
            if self.__current_period.done(mdp, state, player_idx):
                assert player.has_object() and player.get_object().name in self.target_obj
                self.__stage = 2
                if player.get_object().name == "onion":
                    obj = ["unfull_soup_t", "unfull_soup_ot"]
                elif player.get_object().name == "tomato":
                    obj = ["unfull_soup_o", "unfull_soup_ot"]
                else:
                    raise RuntimeError(f"Unexpected: Player has object {player.get_object().name}")
                self.__current_period = Put_Object(obj=obj, terrain_type="P", random_put=self.random_pot)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        current_obj = player.get_object().name
        assert current_obj in ["onion", "tomato"]
        if current_obj == "onion":
            if utils.exists(mdp, state, player_idx, "P", ["unfull_soup_1t"]):
                action, self.__put_pos = utils.interact(
                    mdp,
                    state,
                    player_idx,
                    pre_goal=self.__put_pos,
                    random=self.random_put,
                    terrain_type="P",
                    obj=["unfull_soup_1t"],
                )
                return action
            elif utils.exists(mdp, state, player_idx, "P", ["empty"]):
                action, self.__put_pos = utils.interact(
                    mdp,
                    state,
                    player_idx,
                    pre_goal=self.__put_pos,
                    random=self.random_put,
                    terrain_type="P",
                    obj=["empty"],
                )
                return action
        else:
            if utils.exists(mdp, state, player_idx, "P", ["unfull_soup_1o"]):
                action, self.__put_pos = utils.interact(
                    mdp,
                    state,
                    player_idx,
                    pre_goal=self.__put_pos,
                    random=self.random_put,
                    terrain_type="P",
                    obj=["unfull_soup_1o"],
                )
                return action
            elif utils.exists(mdp, state, player_idx, "P", ["empty"]):
                action, self.__put_pos = utils.interact(
                    mdp,
                    state,
                    player_idx,
                    pre_goal=self.__put_pos,
                    random=self.random_put,
                    terrain_type="P",
                    obj=["empty"],
                )
                return action
        action, self.__random_pos = utils.random_move(mdp, state, player_idx, pre_goal=self.__random_pos)
        return action

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return self.__stage == 2 and not player.has_object()


class Pickup_Onion_and_Place_Mix(Pickup_Ingredient_and_Place_Mix):
    def __init__(self, random_put=True, random_pot=True, random_onion=True):
        super().__init__(
            obj=["onion"],
            random_put=random_put,
            random_pot=random_pot,
            random_ingredient=random_onion,
        )


class Pickup_Tomato_and_Place_Mix(Pickup_Ingredient_and_Place_Mix):
    def __init__(self, random_put=True, random_pot=True, random_tomato=True):
        super().__init__(
            obj=["tomato"],
            random_put=random_put,
            random_pot=random_pot,
            random_ingredient=random_tomato,
        )


class Mixed_Order(BaseScriptPeriod):
    def __init__(
        self,
        random_put=True,
        random_pot=True,
        random_ingredient=True,
        obj=["tomato", "onion"],
    ):
        """
        random_put: bool
            if True, place the object to random position when the player starts with
        random_pot: bool
            if True, find a random pot to place
        random_ingredient: bool
            if True, take a random ingredient
        """
        super().__init__(period_name="Mixed_Order")

        self.random_put = random_put
        self.random_pot = random_pot
        self.random_ingredient = random_ingredient
        self.target_obj = obj
        self.__put_pos = None
        self.__random_pos = None

        self.__stage = 1
        self.__current_period = Pickup_Object(
            obj=random.choice(self.target_obj),
            terrain_type="OTX",
            random_put=self.random_put,
            random_pos=self.random_ingredient,
        )

    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Object(
            obj=random.choice(self.target_obj),
            terrain_type="OTX",
            random_put=self.random_put,
            random_pos=self.random_ingredient,
        )

    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            # print(self.__current_period.target_obj, self.target_obj)
            if self.__current_period.done(mdp, state, player_idx):
                assert player.has_object() and player.get_object().name in self.target_obj
                self.__stage = 2
                if player.get_object().name == "onion":
                    obj = ["unfull_soup_1t", "empty"]
                elif player.get_object().name == "tomato":
                    obj = ["unfull_soup_1o", "unfull_soup_ot", "empty"]
                else:
                    raise RuntimeError(f"Unexpected: Player has object {player.get_object().name}")
                self.__current_period = Put_Object(obj=obj, terrain_type="P", random_put=self.random_pot)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        current_obj = player.get_object().name
        assert current_obj in ["onion", "tomato"]
        if current_obj == "onion":
            if utils.exists(mdp, state, player_idx, "P", ["unfull_soup_1t"]):
                action, self.__put_pos = utils.interact(
                    mdp,
                    state,
                    player_idx,
                    pre_goal=self.__put_pos,
                    random=self.random_put,
                    terrain_type="P",
                    obj=["unfull_soup_1t"],
                )
                return action
            elif utils.exists(mdp, state, player_idx, "P", ["empty"]):
                action, self.__put_pos = utils.interact(
                    mdp,
                    state,
                    player_idx,
                    pre_goal=self.__put_pos,
                    random=self.random_put,
                    terrain_type="P",
                    obj=["empty"],
                )
                return action
        else:
            if utils.exists(mdp, state, player_idx, "P", ["unfull_soup_ot"]):
                action, self.__put_pos = utils.interact(
                    mdp,
                    state,
                    player_idx,
                    pre_goal=self.__put_pos,
                    random=self.random_put,
                    terrain_type="P",
                    obj=["unfull_soup_ot"],
                )
                return action
            elif utils.exists(mdp, state, player_idx, "P", ["unfull_soup_1o"]):
                action, self.__put_pos = utils.interact(
                    mdp,
                    state,
                    player_idx,
                    pre_goal=self.__put_pos,
                    random=self.random_put,
                    terrain_type="P",
                    obj=["unfull_soup_1o"],
                )
                return action
            elif utils.exists(mdp, state, player_idx, "P", ["unfull_soup_1t"]):
                action, self.__put_pos = utils.interact(
                    mdp,
                    state,
                    player_idx,
                    pre_goal=self.__put_pos,
                    random=self.random_put,
                    terrain_type="P",
                    obj=["unfull_soup_1o"],
                )
                return action
            elif utils.exists(mdp, state, player_idx, "P", ["empty"]):
                action, self.__put_pos = utils.interact(
                    mdp,
                    state,
                    player_idx,
                    pre_goal=self.__put_pos,
                    random=self.random_put,
                    terrain_type="P",
                    obj=["empty"],
                )
                return action
        action, self.__random_pos = utils.random_move(mdp, state, player_idx, pre_goal=self.__random_pos)
        return action

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return self.__stage == 2 and not player.has_object()


class Pickup_Ingredient_and_Place_Random(BaseScriptPeriod):
    def __init__(self, random_put=True, random_ingredient=True, obj=["onion", "tomato"]):
        """
        random_put: bool
            if True, place the object to random position when the player starts with
        random_ingredient: bool
            if True, take a random onion
        """
        super().__init__(period_name="Pickup_Ingredient_and_Place_Random")

        self.random_put = random_put
        self.random_ingredient = random_ingredient
        self.target_obj = obj if type(obj) == list else [obj]

        self.__stage = 1
        self.__current_period = Pickup_Object(
            obj=self.target_obj,
            terrain_type="OTX",
            random_put=self.random_put,
            random_pos=self.random_ingredient,
        )

    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Object(
            obj=self.target_obj,
            terrain_type="OTX",
            random_put=self.random_put,
            random_pos=self.random_ingredient,
        )

    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                assert player.has_object() and player.get_object().name in self.target_obj
                self.__stage = 2
                self.__current_period = Put_Object(terrain_type="XOPTDS", random_put=True)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return self.__stage == 2 and not player.has_object()


class Pickup_Onion_and_Place_Random(BaseScriptPeriod):
    def __init__(self, random_put=True, random_onion=True):
        super().__init__(random_put=random_put, random_ingredient=random_onion, obj="onion")


class Pickup_Tomato_and_Place_Random(BaseScriptPeriod):
    def __init__(self, random_put=True, random_tomato=True):
        super().__init__(random_put=random_put, random_ingredient=random_tomato, obj="tomato")


class Put_Ingredient_Everywhere(BaseScriptPeriod):
    def __init__(self, random_put=True, random_ingredient=True, obj=["onion", "tomato"]):
        """
        random_put: bool
            if True, place the object to random position when the player starts with
        random_ingredient: bool
            if True, take a random onion
        """
        super().__init__(period_name="Put_Ingredient_Everywhere")

        self.random_put = random_put
        self.random_ingredient = random_ingredient
        self.target_obj = obj if type(obj) == list else [obj]

        self.__stage = 1
        self.__current_period = Pickup_Object(
            obj=self.target_obj,
            terrain_type="OT",
            random_put=self.random_put,
            random_pos=self.random_ingredient,
        )

    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Object(
            obj=self.target_obj,
            terrain_type="OT",
            random_put=self.random_put,
            random_pos=self.random_ingredient,
        )

    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                assert player.has_object() and player.get_object().name in self.target_obj
                self.__stage = 2
                self.__current_period = Put_Object(terrain_type="X", random_put=True)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return self.__stage == 2 and not player.has_object()


class Put_Onion_Everywhere(Put_Ingredient_Everywhere):
    def __init__(self, random_put=True, random_onion=True):
        super().__init__(random_put=random_put, random_ingredient=random_onion, obj="onion")


class Put_Tomato_Everywhere(Put_Ingredient_Everywhere):
    def __init__(self, random_put=True, random_tomato=True):
        super().__init__(random_put=random_put, random_ingredient=random_tomato, obj="tomato")


class Pickup_Dish_and_Place_Random(BaseScriptPeriod):
    def __init__(self, random_put=True, random_dish=True):
        """
        random_put: bool
            if True, place the object to random position when the player starts with
        random_dish: bool
            if True, take a random dish
        """
        super().__init__(period_name="Pickup_Dish_and_Place_Random")

        self.random_put = random_put
        self.random_dish = random_dish

        self.__stage = 1
        self.__current_period = Pickup_Object(
            obj="dish",
            terrain_type="XOPDST",
            random_put=self.random_put,
            random_pos=self.random_dish,
        )

    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Object(
            obj="dish",
            terrain_type="XOPDTS",
            random_put=self.random_put,
            random_pos=self.random_dish,
        )

    def step(self, mdp, state, player_idx):
        state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                self.__stage = 2
                self.__current_period = Put_Object(terrain_type="XOPTDS", random_put=True)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return self.__stage == 2 and not player.has_object()


class Put_Dish_Everywhere(BaseScriptPeriod):
    def __init__(self, random_put=True, random_dish=True):
        """
        random_put: bool
            if True, place the object to random position when the player starts with
        random_dish: bool
            if True, take a random dish
        """
        super().__init__(period_name="Put_Dish_Everywhere")

        self.random_put = random_put
        self.random_dish = random_dish

        self.__stage = 1
        self.__current_period = Pickup_Object(
            obj="dish",
            terrain_type="D",
            random_put=self.random_put,
            random_pos=self.random_dish,
        )

    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Object(
            obj="dish",
            terrain_type="D",
            random_put=self.random_put,
            random_pos=self.random_dish,
        )

    def step(self, mdp, state, player_idx):
        state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                self.__stage = 2
                self.__current_period = Put_Object(terrain_type="X", random_put=True)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return self.__stage == 2 and not player.has_object()


class Pickup_Soup(BaseScriptPeriod):
    def __init__(self, random_dish=True, random_soup=True):
        super().__init__(period_name="Pickup_Soup")

        self.random_dish = random_dish
        self.random_soup = random_soup

        self.__stage = 1
        self.__current_period = Pickup_Object(
            obj="dish",
            terrain_type="XOTPDS",
            random_put=True,
            random_pos=self.random_dish,
        )

    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        if utils.exists(mdp, state, player_idx, terrain_type="X", obj="soup"):
            #  if there are soups on table, take that on table
            self.__current_period = Pickup_Object(
                obj="soup",
                terrain_type="XP",
                random_put=True,
                random_pos=self.random_soup,
            )
        else:
            self.__current_period = Pickup_Object(
                obj="dish",
                terrain_type="XOTPDS",
                random_put=True,
                random_pos=self.random_dish,
            )

    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                assert player.has_object() and player.get_object().name == "dish"
                self.__stage = 2
                # this is a quick hack to use put as pickup soup
                self.__current_period = Put_Object(
                    terrain_type="P",
                    random_put=self.random_soup,
                    obj=["soup", "cooking_soup"],
                )
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return player.has_object() and player.get_object().name == "soup"


class Pickup_Soup_and_Deliver(BaseScriptPeriod):
    def __init__(self, random_dish=True, random_soup=True):
        super().__init__(period_name="Pickup_Soup_and_Deliver")

        self.random_dish = random_dish
        self.random_soup = random_soup

        self.__stage = 1
        self.__current_period = Pickup_Soup(random_dish=self.random_dish, random_soup=self.random_soup)

    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Soup(random_dish=self.random_dish, random_soup=self.random_soup)

    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                assert player.has_object() and player.get_object().name == "soup"
                self.__stage = 2
                # this is a quick hack to use put as deliver
                self.__current_period = Put_Object(terrain_type="S", random_put=False)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        state.players[player_idx]
        return self.__stage == 2 and self.__current_period.done(mdp, state, player_idx)


class Pickup_Soup_and_Place_Random(BaseScriptPeriod):
    def __init__(self, random_dish=True, random_soup=True):
        super().__init__(period_name="Pickup_Soup_and_Place_Random")

        self.random_dish = random_dish
        self.random_soup = random_soup

        self.__stage = 1
        self.__current_period = Pickup_Soup(random_dish=self.random_dish, random_soup=self.random_soup)

    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Soup(random_dish=self.random_dish, random_soup=self.random_soup)

    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                assert player.has_object() and player.get_object().name == "soup"
                self.__stage = 2
                # this is a quick hack to use put as deliver
                self.__current_period = Put_Object(terrain_type="XOTPDS", random_put=True)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        state.players[player_idx]
        return self.__stage == 2 and self.__current_period.done(mdp, state, player_idx)


SCRIPT_PERIODS_CLASSES = {
    "pickup_object": Pickup_Object,
    "put_object": Put_Object,
    "pickup_onion_and_place_in_pot": Pickup_Onion_and_Place_in_Pot,
    "pickup_tomato_and_place_in_pot": Pickup_Tomato_and_Place_in_Pot,
    "pickup_onion_and_place_random": Pickup_Onion_and_Place_Random,
    "pickup_tomato_and_place_random": Pickup_Tomato_and_Place_Random,
    "pickup_soup": Pickup_Soup,
    "pickup_soup_and_deliver": Pickup_Soup_and_Deliver,
    "pickup_soup_and_place_random": Pickup_Soup_and_Place_Random,
    "pickup_dish_and_place_random": Pickup_Dish_and_Place_Random,
    "put_onion_everywhere": Put_Onion_Everywhere,
    "put_tomato_everywhere": Put_Tomato_Everywhere,
    "put_dish_everywhere": Put_Dish_Everywhere,
    "pickup_tomato_and_place_mix": Pickup_Tomato_and_Place_Mix,
    "pickup_ingredient_and_place_mix": Pickup_Ingredient_and_Place_Mix,
    "mixed_order": Mixed_Order,
}
