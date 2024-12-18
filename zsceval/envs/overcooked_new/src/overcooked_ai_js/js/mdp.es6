import _ from 'lodash';
import assert from 'assert';

const SHAPED_INFOS = [
    "put_onion_on_X",
    "put_tomato_on_X",
    "put_dish_on_X",
    "put_soup_on_X",
    "pickup_onion_from_X",
    "pickup_onion_from_O",
    "pickup_tomato_from_X",
    "pickup_tomato_from_T",
    "pickup_dish_from_X",
    "pickup_dish_from_D",
    "pickup_soup_from_X",
    "USEFUL_DISH_PICKUP",  
    "SOUP_PICKUP",
    "PLACEMENT_IN_POT", 
    "viable_placement",
    "optimal_placement",
    "catastrophic_placement",
    "useless_placement", 
    "potting_onion",
    "potting_tomato",
    "cook",
    "delivery",
    "deliver_size_two_order",
    "deliver_size_three_order",
    "deliver_useless_order",
    "STAY",
    "MOVEMENT",
    "IDLE_MOVEMENT",
    "IDLE_INTERACT",
    "place_onion_on_X",
    "place_tomato_on_X",
    "place_dish_on_X",
    "place_soup_on_X",
    "recieve_onion_via_X",
    "recieve_tomato_via_X",
    "recieve_dish_via_X",
    "recieve_soup_via_X",
    "onion_placed_on_X",
    "tomato_placed_on_X",
    "dish_placed_on_X",
    "soup_placed_on_X"
];
  
const BASE_REW_SHAPING_PARAMS = {
  "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
};
  
const EVENT_TYPES = [
    // Tomato events
    "tomato_pickup",
    "useful_tomato_pickup",
    "tomato_drop",
    "useful_tomato_drop",
    "potting_tomato",
    "placing_tomato",
    "recieve_tomato",
    "placed_tomatoes",
    // Onion events
    "onion_pickup",
    "useful_onion_pickup",
    "onion_drop",
    "useful_onion_drop",
    "potting_onion",
    "placing_onion",
    "recieve_onion",
    "placed_onions",
    // Dish events
    "dish_pickup",
    "useful_dish_pickup",
    "dish_drop",
    "useful_dish_drop",
    "placing_dish",
    "recieve_dish",
    "placed_dishes",
    // Soup events
    "soup_pickup",
    "soup_delivery",
    "soup_drop",
    "placing_soup",
    "recieve_soup",
    "placed_soups",
    // Potting events
    "optimal_onion_potting",
    "optimal_tomato_potting",
    "viable_onion_potting",
    "viable_tomato_potting",
    "catastrophic_onion_potting",
    "catastrophic_tomato_potting",
    "useless_onion_potting",
    "useless_tomato_potting",
];
  
const POTENTIAL_CONSTANTS = {
    "default": {
        "max_delivery_steps": 10,
        "max_pickup_steps": 10,
        "pot_onion_steps": 10,
        "pot_tomato_steps": 10,
    },
    "mdp_test_tomato": {
        "max_delivery_steps": 4,
        "max_pickup_steps": 4,
        "pot_onion_steps": 5,
        "pot_tomato_steps": 6,
    },
};
  
  // itertools.combinations_with_replacementのJavaScript実装
function* combinationsWithReplacement(arr, k) {
    if (k === 0) {
        yield [];
        return;
    }
    for (let i = 0; i < arr.length; i++) {
        let head = arr.slice(i, i + 1);
        for (let tail of combinationsWithReplacement(arr.slice(i), k - 1)) {
            yield head.concat(tail);
        }
    }
}
  
export class Recipe {

    constructor(ingredients) {

        /*
        this.MAX_NUM_INGREDIENTS = MAX_NUM_INGREDIENTS;
        this.TOMATO = TOMATO;
        this.ONION = ONION;
        this.ALL_INGREDIENTS = [ONION, TOMATO];

        this.ALL_RECIPES_CACHE = ALL_RECIPES_CACHE;
        this.STR_REP = STR_REP;

        this._computed = _computed;
        this._configured = _configured; 
        this.conf = _conf;
        */

        if (!Recipe._configured) {
            throw new Error("Recipe class must be configured before recipes can be created");
        }
        // Some basic argument verification
        if (!ingredients || !Array.isArray(ingredients) || ingredients.length === 0) {
            throw new Error("Invalid input recipe. Must be ingredients iterable with non-zero length");
        }
        ingredients.forEach(elem => {
            if (!Recipe.ALL_INGREDIENTS.includes(elem)) {
                throw new Error(`Invalid ingredient: ${elem}. Recipe can only contain ingredients ${Recipe.ALL_INGREDIENTS}`);
            }
        });
        if (ingredients.length > Recipe.MAX_NUM_INGREDIENTS) {
            throw new Error(`Recipe of length ${ingredients.length} is invalid. Recipe can contain at most ${Recipe.MAX_NUM_INGREDIENTS} ingredients`);
        }
        const key = this._hash(ingredients);
        if (Recipe.ALL_RECIPES_CACHE[key]) {
            return Recipe.ALL_RECIPES_CACHE[key];
        }
        Recipe.ALL_RECIPES_CACHE[key] = this;
        this._ingredients = ingredients;
    }

    _hash(ingredients) {
        return ingredients.sort().toString();
    }
  
    __getnewargs__() {
        return [this._ingredients];
    }
  
    __int__() {
        const num_tomatoes = this.ingredients.filter(ingredient => ingredient === Recipe.TOMATO).length;
        const num_onions = this.ingredients.filter(ingredient => ingredient === Recipe.ONION).length;
  
        const mixed_mask = Number(Boolean(num_tomatoes * num_onions));
        const mixed_shift = (Recipe.MAX_NUM_INGREDIENTS + 1) ** Recipe.ALL_INGREDIENTS.length;
        const encoding = num_onions + (Recipe.MAX_NUM_INGREDIENTS + 1) * num_tomatoes;
  
        return mixed_mask * encoding * mixed_shift + encoding;
    }
  
    __hash__() {
        return this._hash(this.ingredients);
    }
  
    __eq__(other) {
        return this.ingredients.toString() === other.ingredients.toString();
    }
  
    __ne__(other) {
        return !this.__eq__(other);
    }
  
    __lt__(other) {
        return this.__int__() < other.__int__();
    }
  
    __le__(other) {
        return this.__int__() <= other.__int__();
    }
  
    __gt__(other) {
        return this.__int__() > other.__int__();
    }
  
    __ge__(other) {
        return this.__int__() >= other.__int__();
    }
  
    __repr__() {
        return this.ingredients.__repr__();
    }
  
    // not sure
    __iter__(){
      return this.ingredients;
    }
  
    __copy__() {
        return new Recipe([...this.ingredients]);
    }
  
    __deepcopy__(memo) {
        const ingredients_cpy = JSON.parse(JSON.stringify(this.ingredients));
        return new Recipe(ingredients_cpy);
    }
  
    static _compute_all_recipes() {
        for (let i = 0; i < Recipe.MAX_NUM_INGREDIENTS; i++) {
            for (let ingredient_list of combinationsWithReplacement(Recipe.ALL_INGREDIENTS, i + 1)) {
                new Recipe(ingredient_list);
            }
        }
    }
  
    get ingredients() {
        return this._ingredients.slice().sort();
    }
  
    set ingredients(_) {
        throw new Error("Recipes are read-only. Do not modify instance attributes after creation");
    }
  
    get value() {
        if (Recipe._delivery_reward) {
            return Recipe._delivery_reward;
        }
        if (Recipe._value_mapping && Recipe._value_mapping.has(this.ingredients.toString())) {
            return Recipe._value_mapping.get(this.ingredients.toString());
        }
        if (Recipe._onion_value && Recipe._tomato_value) {
            let num_onions = this.ingredients.filter(ingredient => ingredient === Recipe.ONION).length;
            let num_tomatoes = this.ingredients.filter(ingredient => ingredient === Recipe.TOMATO).length;
            return Recipe._tomato_value * num_tomatoes + Recipe._onion_value * num_onions;
        }
        return 20;
    }
  
    get time() {
        if (Recipe._cook_time) {
            return Recipe._cook_time;
        }
        if (Recipe._time_mapping && Recipe._time_mapping.has(this.ingredients.toString())) {
            return Recipe._time_mapping.get(this.ingredients.toString());
        }
        if (Recipe._onion_time && Recipe._tomato_time) {
            let num_onions = this.ingredients.filter(ingredient => ingredient === Recipe.ONION).length;
            let num_tomatoes = this.ingredients.filter(ingredient => ingredient === Recipe.TOMATO).length;
            return Recipe._onion_time * num_onions + Recipe._tomato_time * num_tomatoes;
        }
        return 20;
    }
  
    to_dict() {
        return { "ingredients": this.ingredients };
    }
  
    neighbors() {
        let neighbors = [];
        if (this.ingredients.length === Recipe.MAX_NUM_INGREDIENTS) {
            return neighbors;
        }
        for (let ingredient of Recipe.ALL_INGREDIENTS) {
            let new_ingredients = [...this.ingredients, ingredient];
            let new_recipe = new Recipe(new_ingredients);
            neighbors.push(new_recipe);
        }
        return neighbors;
    }
  
    static get ALL_RECIPES() {
        if (!Recipe._computed) {
            Recipe._compute_all_recipes();
            Recipe._computed = true;
        }
        return new Set(Object.values(Recipe.ALL_RECIPES_CACHE));
    }
  
    static get configuration() {
        if (!Recipe._configured) {
            throw new Error("Recipe class not yet configured");
        }
        return Recipe._conf;
    }
  
    static configure(conf) {
        Recipe._conf = conf;
        Recipe._configured = true;
        Recipe._computed = false;
        Recipe.MAX_NUM_INGREDIENTS = conf.max_num_ingredients || 3;
  
        Recipe._cook_time = null;
        Recipe._delivery_reward = null;
        Recipe._value_mapping = null;
        Recipe._time_mapping = null;
        Recipe._onion_value = null;
        Recipe._onion_time = null;
        Recipe._tomato_value = null;
        Recipe._tomato_time = null;
  
        // Basic checks for validity
  
        // Mutual Exclusion
        if ((conf.tomato_time && !conf.onion_time) || (conf.onion_time && !conf.tomato_time)) {
            throw new Error("Must specify both 'onion_time' and 'tomato_time'");
        }
        if ((conf.tomato_value && !conf.onion_value) || (conf.onion_value && !conf.tomato_value)) {
            throw new Error("Must specify both 'onion_value' and 'tomato_value'");
        }
        if (conf.tomato_value && conf.delivery_reward) {
            throw new Error("'delivery_reward' incompatible with '<ingredient>_value'");
        }
        if (conf.tomato_value && conf.recipe_values) {
            throw new Error("'recipe_values' incompatible with '<ingredient>_value'");
        }
        if (conf.recipe_values && conf.delivery_reward) {
            throw new Error("'delivery_reward' incompatible with 'recipe_values'");
        }
        if (conf.tomato_time && conf.cook_time) {
            throw new Error("'cook_time' incompatible with '<ingredient>_time'");
        }
        if (conf.tomato_time && conf.recipe_times) {
            throw new Error("'recipe_times' incompatible with '<ingredient>_time'");
        }
        if (conf.recipe_times && conf.cook_time) {
            throw new Error("'delivery_reward' incompatible with 'recipe_times'");
  
          // recipe_ lists and orders compatibility
          if (conf.recipe_values) {
              if (!conf.all_orders || !conf.all_orders.length) {
                  throw new Error("Must specify 'all_orders' if 'recipe_values' specified");
              }
              if (conf.all_orders.length !== conf.recipe_values.length) {
                  throw new Error("Number of recipes in 'all_orders' must be the same as number in 'recipe_values'");
              }
          }
          if (conf.recipe_times) {
              if (!conf.all_orders || !conf.all_orders.length) {
                  throw new Error("Must specify 'all_orders' if 'recipe_times' specified");
              }
              if (conf.all_orders.length !== conf.recipe_times.length) {
                  throw new Error("Number of recipes in 'all_orders' must be the same as number in 'recipe_times'");
              }
          }
        } 
        if (conf.cook_time) {
            Recipe._cook_time = conf.cook_time;
        }
  
        if (conf.delivery_reward) {
            Recipe._delivery_reward = conf.delivery_reward;
        }
  
        if (conf.recipe_values) {
            Recipe._value_mapping = new Map(
                conf.all_orders.map((recipe, index) => [Recipe.from_dict(recipe).ingredients.toString(), conf.recipe_values[index]])
            );
        }
  
        if (conf.recipe_times) {
            Recipe._time_mapping = new Map(
                conf.all_orders.map((recipe, index) => [Recipe.from_dict(recipe).ingredients.toString(), conf.recipe_times[index]])
            );
        }
  
        if (conf.tomato_time) {
            Recipe._tomato_time = conf.tomato_time;
        }
  
        if (conf.onion_time) {
            Recipe._onion_time = conf.onion_time;
        }
  
        if (conf.tomato_value) {
            Recipe._tomato_value = conf.tomato_value;
        }
  
        if (conf.onion_value) {
            Recipe._onion_value = conf.onion_value;
        }
    }
  
    static generate_random_recipes(n = 1, minSize = 2, maxSize = 3, ingredients = null, recipes = null, unique = true) {
        if (!recips) {
            recipes = Array.from(Recipe.ALL_RECIPES);
        }
  
        ingredients = new Set(ingredients || Recipe.ALL_INGREDIENTS);
        const choiceReplace = !unique;
  
        if (!(1 <= minSize && minSize <= maxSize && maxSize <= Recipe.MAX_NUM_INGREDIENTS)) {
            throw new Error("Invalid size range");
        }
  
        if (!Array.from(ingredients).every(ingredient => Recipe.ALL_INGREDIENTS.includes(ingredient))) {
            throw new Error("Invalid ingredients");
        }
  
        const validSize = r => minSize <= r.ingredients.length && r.ingredients.length <= maxSize;
        const validIngredients = r => r.ingredients.every(i => ingredients.has(i));
  
        const relevantRecipes = recipes.filter(r => validSize(r) && validIngredients(r));
        if (!choiceReplace && n > relevantRecipes.length) {
            throw new Error("Not enough unique recipes available");
        }
  
        return Array.from({ length: n }, () => relevantRecipes[Math.floor(Math.random() * relevantRecipes.length)]);
    }
  
    static from_dict(objDict) {
        return new Recipe(objDict.ingredients);
    }
}

Recipe.MAX_NUM_INGREDIENTS = 3;
  
Recipe.TOMATO = "tomato";
Recipe.ONION = "onion";
Recipe.ALL_INGREDIENTS = [Recipe.ONION, Recipe.TOMATO];

Recipe.ALL_RECIPES_CACHE = {};
Recipe.STR_REP = { "tomato": "†", "onion": "ø" };

Recipe._computed = false;
Recipe._configured = false;
Recipe._conf = {};


export class Direction {
    /**
     * The four possible directions a player can be facing.
     */

    static get_adjacent_directions(direction) {
        /** Returns the directions within 90 degrees of the given direction.
         *
         * direction: One of the Directions, except not Direction.STAY.
         */
        if ([Direction.NORTH, Direction.SOUTH].includes(direction)) {
            return [Direction.EAST, Direction.WEST];
        } else if ([Direction.EAST, Direction.WEST].includes(direction)) {
            return [Direction.NORTH, Direction.SOUTH];
        }
        throw new Error(`Invalid direction: ${direction}`);
    }
}

Direction.NORTH = [0, -1];
Direction.SOUTH = [0, 1];
Direction.EAST = [1, 0];
Direction.WEST = [-1, 0];
Direction.CARDINAL = [
    Direction.NORTH, Direction.SOUTH,
    Direction.EAST, Direction.WEST
];
Direction.INDEX_TO_DIRECTION = [
    Direction.NORTH, Direction.SOUTH,
    Direction.EAST, Direction.WEST
];
Direction.DIRECTION_TO_INDEX =
    _.fromPairs(Direction.INDEX_TO_DIRECTION.map((d, i) => {
        return [d, i]
    }));
    
Direction.ALL_DIRECTIONS = Direction.INDEX_TO_DIRECTION;
Direction.OPPOSITE_DIRECTIONS = _.fromPairs([
    [Direction.NORTH, Direction.SOUTH],
    [Direction.SOUTH, Direction.NORTH],
    [Direction.EAST, Direction.WEST],
    [Direction.WEST, Direction.EAST]
]);
Direction.DIRECTION_TO_NAME = {
    '0,-1': 'NORTH',
    '0,1': 'SOUTH',
    '1,0': 'EAST',
    '-1,0': 'WEST'
}


export class Action {
    /**
     * The six actions available in the OvercookedGridworld.
     *
     * Includes definitions of the actions as well as utility functions for
     * manipulating them or applying them.
     */

    
    static move_in_direction(point, direction) {
        /**
         * Takes a step in the given direction and returns the new point.
         *
         * point: Tuple (x, y) representing a point in the x-y plane.
         * direction: One of the Directions.
         */
        console.assert(Action.MOTION_ACTIONS.includes(direction));
        const [x, y] = point;
        const [dx, dy] = direction;
        return [x + dx, y + dy];
    }

    static determine_action_for_change_in_pos(old_pos, new_pos) {
        /** Determines an action that will enable intended transition */
        if (old_pos === new_pos) {
            return Action.STAY;
        }
        const [new_x, new_y] = new_pos;
        const [old_x, old_y] = old_pos;
        const direction = [new_x - old_x, new_y - old_y];
        console.assert(Direction.ALL_DIRECTIONS.includes(direction));
        return direction;
    }

    static sample(action_probs) {
        return np.random.choice(Action.ALL_ACTIONS, { p: action_probs });
    }

    static argmax(action_probs) {
        const action_idx = np.argmax(action_probs);
        return Action.INDEX_TO_ACTION[action_idx];
    }

    static remove_indices_and_renormalize(probs, indices, eps = 0.0) {
        probs = JSON.parse(JSON.stringify(probs));
        if (Array.isArray(probs[0])) {
            probs = np.array(probs);
            indices.forEach((row, row_idx) => {
                indices.forEach(idx => {
                    probs[row_idx][idx] = eps;
                });
            });
            const norm_probs = probs.T / np.sum(probs, 1);
            return norm_probs.T;
        } else {
            indices.forEach(idx => {
                probs[idx] = eps;
            });
            return probs / probs.reduce((a, b) => a + b, 0);
        }
    }

    static to_char(action) {
        console.assert(Action.ALL_ACTIONS.includes(action));
        return Action.ACTION_TO_CHAR[action];
    }

    static joint_action_to_char(joint_action) {
        console.assert(joint_action.every(a => Action.ALL_ACTIONS.includes(a)));
        return joint_action.map(a => Action.to_char(a));
    }

    static uniform_probs_over_actions() {
        const num_acts = Action.ALL_ACTIONS.length;
        return Array(num_acts).fill(1 / num_acts);
    }
}

Action.STAY = [0, 0];
Action.INTERACT = "interact";

Action.INDEX_TO_ACTION = _.clone(Direction.INDEX_TO_DIRECTION);
Action.INDEX_TO_ACTION.push(Action.STAY);
Action.INDEX_TO_ACTION.push(Action.INTERACT);

Action.ALL_ACTIONS = Action.INDEX_TO_ACTION;

Action.ACTION_TO_INDEX = _.fromPairs(Action.INDEX_TO_ACTION.map((a, i) => [a, i]));

Action.MOTION_ACTIONS = Direction.ALL_DIRECTIONS;
Action.MOTION_ACTIONS.push(Action.STAY);

function product(...arrays) {
    return arrays.reduce((acc, curr) => {
        return acc.flatMap(d => curr.map(e => [d, e].flat()));
    }, [[]]);
}
Action.INDEX_TO_ACTION_INDEX_PAIRS = Array.from(
    product(Array(Action.INDEX_TO_ACTION.length).keys(), Array(Action.INDEX_TO_ACTION.length).keys()));

Action.ACTION_TO_CHAR = {
    [Direction.NORTH]: "↑",
    [Direction.SOUTH]: "↓",
    [Direction.EAST]: "→",
    [Direction.WEST]: "←",
    [Action.STAY]: "stay",
    [Action.INTERACT]: Action.INTERACT,
};
Action.NUM_ACTIONS = Action.ALL_ACTIONS.length;


export class ObjectState {
    /**
   * State of an object in OvercookedGridworld.
   * @param {string} name - The name of the object
   * @param {Array} position - Tuple for the current location of the object.
   *  @param {Array} last_owner
   */
    constructor(name, position, last_owner=null, ...kwargs) {
        this.name = name;
        this._position = position
        this._last_owner = last_owner
    }

    get position() {
        return this._position;
    }

    set position(new_pos) {
        this._position = new_pos;
    }

    get last_owner() {
        return this._last_owner;
    }

    set last_owner(last_owner) {
        this._last_owner = last_owner;
    }

    is_valid() {
        return ["onion", "tomato", "dish"].includes(this.name);
    }

    deepcopy() {
        return new ObjectState(this.name, this.position, self.last_owner);
    }

    equals(other) {
        return other instanceof ObjectState && this.name === other.name && this.position === other.position && this.last_owner == other.last_owner;
    }

    hashCode() {
        return hash([this.name, this.position]);
    }

    toString() {
        return `${this.name}@${this.position}`;
    }

    __repr__() {
        return `${this.name}@${this.position}, last_owner:${this.last_owner}`
    }

    to_dict() {
        return { name: this.name, position: this.position, last_owner : this.last_owner };
    }

    static from_dict(obj_dict) {
        obj_dict = { ...obj_dict };
        return new ObjectState(obj_dict.name, obj_dict.position);
    }
}

export class SoupState extends ObjectState {
    /**
     * Represents a soup object. An object becomes a soup the instant it is placed in a pot. The
     * soup's recipe is a list of ingredient names used to create it. A soup's recipe is undetermined
     * until it has begun cooking.
     * @param {Array} position - (x, y) coordinates in the grid
     * @param {Array} ingredients - Objects that have been used to cook this soup. Determines @property recipe
     * @param {number} cooking_tick - How long the soup has been cooking for. -1 means cooking hasn't started yet
     * @param {number} cook_time - How long soup needs to be cooked, used only mostly for getting soup from dict with supplied cook_time, if None self.recipe.time is used
     */
    constructor(position, last_owner=None,ingredients = [], cooking_tick = -1, cook_time = null, ...kwargs) {
        super("soup", position, last_owner);
        this._ingredients = ingredients;
        this._cooking_tick = cooking_tick;
        this._recipe = null;
        this._cook_time = cook_time;
    }
  
    equals(other) {
        return (
            other instanceof SoupState &&
            this.name === other.name &&
            this.position === other.position &&
            this._cooking_tick === other._cooking_tick &&
            this._ingredients.every((this_i, index) => this_i.equals(other._ingredients[index]))
        );
    }
  
    hashCode() {
        const ingredient_hash = hash(this._ingredients.map(i => i.hashCode()));
        const supercls_hash = super.hashCode();
        return hash([supercls_hash, this._cooking_tick, ingredient_hash]);
    }
  
    toString() {
        const supercls_str = super.toString();
        const ingredients_str = this._ingredients.toString();
        return `${supercls_str}\nIngredients:\t${ingredients_str}\nCooking Tick:\t${this._cooking_tick}`;
    }
  
    toString() {
        let res = "{";
        for (let ingredient of this.ingredients.sort()) {
            res += Recipe.STR_REP[ingredient];
        }
        if (this.is_cooking) {
            res += this._cooking_tick.toString();
        } else if (this.is_ready) {
            res += "✓";
        }
        return res;
    }

    get position() {
        return this._position;
    }
   
    set position(new_pos) {
        this._position = new_pos;

        if(this.ingredients.length === 0) return;

        for (let ingredient of this._ingredients) {
            ingredient.position = new_pos;
        }
    }
  
    get ingredients() {
        return this._ingredients.map(ingredient => ingredient.name);
    }
  
    get is_cooking() {
        return !this.is_idle && !this.is_ready;
    }
  
    get recipe() {
        if (this.is_idle) {
            throw new Error("Recipe is not determined until soup begins cooking");
        }
        if (!this._recipe) {
            this._recipe = new Recipe(this.ingredients);
        }
        return this._recipe;
    }
  
    get value() {
        return this.recipe.value;
    }
  
    get cook_time() {
        // used mostly when cook time is supplied by state dict
        return this._cook_time !== null ? this._cook_time : this.recipe.time;
    }
  
    get cook_time_remaining() {
        return Math.max(0, this.cook_time - this._cooking_tick);
    }
  
    get is_ready() {
        return !this.is_idle && this._cooking_tick >= this.cook_time;
    }
  
    get is_idle() {
        return this._cooking_tick < 0;
    }
  
    get is_full() {
        return !this.is_idle || this.ingredients.length === Recipe.MAX_NUM_INGREDIENTS;
    }
  
    is_valid() {
        if (!this._ingredients.every(ingredient => ingredient.position === this.position)) {
            return false;
        }
        if (this.ingredients.length > Recipe.MAX_NUM_INGREDIENTS) {
            return false;
        }
        return true;
    }
  
    auto_finish() {
        if (this.ingredients.length === 0) {
            throw new Error("Cannot finish soup with no ingredients");
        }
        this._cooking_tick = 0;
        this._cooking_tick = this.cook_time;
    }
  
    add_ingredient(ingredient) {
        if (!Recipe.ALL_INGREDIENTS.includes(ingredient.name)) {
            throw new Error("Invalid ingredient");
        }
        if (this.is_full) {
            throw new Error("Reached maximum number of ingredients in recipe");
        }
        ingredient.position = this.position;
        this._ingredients.push(ingredient);
    }
  
    add_ingredient_from_str(ingredient_str) {
        const ingredient_obj = new ObjectState(ingredient_str, this.position);
        this.add_ingredient(ingredient_obj);
    }
  
    pop_ingredient() {
        if (!this.is_idle) {
            throw new Error("Cannot remove an ingredient from this soup at this time");
        }
        if (this._ingredients.length === 0) {
            throw new Error("No ingredient to remove");
        }
        return this._ingredients.pop();
    }
  
    begin_cooking() {
        if (!this.is_idle) {
            throw new Error("Cannot begin cooking this soup at this time");
        }
        if (this.ingredients.length === 0) {
            throw new Error("Must add at least one ingredient to soup before you can begin cooking");
        }
        this._cooking_tick = 0;
    }
  
    cook() {
        if (this.is_idle) {
            throw new Error("Must begin cooking before advancing cook tick");
        }
        if (this.is_ready) {
            throw new Error("Cannot cook a soup that is already done");
        }
        this._cooking_tick += 1;
    }
  
    deepcopy() {
        return new SoupState(
            this.position,
            this.last_owner,
            this._ingredients.map(ingredient => ingredient.deepcopy()),
            this._cooking_tick
        );
    }
  
    to_dict() {
        const info_dict = super.to_dict();
        const ingredients_dict = this._ingredients.map(ingredient => ingredient.to_dict());
        info_dict["_ingredients"] = ingredients_dict;
        info_dict["cooking_tick"] = this._cooking_tick;
        info_dict["is_cooking"] = this.is_cooking;
        info_dict["is_ready"] = this.is_ready;
        info_dict["is_idle"] = this.is_idle;
        info_dict["cook_time"] = this.is_idle ? -1 : this.cook_time;
  
        // This is for backwards compatibility w/ overcooked-demo
        // Should be removed once overcooked-demo is updated to use 'cooking_tick' instead of '_cooking_tick'
        info_dict["_cooking_tick"] = this._cooking_tick;
        return info_dict;
    }
  
    static from_dict(obj_dict) {
        obj_dict = { ...obj_dict };
        if (obj_dict.name !== "soup") {
            return super.from_dict(obj_dict);
        }
  
        if ("state" in obj_dict) {
            const [ingredient, num_ingredient, time] = obj_dict.state;
            const cooking_tick = time === 0 ? -1 : time;
            const finished = time >= 20;
            if (ingredient === Recipe.TOMATO) {
                return SoupState.get_soup(
                    obj_dict.position,
                    { num_tomatoes: num_ingredient, cooking_tick, finished }
                );
            } else {
                return SoupState.get_soup(
                    obj_dict.position,
                    { num_onions: num_ingredient, cooking_tick, finished }
                );
            }
        }
  
        const ingredients_objs = obj_dict._ingredients.map(ing_dict => ObjectState.from_dict(ing_dict));
        obj_dict.ingredients = ingredients_objs;
        return new SoupState(obj_dict.position, obj_dict.ingredients, obj_dict.cooking_tick, obj_dict.cook_time);
    }
  
    static get_soup(position, { num_onions = 1, num_tomatoes = 0, cooking_tick = -1, finished = false, ...kwargs } = {}) {
        if (num_onions < 0 || num_tomatoes < 0) {
            throw new Error("Number of active ingredients must be positive");
        }
        if (num_onions + num_tomatoes > Recipe.MAX_NUM_INGREDIENTS) {
            throw new Error("Too many ingredients specified for this soup");
        }
        if (cooking_tick >= 0 && num_tomatoes + num_onions === 0) {
            throw new Error("_cooking_tick must be -1 for empty soup");
        }
        if (finished && num_tomatoes + num_onions === 0) {
            throw new Error("Empty soup cannot be finished");
        }
        const onions = Array.from({ length: num_onions }, () => new ObjectState(Recipe.ONION, position));
        const tomatoes = Array.from({ length: num_tomatoes }, () => new ObjectState(Recipe.TOMATO, position));
        const ingredients = onions.concat(tomatoes);
        const soup = new SoupState(position, ingredients, cooking_tick);
        if (finished) {
            soup.auto_finish();
        }
        return soup;
    }
}

export class PlayerState {
    /**
     * State of a player in OvercookedGridworld.
     * @param {Array} position - (x, y) tuple representing the player's location.
     * @param {Array} orientation - Direction.NORTH/SOUTH/EAST/WEST representing orientation.
     * @param {ObjectState} held_object - ObjectState representing the object held by the player, or null if there is no such object.
     */
    constructor(position, orientation, held_object = null) {
        this.position = position;
        this.orientation = orientation;
        this.held_object = held_object;
        assert(Direction.ALL_DIRECTIONS.includes(this.orientation));
        if (this.held_object !== null) {
            assert(this.held_object instanceof ObjectState);
            assert(this.held_object.position === this.position);
        }
    }
  
    get pos_and_or() {
        return [this.position, this.orientation];
    }
  
    has_object() {
        return this.held_object !== null;
    }
  
    get_object() {
        assert(this.has_object());
        return this.held_object;
    }
  
    set_object(obj) {
        assert(!this.has_object());
        obj.position = this.position;
        this.held_object = obj;
    }
  
    remove_object() {
        assert(this.has_object());
        const obj = this.held_object;
        this.held_object = null;
        return obj;
    }
  
    update_pos_and_or(new_position, new_orientation) {
        this.position = new_position;
        this.orientation = new_orientation;
        if (this.has_object()) {
            this.get_object().position = new_position;
        }
    }
  
    deepcopy() {
        const new_obj = this.held_object === null ? null : this.held_object.deepcopy();
        return new PlayerState(this.position, this.orientation, new_obj);
    }
  
    equals(other) {
        return (
            other instanceof PlayerState &&
            this.position === other.position &&
            this.orientation === other.orientation &&
            this.held_object === other.held_object
        );
    }
  
    hashCode() {
        return hash([this.position, this.orientation, this.held_object]);
    }
  
    toString() {
        return `${this.position} facing ${this.orientation} holding ${this.held_object}`;
    }
  
    to_dict() {
        return {
            position: this.position,
            orientation: this.orientation,
            held_object: this.held_object ? this.held_object.to_dict() : null,
        };
    }
  
    static from_dict(player_dict) {
        player_dict = { ...player_dict };
        const held_obj = player_dict.held_object;
        if (held_obj !== null) {
            player_dict.held_object = SoupState.from_dict(held_obj);
        }
        return new PlayerState(player_dict.position, player_dict.orientation, player_dict.held_object);
    }
}

export function dictToState(state_dict) {
    let object_dict = {}
    if (state_dict['objects'].length > 0) {
        state_dict['objects'].forEach(function (item, index) {
            object_dict[item['position']] = dictToObjectState(item)
        })
    }
    return new OvercookedState({
        players: [dictToPlayerState(state_dict['players'][0]), dictToPlayerState(state_dict['players'][1])],
        objects: object_dict,
        order_list: state_dict['order_list']
    })
}

export function dictToPlayerState(player_dict) {
    if (player_dict['held_object'] == null) {
        player_dict['held_object'] = undefined
    }
    else {
        player_dict['held_object'] = dictToObjectState(player_dict['held_object'])
    }
    return new PlayerState({
        position: player_dict['position'],
        orientation: player_dict['orientation'],
        held_object: player_dict['held_object']
    })
}

export function dictToObjectState(object_dict) {
    if (object_dict['state'] == null) {
        object_dict['state'] = undefined;
    }
    return new ObjectState(
        {
            name: object_dict['name'],
            position: object_dict['position'],
            state: object_dict['state']
        })
}


export function lookupActions(actions_arr) {
    let actions = [];
    actions_arr.forEach(function (item, index) {
        if (item == "interact") {
            item = Action.INTERACT;
        }
        if (arraysEqual(Direction.STAY, item) || item == "stay") {
            item = Direction.STAY;
        }
        actions.push(item);
    }
    )
    return actions;

}

function arraysEqual(a, b) {
    // Stolen from https://stackoverflow.com/questions/3115982/how-to-check-if-two-arrays-are-equal-with-javascript
    if (a === b) return true;
    if (a == null || b == null) return false;
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; ++i) {
        if (a[i] !== b[i]) return false;
    }
    return true;

}

export class OvercookedState {
    /**
     * A state in OvercookedGridworld.
     * @param {Array} players - Currently active PlayerStates (index corresponds to number)
     * @param {Object} objects - Dictionary mapping positions (x, y) to ObjectStates.
     * @param {Array} bonus_orders - Current orders worth a bonus
     * @param {Array} all_orders - Current orders allowed at all
     * @param {number} timestep - The current timestep of the state
     */
    constructor(players, objects, bonus_orders = [], all_orders = [], timestep = 0, ...kwargs) {
        this.players = players;
        this.objects = objects;
        if(bonus_orders.map) this._bonus_orders = bonus_orders.map(order => Recipe.from_dict(order));
        else this._bonus_orders = [];
        this._all_orders = all_orders.map(order => Recipe.from_dict(order));
        this.timestep = timestep;

        
        for (let [pos, obj] of Object.entries(objects)) {
            let pos_array = [];
            for(let e of pos.split(',')){
                pos_array.push(Number(e));
            }
            assert(obj.position[0]===pos_array[0] && obj.position[1]===pos_array[1]);
        }

        assert(new Set(this.bonus_orders).size === this.bonus_orders.length, "Bonus orders must not have duplicates");
        assert(new Set(this.all_orders).size === this.all_orders.length, "All orders must not have duplicates");
        assert(new Set(this.bonus_orders).isSubsetOf(new Set(this.all_orders)), "Bonus orders must be a subset of all orders");
    }

    get player_positions() {
        return this.players.map(player => player.position);
    }

    get player_orientations() {
        return this.players.map(player => player.orientation);
    }

    get players_pos_and_or() {
        return this.player_positions.map((pos, index) => [pos, this.player_orientations[index]]);
    }

    get unowned_objects_by_type() {
        const objects_by_type = {};
        for (let [pos, obj] of Object.entries(this.objects)) {
            if (!objects_by_type[obj.name]) {
                objects_by_type[obj.name] = [];
            }
            objects_by_type[obj.name].push(obj);
        }
        return objects_by_type;
    }

    get player_objects_by_type() {
        const player_objects = {};
        for (let player of this.players) {
            if (player.has_object()) {
                const player_obj = player.get_object();
                if (!player_objects[player_obj.name]) {
                    player_objects[player_obj.name] = [];
                }
                player_objects[player_obj.name].push(player_obj);
            }
        }
        return player_objects;
    }

    get all_objects_by_type() {
        const all_objs_by_type = { ...this.unowned_objects_by_type };
        for (let [obj_type, player_objs] of Object.entries(this.player_objects_by_type)) {
            if (!all_objs_by_type[obj_type]) {
                all_objs_by_type[obj_type] = [];
            }
            all_objs_by_type[obj_type].push(...player_objs);
        }
        return all_objs_by_type;
    }

    get all_objects_list() {
        return Object.values(this.all_objects_by_type).flat();
    }

    get all_orders() {
        return this._all_orders.length ? this._all_orders.sort() : Array.from(Recipe.ALL_RECIPES).sort();
    }

    get bonus_orders() {
        return this._bonus_orders.sort();
    }

    has_object(pos) {
        return pos in this.objects;
    }

    get_object(pos) {
        assert(this.has_object(pos));
        return this.objects[pos];
    }

    add_object(obj, pos = null) {
        if (pos === null) {
            pos = obj.position;
        }
        assert(!this.has_object(pos));
        obj.position = pos;
        this.objects[pos] = obj;
    }

    remove_object(pos) {
        assert(this.has_object(pos));
        const obj = this.objects[pos];
        delete this.objects[pos];
        return obj;
    }
    
    count_onions_on_X(grid) { return Object.values(this.objects).reduce((sum, obj) => sum + (obj.name === "onion" && grid[obj.position[1]][obj.position[0]] === "X" ? 1 : 0), 0); } 
    count_tomatoes_on_X(grid) { return Object.values(this.objects).reduce((sum, obj) => sum + (obj.name === "tomato" && grid[obj.position[1]][obj.position[0]] === "X" ? 1 : 0), 0); } 
    count_dishes_on_X(grid) { return Object.values(this.objects).reduce((sum, obj) => sum + (obj.name === "dish" && grid[obj.position[1]][obj.position[0]] === "X" ? 1 : 0), 0); }
    count_soups_on_X(grid) { return Object.values(this.objects).reduce((sum, obj) => sum + (obj.name === "soup" && grid[obj.position[1]][obj.position[0]] === "X" ? 1 : 0), 0); }

    static from_players_pos_and_or(players_pos_and_or, bonus_orders = [], all_orders = []) {
        return new OvercookedState(
            players_pos_and_or.map(player_pos_and_or => new PlayerState(...player_pos_and_or)),
            {},
            bonus_orders,
            all_orders
        );
    }

    static from_player_positions(player_positions, bonus_orders = [], all_orders = []) {
        const dummy_pos_and_or = player_positions.map(pos => [pos, Direction.NORTH]);
        return OvercookedState.from_players_pos_and_or(dummy_pos_and_or, bonus_orders, all_orders);
    }

    deepcopy() {
        return new OvercookedState(
            this.players.map(player => player.deepcopy()),
            Object.fromEntries(Object.entries(this.objects).map(([pos, obj]) => [pos, obj.deepcopy()])),
            this.bonus_orders.map(order => order.to_dict()),
            this.all_orders.map(order => order.to_dict()),
            this.timestep
        );
    }

    time_independent_equal(other) {
        const order_lists_equal = this.all_orders === other.all_orders && this.bonus_orders === other.bonus_orders;
        return (
            other instanceof OvercookedState &&
            this.players === other.players &&
            new Set(Object.entries(this.objects)).equals(new Set(Object.entries(other.objects))) &&
            order_lists_equal
        );
    }

    equals(other) {
        return this.time_independent_equal(other) && this.timestep === other.timestep;
    }

    hashCode() {
        const order_list_hash = hash(tuple(this.bonus_orders)) + hash(tuple(this.all_orders));
        return hash([this.players, Object.values(this.objects), order_list_hash]);
    }

    toString() {
        return `Players: ${this.players}, Objects: ${Object.values(this.objects)}, Bonus orders: ${this.bonus_orders}, All orders: ${this.all_orders}, Timestep: ${this.timestep}`;
    }

    to_dict() {
        return {
            players: this.players.map(p => p.to_dict()),
            objects: Object.values(this.objects).map(obj => obj.to_dict()),
            bonus_orders: this.bonus_orders.map(order => order.to_dict()),
            all_orders: this.all_orders.map(order => order.to_dict()),
            timestep: this.timestep
        };
    }

    static from_dict(state_dict) {
        state_dict = { ...state_dict };
        state_dict.players = state_dict.players.map(p => PlayerState.from_dict(p));
        const object_list = state_dict.objects.map(o => SoupState.from_dict(o));
        state_dict.objects = Object.fromEntries(object_list.map(ob => [ob.position, ob]));
        return new OvercookedState(state_dict);
    }
}

function dictToState(state_dict) {
    let object_dict = {};
    if (state_dict['objects'].length > 0) {
        state_dict['objects'].forEach(function (item, index) {
            object_dict[item['position']] = dictToObjectState(item);
        });
    }
    return new OvercookedState({
        players: [dictToPlayerState(state_dict['players'][0]), dictToPlayerState(state_dict['players'][1])],
        objects: object_dict,
        order_list: state_dict['order_list']
    });
}

function dictToPlayerState(player_dict) {
    if (player_dict['held_object'] == null) {
        player_dict['held_object'] = undefined;
    } else {
        player_dict['held_object'] = dictToObjectState(player_dict['held_object']);
    }
    return new PlayerState({
        position: player_dict['position'],
        orientation: player_dict['orientation'],
        held_object: player_dict['held_object']
    });
}

function dictToObjectState(object_dict) {
    if (object_dict['state'] == null) {
        object_dict['state'] = undefined;
    }
    return new ObjectState({
        name: object_dict['name'],
        position: object_dict['position'],
        state: object_dict['state']
    });
}

function lookupActions(actions_arr) {
    let actions = [];
    actions_arr.forEach(function (item, index) {
        if (item == "interact") {
            item = Action.INTERACT;
        }
        if (arraysEqual(Direction.STAY, item) || item == "stay") {
            item = Direction.STAY;
        }
        actions.push(item);
    });
    return actions;
}

function arraysEqual(a, b) {
    // Stolen from https://stackoverflow.com/questions/3115982/how-to-check-if-two-arrays-are-equal-with-javascript
    if (a === b) return true;
    if (a == null || b == null) return false;
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; ++i) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

/*

    Main MDP Class

 */

export class OvercookedGridworld {
    /**
     * @param {Array} terrain - A matrix of strings that encode the MDP layout
     * @param {Array} start_player_positions - Tuple of positions for both players' starting positions
     * @param {Array} start_bonus_orders - List of recipes dicts that are worth a bonus
     * @param {Object} rew_shaping_params - Reward given for completion of specific subgoals
     * @param {string} layout_name - String identifier of the layout
     * @param {Array} start_all_orders - List of all available order dicts the players can make, defaults to all possible recipes if empty list provided
     * @param {number} max_num_items_for_soup - Maximum number of ingredients that can be placed in a soup
     * @param {number} order_bonus - Multiplicative factor for serving a bonus recipe
     * @param {Object} start_state - Default start state returned by get_standard_start_state
     * @param {boolean} old_dynamics - Determines whether to start cooking automatically once 3 items are in the pot
     * @param {Object} params_to_overwrite - Parameters to overwrite the default mdp configuration 
    */
    constructor({
        terrain,
        start_player_positions,
        start_bonus_orders = [],
        rew_shaping_params = null,
        layout_name = "unnamed_layout",
        start_all_orders = [],
        max_num_items_for_soup = 3,
        order_bonus = 2,
        start_state = null,
        old_dynamics = false,
        ...kwargs
    }) {
        this._configure_recipes(start_all_orders, max_num_items_for_soup, kwargs);
        this.start_all_orders = start_all_orders.length === 0 ? Array.from(Recipe.ALL_RECIPES).map(r => r.to_dict()) : start_all_orders;
        if (old_dynamics) {
            assert(
                this.start_all_orders.every(order => order.ingredients.length === 3),
                "Only accept orders with 3 items when using the old_dynamics"
            );
        }
        this.height = terrain.length;
        this.width = terrain[0].length;
        this.shape = [this.width, this.height];
        this.terrain_mtx = terrain;
        this.terrain_pos_dict = this._get_terrain_type_pos_dict();
        this.start_player_positions = start_player_positions;
        this.num_players = start_player_positions.length;
        this.start_bonus_orders = start_bonus_orders;
        this.reward_shaping_params = rew_shaping_params === null ? BASE_REW_SHAPING_PARAMS : rew_shaping_params;
        this.layout_name = layout_name;
        this.order_bonus = order_bonus;
        this.start_state = start_state;
        this.max_num_items_for_soup = max_num_items_for_soup;
        this._opt_recipe_discount_cache = {};
        this._opt_recipe_cache = {};
        this._prev_potential_params = {};
        this.old_dynamics = old_dynamics;
    }

    static from_grid(layout_grid, base_layout_params = {}, params_to_overwrite = {}, debug = false) {
        const mdp_config = { ...base_layout_params };

        layout_grid = layout_grid.map(row => [...row]);
        OvercookedGridworld._assert_valid_grid(layout_grid);

        if (!mdp_config.layout_name) {
            const layout_name = layout_grid.map(line => line.join("")).join("|");
            mdp_config.layout_name = layout_name;
        }

        const player_positions = Array(9).fill(null);
        layout_grid.forEach((row, y) => {
            row.forEach((c, x) => {
                if (["1", "2", "3", "4", "5", "6", "7", "8", "9"].includes(c)) {
                    layout_grid[y][x] = " ";
                    assert(player_positions[parseInt(c) - 1] === null, "Duplicate player in grid");
                    player_positions[parseInt(c) - 1] = [x, y];
                }
            });
        });

        const num_players = player_positions.filter(x => x !== null).length;
        const final_player_positions = player_positions.slice(0, num_players);

        // After removing player positions from grid we have a terrain mtx
        mdp_config.terrain = layout_grid;
        mdp_config.start_player_positions = final_player_positions;

        Object.entries(params_to_overwrite).forEach(([k, v]) => {
            const curr_val = mdp_config[k];
            if (debug) {
                console.log(`Overwriting mdp layout standard config value ${k}:${curr_val} -> ${v}`);
            }
            mdp_config[k] = v;
        });

        return new OvercookedGridworld(mdp_config);
    }

    _configure_recipes(start_all_orders, max_num_items_for_soup, kwargs) {
        this.recipe_config = {
            max_num_items_for_soup,
            all_orders: start_all_orders,
            ...kwargs,
        };
        Recipe.configure(this.recipe_config);
    }

    equals(other) {
        return (
            np.array_equal(this.terrain_mtx, other.terrain_mtx) &&
            this.start_player_positions === other.start_player_positions &&
            this.start_bonus_orders === other.start_bonus_orders &&
            this.start_all_orders === other.start_all_orders &&
            this.reward_shaping_params === other.reward_shaping_params &&
            this.layout_name === other.layout_name
        );
    }

    /**
     * Creates a copy of the current OvercookedGridworld instance.
     * @returns {OvercookedGridworld} - A new OvercookedGridworld instance that is a copy of the current instance.
     */
    copy() {
        return new OvercookedGridworld({
            terrain: this.terrain_mtx.slice(),
            start_player_positions: this.start_player_positions,
            start_bonus_orders: this.start_bonus_orders,
            rew_shaping_params: JSON.parse(JSON.stringify(this.reward_shaping_params)),
            layout_name: this.layout_name,
            start_all_orders: this.start_all_orders,
        });
    }

    /**
     * Gets the MDP parameters of the current OvercookedGridworld instance.
     * @returns {Object} - An object containing the MDP parameters.
     */
    get mdp_params() {
        return {
            layout_name: this.layout_name,
            terrain: this.terrain_mtx,
            start_player_positions: this.start_player_positions,
            start_bonus_orders: this.start_bonus_orders,
            rew_shaping_params: JSON.parse(JSON.stringify(this.reward_shaping_params)),
            start_all_orders: this.start_all_orders,
        };
    }


    // GAME LOGIC

    get_actions(state) {
        /* Returns the list of lists of valid actions for 'state'.
        The ith element of the list is the list of valid actions that player i
        can take.
        Note that you can request moves into terrain, which are equivalent to
        STAY. The order in which actions are returned is guaranteed to be
        deterministic, in order to allow agents to implement deterministic
        behavior. */
        this._check_valid_state(state);
        return state.players.map((p, i) => {
            return this._get_player_actions(state, i);
        });
    }

    _get_player_actions(state, player_num) {
        return Action.ALL_ACTIONS;
    }

    _check_action(state, joint_action) {
        joint_action.forEach((p_action, i) => {
            const p_legal_actions = this.get_actions(state)[i];
            if (!p_legal_actions.includes(p_action)) {
                throw new Error("Invalid action");
            }
        });
    }

    get_standard_start_state() {
        const start_state = OvercookedState.from_player_positions(
            this.start_player_positions,
            this.start_bonus_orders,
            this.start_all_orders
        );
        return start_state;
    }

    get_random_start_state(random_player_pos = false) {
        const state = this.get_standard_start_state();
        const empty_slots = [];
        for (let y = 0; y < this.terrain_mtx.length; y++) {
            for (let x = 0; x < this.terrain_mtx[0].length; x++) {
                const pos = [x, y];
                const terrain_type = this.get_terrain_type_at_pos(pos);
                if (terrain_type === " ") {
                    empty_slots.push(pos);
                } else if (terrain_type === "X") {
                    // closet
                    // randomly put onion, tomato, dish, soup or empty
                    const obj_type = np.random.choice(["onion", "tomato", "dish", "soup", "empty"]);
                    if (obj_type === "onion") {
                        state.add_object(new ObjectState("onion", pos));
                    } else if (obj_type === "tomato") {
                        state.add_object(new ObjectState("tomato", pos));
                    } else if (obj_type === "dish") {
                        state.add_object(new ObjectState("dish", pos));
                    } else if (obj_type === "soup") {
                        state.add_object(new SoupState(pos, [], 20));
                        const soup = state.get_object(pos);
                        const soup_type = np.random.choice(["onion", "tomato"]);
                        Array.from({ length: 3 }).forEach(() => soup.add_ingredient(new ObjectState(soup_type, pos)));
                        soup._cooking_tick = 20;
                    } else if (obj_type === "empty") {
                        // do nothing
                    } else {
                        throw new Error(`undefined object ${obj_type}`);
                    }
                } else if (terrain_type === "P") {
                    // pot
                    // randomly put empty, soup with 1/2/3 items
                    const item_type = np.random.choice(["onion", "tomato"]);
                    const num = np.random.randint(0, 4);
                    if (num === 0) {
                        // do nothing
                    } else if (num >= 1 && num <= 3) {
                        state.add_object(new SoupState(pos, [], 20));
                        const soup = state.get_object(pos);
                        Array.from({ length: num }).forEach(() => soup.add_ingredient(new ObjectState(item_type, pos)));
                        if (num === 3) {
                            soup._cooking_tick = 20;
                        }
                    } else {
                        throw new Error(`cannot put ${num} onions in soup`);
                    }
                }
            }
        }
        if (random_player_pos) {
            const p = [];
            const ori = [];
            const players = [];
            for (let i = 0; i < this.num_players; i++) {
                let b = true;
                while (b) {
                    const pos = np.random.choice(empty_slots);
                    if (!p.includes(pos)) {
                        p.push(pos);
                        ori.push(np.random.choice(Direction.ALL_DIRECTIONS));
                        players.push(new PlayerState(p[p.length - 1], ori[ori.length - 1]));
                        b = false;
                    }
                }
            }
            state.players = players;
        }
        return state;
    }

    
    get_random_start_state_fn(random_start_pos = false, rnd_obj_prob_thresh = 0.0) {
        const start_state_fn = () => {
            let start_pos;
            if (random_start_pos) {
                const valid_positions = this.get_valid_joint_player_positions();
                start_pos = valid_positions[np.random.choice(valid_positions.length)];
            } else {
                start_pos = this.start_player_positions;
            }

            const start_state = OvercookedState.from_player_positions(
                start_pos,
                { bonus_orders: this.start_bonus_orders, all_orders: this.start_all_orders }
            );

            if (rnd_obj_prob_thresh === 0) {
                return start_state;
            }

            // Arbitrary hard-coding for randomization of objects
            // For each pot, add a random amount of onions and tomatoes with prob rnd_obj_prob_thresh
            // Begin the soup cooking with probability rnd_obj_prob_thresh
            const pots = this.get_pot_states(start_state).empty;
            pots.forEach(pot_loc => {
                const p = Math.random();
                if (p < rnd_obj_prob_thresh) {
                    const n = Math.floor(Math.random() * 3) + 1;
                    const m = Math.floor(Math.random() * (4 - n));
                    const q = Math.random();
                    const cooking_tick = q < rnd_obj_prob_thresh ? 0 : -1;
                    start_state.objects[pot_loc] = SoupState.get_soup(
                        pot_loc, { num_onions: n, num_tomatoes: m, cooking_tick }
                    );
                }
            });

            // For each player, add a random object with prob rnd_obj_prob_thresh
            start_state.players.forEach(player => {
                const p = Math.random();
                if (p < rnd_obj_prob_thresh) {
                    // Different objects have different probabilities
                    const obj = np.random.choice(["dish", "onion", "soup"], { p: [0.2, 0.6, 0.2] });
                    const n = Math.floor(Math.random() * 3) + 1;
                    const m = Math.floor(Math.random() * (4 - n));
                    if (obj === "soup") {
                        player.set_object(
                            SoupState.get_soup(
                                player.position,
                                { num_onions: n, num_tomatoes: m, finished: true }
                            )
                        );
                    } else {
                        player.set_object(new ObjectState(obj, player.position));
                    }
                }
            });

            return start_state;
        };

        return start_state_fn;
    }

    is_terminal({ state }) {
        return state.order_list.length === 0;
    }

    get_transition_states_and_probs({ state, joint_action }) {
        /*Gets information about possible transitions for the action.
        Returns list of (next_state, prob) pairs representing the states
        reachable from 'state' by taking 'action' along with their transition
        probabilities.*/

        let events_infos = {}
        EVENT_TYPES.forEach(event => {
            events_infos[event] = Array(this.num_players).fill(false);
        });

        let action_sets = this.get_actions(state);
        for (let pi = 0; pi < state.players.length; pi++) {
            let [player, action, action_set] =
                [state.players[pi], joint_action[pi], action_sets[pi]];
            assert(_.includes(action_set.map(String), String(action)))
        }
        let new_state = state.deepcopy();

        //resolve interacts first
        const [sparse_reward_by_agent, shaped_reward_by_agent, shaped_info_by_agent] 
        = this.resolve_interacts(new_state, joint_action, events_infos);

        assert(_.isEqual(new_state.player_positions.map(String), state.player_positions.map(String)));
        assert(_.isEqual(new_state.player_orientations.map(String), state.player_orientations.map(String)));
        
        //resolve player movements
        this.resolve_movement(new_state, joint_action, shaped_info_by_agent);

        //finally, environment effects
        this.step_environment_effects(new_state);

        let infos = {"event_infos": events_infos,
                "sparse_reward_by_agent": sparse_reward_by_agent,
                "shaped_reward_by_agent": shaped_reward_by_agent,
                "shaped_info_by_agent": shaped_info_by_agent}

        return [[new_state, 1.0], infos];
    }

    resolve_interacts(new_state, joint_action, events_infos) {
        /**
         * Resolve any INTERACT actions, if present.
         *
         * Currently if two players both interact with a terrain, we resolve player 1's interact
         * first and then player 2's, without doing anything like collision checking.
         */
        let pot_states = this.get_pot_states(new_state);
        // We divide reward by agent to keep track of who contributed
        let sparse_reward = Array(this.num_players).fill(0);
        let shaped_reward = Array(this.num_players).fill(0);
        // MARK
        let shaped_info = Array.from({ length: this.num_players }, () => Object.fromEntries(SHAPED_INFOS.map(info => [info, 0])));
        for (let player_idx = 0; player_idx < new_state.players.length; player_idx++) {
            let player = new_state.players[player_idx];
            let action = joint_action[player_idx];
            if (action !== Action.INTERACT) {
                if (Direction.ALL_DIRECTIONS.includes(action)) {
                    shaped_info[player_idx]["MOVEMENT"] += 1;
                } else if (action === Action.STAY) {
                    shaped_info[player_idx]["STAY"] += 1;
                }
                continue;
            }

            let pos = player.position, o = player.orientation;
            let i_pos = Action.move_in_direction(pos, o);
            let terrain_type = this.get_terrain_type_at_pos(i_pos);

            // NOTE: we always log pickup/drop before performing it, as that's
            // what the logic of determining whether the pickup/drop is useful assumes
            if (terrain_type === "X") {
                if (player.has_object() && !new_state.has_object(i_pos)) {
                    let obj_name = player.get_object().name;
                    this.log_object_drop(events_infos, new_state, obj_name, pot_states, player_idx);
                    shaped_info[player_idx][`put_${obj_name}_on_X`] += 1;

                    shaped_info[player_idx][`place_${obj_name}_on_X`] += 1;
                    shaped_info[player_idx][`${obj_name}_placed_on_X`] += 1;

                    // Drop object on counter
                    let obj = player.remove_object();
                    obj.last_owner = player_idx
                    new_state.add_object(obj, i_pos);

                } else if (!player.has_object() && new_state.has_object(i_pos)) {
                    let obj_name = new_state.get_object(i_pos).name;
                    this.log_object_pickup(events_infos, new_state, obj_name, pot_states, player_idx);
                    shaped_info[player_idx][`pickup_${obj_name}_from_X`] += 1;
                    shaped_info[player_idx][`${obj_name}_placed_on_X`] -= 1;

                    // Pick up object from counter
                    let obj = new_state.remove_object(i_pos);
                    if(obj.last_owner === player_idx){
                        shaped_info[player_idx][`place_${obj_name}_on_X`] -= 1 
                    }else{
                        shaped_info[player_idx][`recieve_${obj_name}_via_X`] += 1 
                    }

                    player.set_object(obj);
                } else {
                    shaped_info[player_idx]["IDLE_INTERACT"] += 1;
                }

            } else if (terrain_type === "O" && player.held_object === null) {
                this.log_object_pickup(events_infos, new_state, "onion", pot_states, player_idx);
                shaped_info[player_idx][`pickup_onion_from_O`] += 1;

                // Onion pickup from dispenser
                let obj = new ObjectState("onion", pos);
                player.set_object(obj);

            } else if (terrain_type === "T" && player.held_object === null) {
                shaped_info[player_idx][`pickup_tomato_from_T`] += 1;
                // Tomato pickup from dispenser
                player.set_object(new ObjectState("tomato", pos));

            } else if (terrain_type === "D" && player.held_object === null) {
                this.log_object_pickup(events_infos, new_state, "dish", pot_states, player_idx);
                shaped_info[player_idx][`pickup_dish_from_D`] += 1;

                // Give shaped reward if pickup is useful
                if (this.is_dish_pickup_useful(new_state, pot_states)) {
                    shaped_reward[player_idx] += this.reward_shaping_params["DISH_PICKUP_REWARD"];
                    shaped_info[player_idx][`USEFUL_DISH_PICKUP`] += 1;
                }

                // Perform dish pickup from dispenser
                let obj = new ObjectState("dish", pos);
                player.set_object(obj);

            } else if (terrain_type === "P" && !player.has_object()) {
                // An interact action will only start cooking the soup if we are using the new dynamics
                if (!this.old_dynamics && this.soup_to_be_cooked_at_location(new_state, i_pos)) {
                    let soup = new_state.get_object(i_pos);
                    soup.begin_cooking();
                    shaped_info[player_idx]["cook"] += 1;
                }
            } else if (terrain_type == "P" && player.has_object()) {
                if (player.get_object().name === "dish" && this.soup_ready_at_location(new_state, i_pos)) {
                    this.log_object_pickup(events_infos, new_state, "soup", pot_states, player_idx);
                    shaped_info[player_idx][`SOUP_PICKUP`] += 1;
            
                    // Pick up soup
                    player.remove_object();  // Remove the dish
                    let obj = new_state.remove_object(i_pos);  // Get soup
                    player.set_object(obj);
                    shaped_reward[player_idx] += this.reward_shaping_params["SOUP_PICKUP_REWARD"];
                } else if (Recipe.ALL_INGREDIENTS.includes(player.get_object().name)) {
                    // Adding ingredient to soup
            
                    if (!new_state.has_object(i_pos)) {
                        // Pot was empty, add soup to it
                        new_state.add_object(new SoupState(i_pos, [] ));
                    }
            
                    // Add ingredient if possible
                    let soup = new_state.get_object(i_pos);
                    if (!soup.is_full) {
                        let old_soup = soup.deepcopy();
                        let obj = player.remove_object();
                        soup.add_ingredient(obj);
                        // TODO: reward
                        if (this.is_potting_optimal(new_state, old_soup, soup) || this.is_potting_viable(new_state, old_soup, soup)) {
                            shaped_reward[player_idx] += this.reward_shaping_params["PLACEMENT_IN_POT_REW"];
                        }
                        if (this.is_potting_optimal(new_state, old_soup, soup)) {
                            shaped_info[player_idx]["optimal_placement"] += 1;
                        }
                        if (this.is_potting_viable(new_state, old_soup, soup)) {
                            shaped_info[player_idx]["viable_placement"] += 1;
                        }
                        if (this.is_potting_catastrophic(new_state, old_soup, soup)) {
                            shaped_info[player_idx]["catastrophic_placement"] += 1;
                        }
                        if (this.is_potting_useless(new_state, old_soup, soup)) {
                            shaped_info[player_idx]["useless_placement"] += 1;
                        }
                        shaped_info[player_idx][`potting_${obj.name}`] += 1;
                        shaped_info[player_idx]["PLACEMENT_IN_POT"] += 1;
            
                        // Log potting
                        this.log_object_potting(events_infos, new_state, old_soup, soup, obj.name, player_idx);
                    }
                }
            
            } else if (terrain_type === "S" && player.has_object()) {
                let obj = player.get_object();
                if (obj.name === "soup") {
                    let delivery_rew = this.deliver_soup(new_state, player, obj);
                    sparse_reward[player_idx] += delivery_rew;
                    shaped_info[player_idx]["delivery"] += 1;
                    let _map = {
                        2: "two",
                        3: "three",
                    };  // MARK: order with different sizes may exist
                    if (obj.ingredients.length > 1) {
                        shaped_info[player_idx][`deliver_size_${_map[obj.ingredients.length]}_order`] += 1;
                    }
                    if (delivery_rew <= 0) {
                        shaped_info[player_idx]["deliver_useless_order"] += 1;
                    }
        
                    // Log soup delivery
                    events_infos["soup_delivery"][player_idx] = true;
                }
            } else {
                shaped_info[player_idx]["IDLE_INTERACT"] += 1;
            }
        }
        return [sparse_reward, shaped_reward, shaped_info];
    }

    get_recipe_value(
        state,
        recipe,
        discounted = false,
        base_recipe = null,
        potential_params = {}
    ) {
        /**
         * Return the reward the player should receive for delivering this recipe
         *
         * The player receives 0 if recipe not in all_orders, receives base value * order_bonus
         * if recipe is in bonus orders, and receives base value otherwise
         */
        if (!discounted) {
            if (!state.all_orders.some(
                _recipe => _recipe.ingredients.toString() == recipe.ingredients.toString()
            )) {
                // TODO: penalty
                return -10;
            }
    
            if (!state.bonus_orders.some(
                _recipe => _recipe.ingredients.toString() == recipe.ingredients.toString()
            )) {
                return recipe.value;
            }
    
            return this.order_bonus * recipe.value;
            
        } else {
            // Calculate missing ingredients needed to complete recipe
            let missing_ingredients = [...recipe.ingredients];
            let prev_ingredients = base_recipe ? [...base_recipe.ingredients] : [];
            for (let ingredient of prev_ingredients) {
                let index = missing_ingredients.indexOf(ingredient);
                if (index > -1) {
                    missing_ingredients.splice(index, 1);
                }
            }
            let n_tomatoes = missing_ingredients.filter(i => i === Recipe.TOMATO).length;
            let n_onions = missing_ingredients.filter(i => i === Recipe.ONION).length;
    
            let { gamma, pot_onion_steps, pot_tomato_steps } = potential_params;
    
            return (
                Math.pow(gamma, recipe.time) *
                Math.pow(gamma, pot_onion_steps * n_onions) *
                Math.pow(gamma, pot_tomato_steps * n_tomatoes) *
                this.get_recipe_value(state, recipe, false)
            );
        }
    }
    
    deliver_soup(state, player, soup) {
        /**
         * Deliver the soup, and get reward if there is no order list
         * or if the type of the delivered soup matches the next order.
         */
        assert(soup.name === "soup", "Tried to deliver something that wasn't soup");
        assert(soup.is_ready, "Tried to deliver soup that isn't ready");
        player.remove_object();
    
        return this.get_recipe_value(state, soup.recipe);
    }
    

    resolve_movement(state, joint_action, shaped_info_by_agent) {
        /**
         * Resolve player movement and deal with possible collisions
         */
        let old_positions = state.players.map(p => p.position);
        let old_orientations = state.players.map(p => p.orientation);
        let { new_positions, new_orientations } = this.compute_new_positions_and_orientations(state.players, joint_action);
    
        if (shaped_info_by_agent !== null) {
            for (let player_idx = 0; player_idx < state.players.length; player_idx++) {
                let old_pos = old_positions[player_idx];
                let new_pos = new_positions[player_idx];
                let old_o = old_orientations[player_idx];
                let new_o = new_orientations[player_idx];
    
                if (Direction.ALL_DIRECTIONS.includes(joint_action[player_idx]) && old_pos === new_pos && old_o === new_o) {
                    shaped_info_by_agent[player_idx]["IDLE_MOVEMENT"] += 1;
                }
            }
        }
    
        for (let player_idx = 0; player_idx < state.players.length; player_idx++) {
            let player_state = state.players[player_idx];
            let new_pos = new_positions[player_idx];
            let new_o = new_orientations[player_idx];
            player_state.update_pos_and_or(new_pos, new_o);
        }
    }
    

    compute_new_positions_and_orientations(old_player_states, joint_action) {
        //Compute new positions and orientations ignoring collisions
        let new_positions = [];
        let old_positions = [];
        let new_orientations = [];
        for (let pi = 0; pi < old_player_states.length; pi++) {
            let p = old_player_states[pi];
            let a = joint_action[pi];
            let [new_pos, new_o] = this._move_if_direction(p.position, p.orientation, a);
            new_positions.push(new_pos);
            old_positions.push(p.position);
            new_orientations.push(new_o);
        }
        new_positions = this._handle_collisions(old_positions, new_positions);
        return {new_positions, new_orientations};
    }

    _handle_collisions(old_positions, new_positions) {
        //only 2 players for nwo
        if (this.is_collision(old_positions, new_positions)) {
            return old_positions;
        }
        return new_positions;
    }

    is_collision(old_positions, new_positions) {
        let [p1_old, p2_old] = old_positions;
        let [p1_new, p2_new] = new_positions;
        if (_.isEqual(p1_new, p2_new)) {
            return true
        }
        else if (_.isEqual(p1_new, p2_old) && _.isEqual(p1_old, p2_new)) {
            return true
        }
        return false
    }

    _move_if_direction(position, orientation, action) {
        if (!_.includes(Action.MOTION_ACTIONS.map(String), String(action))) {
            return [position, orientation]
        }
        let new_pos = Action.move_in_direction(position, action);
        let new_orientation;

        if (_.isEqual(Action.STAY, action)) {
            new_orientation = orientation;
        }
        else {
            new_orientation = action;
        }

        if (!_.includes(this.get_valid_player_positions().map(String), String(new_pos))) {
            return [position, new_orientation]
        }

        return [new_pos, new_orientation]
    }

    _get_terrain_type_pos_dict() {
        let pos_dict = {};
        for (let y = 0; y < this.terrain_mtx.length; y++) {
            for (let x = 0; x < this.terrain_mtx[y].length; x++) {
                let ttype = this.terrain_mtx[y][x];
                if (!pos_dict.hasOwnProperty(ttype)) {
                    pos_dict[ttype] = [];
                }
                pos_dict[ttype].push([x, y]);
            }
        }
        return pos_dict;
    }

    get_terrain_type_at(pos) {
        let [x, y] = pos;
        return this.terrain_mtx[y][x];
    }

    step_environment_effects(state) {
        state.timestep += 1;
        for (let obj of Object.values(state.objects)) {
            if (obj.name === "soup") {
                // automatically starts cooking when the pot has 3 ingredients
                if (this.old_dynamics && (!obj.is_cooking && !obj.is_ready && obj.ingredients.length === 3)) {
                    obj.begin_cooking();
                }
                if (obj.is_cooking) {
                    obj.cook();
                }
            }
        }
    }
    

    // LAYOUT / STATE INFO

    get_valid_player_positions(){
        return this.terrain_pos_dict[' ']
    }

    get_valid_joint_player_positions(){

        // Helper function to generate Cartesian product
        function cartesianProduct(arr, repeat) {
        if (repeat === 1) return arr.map(e => [e]);
        const rest = cartesianProduct(arr, repeat - 1);
        return arr.flatMap(e => rest.map(r => [e, ...r]));
        }
        // Returns all valid tuples of the form (p0_pos, p1_pos, p2_pos, ...)
        const validPositions = this.get_valid_player_positions();
        const allJointPositions = cartesianProduct(validPositions, this.num_players);
        const validJointPositions = allJointPositions.filter(jPos => !this.is_joint_position_collision(jPos));
        return validJointPositions;
    }

    get_valid_player_positions_and_orientations() {
        const validStates = [];
        const validPositions = this.get_valid_player_positions();
        const allDirections = Direction.ALL_DIRECTIONS;
    
        validPositions.forEach(pos => {
            allDirections.forEach(d => {
                validStates.push([pos, d]);
            });
        });
        return validStates;
    }

    get_valid_joint_player_positions_and_orientations() {
        // Helper function to generate Cartesian product
        function cartesianProduct(arr, repeat) {
        if (repeat === 1) return arr.map(e => [e]);
        const rest = cartesianProduct(arr, repeat - 1);
        return arr.flatMap(e => rest.map(r => [e, ...r]));
        }
        // All joint player position and orientation pairs that are not overlapping and on empty terrain.
        const validPlayerStates = this.get_valid_player_positions_and_orientations();
    
        const validJointPlayerStates = [];
        const allCombinations = cartesianProduct(validPlayerStates, this.num_players);
    
        allCombinations.forEach(playersPosAndOrientations => {
            const jointPosition = playersPosAndOrientations.map(plyerPosAndOr => plyerPosAndOr[0]);
            if (!this.is_joint_position_collision(jointPosition)) {
                validJointPlayerStates.push(playersPosAndOrientations);
            }
        });
        return validJointPlayerStates;
    }

    get_adjacent_features(player) {
        const adjFeats = [];
        const pos = player.position;
        const allDirections = Direction.ALL_DIRECTIONS;
    
        allDirections.forEach(d => {
            const adjPos = Action.moveInDirection(pos, d);
            adjFeats.push([adjPos, this.get_terrain_type_at_pos(adjPos)]);
        });
    
        return adjFeats;
    }

    get_terrain_type_at_pos(pos) {
        const [x, y] = pos;
        return this.terrain_mtx[y][x];
    }

    get_dish_dispenser_locations() {
        return Array.from(this.terrain_pos_dict["D"]);
    }
    

    get_onion_dispenser_locations(){
        return Array.from(this.terrain_pos_dict["O"]);
    }

    get_tomato_dispenser_locations(){
        return Array.from(this.terrain_pos_dict["T"]);
    }

    get_serving_locations(){
        return Array.from(this.terrain_pos_dict["S"]);
        
    }

    get_pot_locations(){
        return Array.from(this.terrain_pos_dict["P"]);
    }

    get_counter_locations(){
        return Array.from(this.terrain_pos_dict["X"]);
    }

    get_num_pots(){
        return this.get_pot_locations.length;
    }
    
    get_pot_states(state){
        /**
         * Returns dict with structure:
         * {
         *  empty: [positions of empty pots]
         * 'x_items': [soup objects with x items that have yet to start cooking],
         * 'cooking': [soup objs that are cooking but not ready]
         * 'ready': [ready soup objs],
         * }
         * NOTE: all returned pots are just pot positions
         */
        const potsStatesDict = {
            empty: [],
            ready: [],
            cooking: []
        };

        this.get_pot_locations().forEach(potPos => {
            if (!state.has_object(potPos)) {
                potsStatesDict.empty.push(potPos);
            } else {
                const soup = state.get_object(potPos);
                assert(soup.name === "soup", `soup at ${potPos} is not a soup but a ${soup.name}`);
                if (soup.isReady) {
                    potsStatesDict.ready.push(potPos);
                } else if (soup.isCooking) {
                    potsStatesDict.cooking.push(potPos);
                } else {
                    const numIngredients = soup.ingredients.length;
                    if (!potsStatesDict[`${numIngredients}_items`]) {
                        potsStatesDict[`${numIngredients}_items`] = [];
                    }
                    potsStatesDict[`${numIngredients}_items`].push(potPos);
                }
            }
        });

        return potsStatesDict;
    }

    get_counter_objects_dict(state, counterSubset = null){
        // Returns a dictionary of pos:objects on counters by type
        const countersConsidered = counterSubset === null ? this.terrain_pos_dict["X"] : counterSubset;
        const counterObjectsDict = {};
    
        for(let key in state.objects){
            let obj = state.objects[key];
            if (countersConsidered.some(
                array => array[0]===obj.position[0] && array[1]===obj.position[1]
            ))
            {
                if (!counterObjectsDict[obj.name]) {
                    counterObjectsDict[obj.name] = [];
                }
                counterObjectsDict[obj.name].push(obj.position);
            }
        }
    
        return counterObjectsDict;
    } 

    get_empty_counter_locations(state){
        const counterLocations = this.get_counter_locations();
        return counterLocations.filter(pos => !state.has_object(pos));
    }

    get_empty_pot(pot_states){
        // Returns pots that have 0 items in them
        return pot_states["empty"];

    }

    get_non_empty_pots(pot_states){
        return this.get_full_pots(pot_states) + this.get_partially_full_pots(pot_states)
    }

    get_ready_pots(pot_states){
        return pot_states["ready"];
    }

    get_cooking_pots(pot_states){
        return pot_states["cooking"];
    }

    get_full_but_not_cooking_pots(pot_states){
        let full_pots = pot_states[`${Recipe.MAX_NUM_INGREDIENTS}_items`];
        if(full_pots === undefined) return [];
        else return full_pots;
    }
    
    get_full_pots(pot_states) {
        return [
            ...this.get_cooking_pots(pot_states),
            ...this.get_ready_pots(pot_states),
            ...this.get_full_but_not_cooking_pots(pot_states)
        ];
    }

    get_partially_full_pots(pot_states) {
        return Array.from(new Set([].concat(...Array.from({ length: Recipe.MAX_NUM_INGREDIENTS - 1 }, (_, i) => pot_states[`${i + 1}_items`] || []))));
    }

    soup_ready_at_location(state, pos) {
        if (!state.has_object(pos)) {
            return false;
        }
        let obj = state.get_object(pos);
        assert(obj.name === "soup", "Object in pot was not soup");
        return obj.is_ready;
    }  

    soup_to_be_cooked_at_location(state, pos) {
        if (!state.has_object(pos)) {
            return false;
        }
        let obj = state.get_object(pos);
        return obj.name === "soup" && !obj.is_cooking && !obj.is_ready && obj.ingredients.length > 0;
    }

    _check_valid_state(state) {
        /**
         * Checks that the state is valid.
         *
         * Conditions checked:
         * - Players are on free spaces, not terrain
         * - Held objects have the same position as the player holding them
         * - Non-held objects are on terrain
         * - No two players or non-held objects occupy the same position
         * - Objects have a valid state (eg. no pot with 4 onions)
         */
        let all_objects = Object.values(state.objects);
        for (let player_state of state.players) {
            // Check that players are not on terrain
            let pos = player_state.position;
            assert(this.get_valid_player_positions().some(
                array => array[0]===pos[0] && array[1]===pos[1]
            ));

            // Check that held objects have the same position
            if (player_state.held_object !== null) {
                all_objects.push(player_state.held_object);
                assert(player_state.held_object.position === player_state.position);
            }
        }

        for (let [obj_pos, obj_state] of Object.entries(state.objects)) {
            // Check that the hash key position agrees with the position stored
            // in the object state
            let obj_pos_array = obj_state.position;

            // Check that non-held objects are on terrain
            assert(this.get_terrain_type_at_pos(obj_pos_array)  !== " ");
        }

        // Check that players and non-held objects don't overlap
        let all_pos = state.players.map(player_state => player_state.position);
        all_pos = all_pos.concat(Object.values(state.objects).map(obj_state => obj_state.position));
        assert(all_pos.length === new Set(all_pos).size, "Overlapping players or objects");

        // Check that objects have a valid state
        for (let obj_state of all_objects) {
            assert(obj_state.is_valid());
        }
    }

    find_free_counters_valid_for_both_players(state, mlam) {
        /**
         * Finds all empty counter locations that are accessible to both players
         */
        let [one_player, other_player] = state.players;
        let free_counters = this.get_empty_counter_locations(state);
        let free_counters_valid_for_both = [];
        for (let free_counter of free_counters) {
            let goals = mlam.motion_planner.motion_goals_for_pos[free_counter];
            if (goals.some(goal => mlam.motion_planner.is_valid_motion_start_goal_pair(one_player.pos_and_or, goal)) &&
                goals.some(goal => mlam.motion_planner.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal))) {
                free_counters_valid_for_both.push(free_counter);
            }
        }
        return free_counters_valid_for_both;
    }

    _get_optimal_possible_recipe(state, recipe, discounted, potential_params, return_value) {
        /**
         * Traverse the recipe-space graph using DFS to find the best possible recipe that can be made
         * from the current recipe
         *
         * Because we can't have empty recipes, we handle the case by letting recipe==None be a stand-in for empty recipe
         */
        let start_recipe = recipe;
        let visited = new Set();
        let stack = [];
        let best_recipe = recipe;
        let best_value = 0;
        if (!recipe) {
            for (let ingredient of Recipe.ALL_INGREDIENTS) {
                stack.push(new Recipe([ingredient]));
            }
        } else {
            stack.push(recipe);
        }
    
        while (stack.length > 0) {
            let curr_recipe = stack.pop();
            if (!visited.has(curr_recipe)) {
                visited.add(curr_recipe);
                let curr_value = this.get_recipe_value(
                    state,
                    curr_recipe,
                    start_recipe,
                    discounted,
                    potential_params
                );
                if (curr_value > best_value) {
                    best_value = curr_value;
                    best_recipe = curr_recipe;
                }
    
                for (let neighbor of curr_recipe.neighbors()) {
                    if (!visited.has(neighbor)) {
                        stack.push(neighbor);
                    }
                }
            }
        }
    
        if (return_value) {
            return [best_recipe, best_value];
        }
        return best_recipe;
    }

    get_optimal_possible_recipe(state, recipe, discounted = false, potential_params = {}, return_value = false) {
        /**
         * Return the best possible recipe that can be made starting with ingredients in `recipe`
         * Uses self._optimal_possible_recipe as a cache to avoid re-computing. This only works because
         * the recipe values are currently static (i.e. bonus_orders doesn't change). Would need to have cache
         * flushed if order dynamics are introduced
         */
        let cache_valid = !discounted || this._prev_potential_params === potential_params;
        if (!cache_valid) {
            if (discounted) {
                this._opt_recipe_discount_cache = {};
            } else {
                this._opt_recipe_cache = {};
            }
        }
    
        let cache;
        if (discounted) {
            cache = this._opt_recipe_discount_cache;
            this._prev_potential_params = potential_params;
        } else {
            cache = this._opt_recipe_cache;
        }
    
        if (!cache.hasOwnProperty(recipe)) {
            // Compute best recipe now and store in cache for later use
            let [opt_recipe, value] = this._get_optimal_possible_recipe(
                state,
                recipe,
                discounted,
                potential_params,
                true
            );
            cache[recipe] = [opt_recipe, value];
        }
    
        // Return best recipe (and value) from cache
        if (return_value) {
            return cache[recipe];
        }
        return cache[recipe][0];
    }
    

    static _assert_valid_grid(grid){
        /*
        Raises an AssertionError if the grid is invalid.

        grid:  A sequence of sequences of spaces, representing a grid of a
        certain height and width. grid[y][x] is the space at row y and column
        x. A space must be either 'X' (representing a counter), ' ' (an empty
        space), 'O' (onion supply), 'P' (pot), 'D' (dish supply), 'S' (serving
        location), '1' (player 1) and '2' (player 2).
        */
        
        const height = grid.length;
        const width = grid[0].length;
        
        // Make sure the grid is not ragged
        if (!grid.every(row => row.length === width)) {
            throw new Error("Ragged grid");
        }
        
        // Borders must not be free spaces
        const isNotFree = (c) => "XOPDST".includes(c);
        
        for (let y = 0; y < height; y++) {
            if (!isNotFree(grid[y][0])) {
                throw new Error("Left border must not be free");
            }
            if (!isNotFree(grid[y][width - 1])) {
                throw new Error("Right border must not be free");
            }
        }
        
        for (let x = 0; x < width; x++) {
            if (!isNotFree(grid[0][x])) {
                throw new Error("Top border must not be free");
            }
            if (!isNotFree(grid[height - 1][x])) {
                throw new Error("Bottom border must not be free");
            }
        }
        
        const allElements = grid.flat();
        const digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9"];
        const layoutDigits = allElements.filter(e => digits.includes(e));
        const numPlayers = layoutDigits.length;
        
        if (numPlayers <= 0) {
            throw new Error("No players (digits) in grid");
        }
        
        const sortedLayoutDigits = layoutDigits.map(Number).sort((a, b) => a - b);
        if (!sortedLayoutDigits.every((val, index) => val === index + 1)) {
            throw new Error("Some players were missing");
        }
        
        if (!allElements.every(c => "XOPDST123456789 ".includes(c))) {
            throw new Error("Invalid character in grid");
        }
        
        if (allElements.filter(c => c === "1").length !== 1) {
            throw new Error("'1' must be present exactly once");
        }
        if (allElements.filter(c => c === "D").length < 1) {
            throw new Error("'D' must be present at least once");
        }
        if (allElements.filter(c => c === "S").length < 1) {
            throw new Error("'S' must be present at least once");
        }
        if (allElements.filter(c => c === "P").length < 1) {
            throw new Error("'P' must be present at least once");
        }
        if (allElements.filter(c => c === "O").length < 1 && allElements.filter(c => c === "T").length < 1) {
            throw new Error("'O' or 'T' must be present at least once");
        }
    


    }
    

    // EVENT LOGGING HELPER METHODS

    log_object_potting(events_infos, state, old_soup, new_soup, obj_name, player_index){
        // """Player added an ingredient to a pot"""
        let obj_pickup_key = "potting_" + obj_name;
        if(!(obj_pickup_key in events_infos)){
        console.log("Unknown event " + (obj_pickup_key));
        }

        events_infos[obj_pickup_key][player_index] = true;

        let boundIsPottingOptimal = this.is_potting_optimal.bind(this);
        let boundIsPottingCatastriphic = this.is_potting_catastrophic.bind(this);
        let boundIsPottingViable = this.is_potting_viable.bind(this);
        let boundIsPottingUseless = this.is_potting_useless.bind(this);

        const POTTING_FNS = {
            "optimal": boundIsPottingOptimal,
            "catastrophic": boundIsPottingCatastriphic,
            "viable": boundIsPottingViable,
            "useless": boundIsPottingUseless
        };

        for(const outcome in POTTING_FNS){
            let outcome_fn = POTTING_FNS[outcome];
            if(outcome_fn(state, old_soup, new_soup)){
                let potting_key = `${outcome}_${obj_name}_potting`;
                events_infos[potting_key][player_index] = true;
            }
        }
    }

    log_object_pickup(events_infos, state, obj_name, pot_states, player_index){
        // """Player picked an object up from a counter or a dispenser"""

        let obj_pickup_key = obj_name + "_pickup";
        if((!(obj_pickup_key in events_infos))){
            console.log("Unknown event " + (obj_pickup_key));
        }  

        events_infos[obj_pickup_key][player_index] = true;

        let boundIsIngredientPickupUseful = this.is_ingredient_pickup_useful.bind(this);
        let boundIsDishPickupUseful = this.is_dish_pickup_useful.bind(this);

        const USEFUL_PICKUP_FNS = {
            "tomato": boundIsIngredientPickupUseful,
            "onion": boundIsIngredientPickupUseful,
            "dish": boundIsDishPickupUseful
        };

        if (obj_name in USEFUL_PICKUP_FNS){
            if(USEFUL_PICKUP_FNS[obj_name](state, pot_states, player_index)){
            let obj_useful_key = "useful_" + obj_name + "_pickup";
            events_infos[obj_useful_key][player_index] = true;
            }
        }
    }

    log_object_drop(events_infos, state, obj_name, pot_states, player_index){
        // """Player dropped the object on a counter"""
        let obj_drop_key = obj_name + "_drop";
        if((!(obj_drop_key in events_infos))){
            console.log("Unknown event " + (obj_pickup_key));
        }

        events_infos[obj_drop_key][player_index] = true;

        let boundIsIngredientDropUseful = this.is_ingredient_drop_useful.bind(this);
        let boundIsDishDropUseful = this.is_dish_pickup_useful.bind(this);

        const USEFUL_DROP_FNS = {
            "tomato": boundIsIngredientDropUseful,
            "onion": boundIsIngredientDropUseful,
            "dish": boundIsDishDropUseful,
        };

        if (obj_name in USEFUL_DROP_FNS){
            if(USEFUL_DROP_FNS[obj_name](state, pot_states, player_index)){
                let obj_useful_key = "useful_" + obj_name + "_drop";
                events_infos[obj_useful_key][player_index] = true;
            }
        }

    }

    is_dish_pickup_useful(state, pot_states, player_index = null) {
        /**
         * NOTE: this only works if self.num_players == 2
         * Useful if:
         * - Pot is ready/cooking and there is no player with a dish
         * - 2 pots are ready/cooking and there is one player with a dish          | -> number of dishes in players hands < number of ready/cooking/partially full soups
         * - Partially full pot is ok if the other player is on course to fill it
         *
         * We also want to prevent picking up and dropping dishes, so add the condition
         * that there must be no dishes on counters
         */
        if (this.num_players !== 2) {
            return false;
        }
    
        // This next line is to prevent reward hacking (this logic is also used by reward shaping)

        let dishes_on_counters = this.get_counter_objects_dict(state)["dish"];
        let length;
        if(dishes_on_counters === undefined) length = 0;
        else length = dishes_on_counters.length;

        let no_dishes_on_counters = length === 0;

        if(state.player_objects_by_type["dish"] === undefined) length = 0;
        else length = state.player_objects_by_type["dish"].length;

        let num_player_dishes = length;

        let non_empty_pots = this.get_ready_pots(pot_states).length
            + this.get_cooking_pots(pot_states).length
            + this.get_partially_full_pots(pot_states).length;
    
        return no_dishes_on_counters && num_player_dishes < non_empty_pots;
    }

    is_dish_drop_useful(state, pot_states, player_index) {
        /**
         * NOTE: this only works if self.num_players == 2
         * Useful if:
         * - Onion is needed (all pots are non-full)
         * - Nobody is holding onions
         */
        if (this.num_players !== 2) {
            return false;
        }
        let all_non_full = this.get_full_pots(pot_states).length === 0;
        let other_player = state.players[1 - player_index];
        let other_player_holding_onion = other_player.has_object() && other_player.get_object().name === "onion";
        return all_non_full && !other_player_holding_onion;
    }
    
    is_ingredient_pickup_useful(state, pot_states, player_index) {
        /**
         * NOTE: this only works if self.num_players == 2
         * Always useful unless:
         * - All pots are full & other agent is not holding a dish
         */
        if (this.num_players !== 2) {
            return false;
        }
        let all_pots_full = this.num_pots === this.get_full_pots(pot_states).length;
        let other_player = state.players[1 - player_index];
        let other_player_has_dish = other_player.has_object() && other_player.get_object().name === "dish";
        return !(all_pots_full && !other_player_has_dish);
    }

    is_ingredient_drop_useful(state, pot_states, player_index) {
        /**
         * NOTE: this only works if self.num_players == 2
         * Useful if:
         * - Dish is needed (all pots are full)
         * - Nobody is holding a dish
         */
        if (this.num_players !== 2) {
            return false;
        }
        let all_pots_full = this.get_full_pots(pot_states).length === this.num_pots;
        let other_player = state.players[1 - player_index];
        let other_player_holding_dish = other_player.has_object() && other_player.get_object().name === "dish";
        return all_pots_full && !other_player_holding_dish;
    }
    
    is_potting_optimal(state, old_soup, new_soup) {
        /**
         * True if the highest valued soup possible is the same before and after the potting
         */
        let old_recipe = old_soup.ingredients && old_soup.ingredients.length != 0 ? new Recipe(old_soup.ingredients) : null;
        let new_recipe = new Recipe(new_soup.ingredients);
        let old_val = this.get_recipe_value(state, this.get_optimal_possible_recipe(state, old_recipe));
        let new_val = this.get_recipe_value(state, this.get_optimal_possible_recipe(state, new_recipe));
        return old_val === new_val;
    }

    is_potting_viable(state, old_soup, new_soup) {
        /**
         * True if there exists a non-zero reward soup possible from new ingredients
         */
        let new_recipe = new Recipe(new_soup.ingredients);
        let new_val = this.get_recipe_value(state, this.get_optimal_possible_recipe(state, new_recipe));
        return new_val > 0;
    }
    
    is_potting_catastrophic(state, old_soup, new_soup) {
        /**
         * True if no non-zero reward soup is possible from new ingredients
         */
        let old_recipe = old_soup.ingredients && old_soup.ingredients.length != 0 ? new Recipe(old_soup.ingredients) : null;
        let new_recipe = new Recipe(new_soup.ingredients);
        let old_val = this.get_recipe_value(state, this.get_optimal_possible_recipe(state, old_recipe));
        let new_val = this.get_recipe_value(state, this.get_optimal_possible_recipe(state, new_recipe));
        return old_val > 0 && new_val <= 0;
    }
    
    is_potting_useless(state, old_soup, new_soup) {
        /**
         * True if ingredient added to a soup that was already guaranteed to be worth at most 0 points
         */
        let old_recipe = old_soup.ingredients && old_soup.ingredients.length != 0 ? new Recipe(old_soup.ingredients) : null;
        let old_val = this.get_recipe_value(state, this.get_optimal_possible_recipe(state, old_recipe));
        return old_val <= 0;
    }
    

    lossless_state_encoding(state, playerIndex, padding) {
        const BASE_FEATURE_INDICES = {
            "P": 10,
            "X": 11,
            "O": 12,
            "D": 13,
            "S": 14
        };

        // The soup numbers are duplicated in preprocessState
        const VARIABLE_FEATURE_INDICES = {
            "onions_in_pot": 15,
            "onions_cook_time": 16,
            "onion_soup_loc": 17,
            "dish": 18,
            "onion": 19
        };

        function constant(element, shape) {
            function helper(i) {
                let size = shape[i];
                if (i === shape.length - 1) {
                    return Array(size).fill(element);
                }
                return Array(size).fill().map(() => helper(i + 1));
            }
            return helper(0);
        }

        // All of our models have a batch size of 30, but we only want to predict on
        // a single state, so we put zeros everywhere else.
        let terrain = this.terrain_mtx;
        let shape = [padding, terrain[0].length, terrain.length, 20];
        let result = constant(0, shape);

        function handle_object(obj) {
            let [x, y] = obj.position;
            if (obj.name === 'soup') {
                let [soup_type, num_onions, cook_time] = obj.state;
                if (terrain[y][x] === 'P') {
                    result[0][x][y][15] = num_onions;
                    result[0][x][y][16] = Math.min(cook_time, 20);
                } else {
                    result[0][x][y][17] = 1;
                }
            } else {
                let feature_index = VARIABLE_FEATURE_INDICES[obj.name];
                result[0][x][y][feature_index] = 1;
            }
        }

        for (let i = 0; i < state.players.length; i++) {
            let player = state.players[i];
            let [x, y] = player.position;
            let orientation = Direction.DIRECTION_TO_INDEX[player.orientation];
            if (playerIndex === i) {
                result[0][x][y][0] = 1;
                result[0][x][y][2 + orientation] = 1;
            } else {
                result[0][x][y][1] = 1;
                result[0][x][y][6 + orientation] = 1;
            }

            if (player.has_object()) {
                handle_object(player.held_object);
            }
        }

        let pos_dict = this._get_terrain_type_pos_dict();
        for (let ttype in BASE_FEATURE_INDICES) {
            let t_index = BASE_FEATURE_INDICES[ttype];
            for (let i in pos_dict[ttype]) {
                let [x, y] = pos_dict[ttype][i];
                result[0][x][y][t_index] = 1;
            }
        }

        for (let i in state.objects) {
            let obj = state.objects[i];
            handle_object(obj);
        }
        return [result, shape];
    }
}
OvercookedGridworld.COOK_TIME = 20;
OvercookedGridworld.DELIVERY_REWARD = 20;
OvercookedGridworld.ORDER_TYPES = ObjectState.SOUP_TYPES + ['any'];
OvercookedGridworld.num_items_for_soup = 3;

let str_to_array = (val) => {
    if (Array.isArray(val)) {
        return val
    }
    return val.split(',').map((i) => parseInt(i))
};

// let assert = function (bool, msg) {
//     if (typeof(msg) === 'undefined') {
//         msg = "Assert Failed";
//     }
//     if (bool) {
//         return
//     }
//     else {
//         console.log(msg);
//         console.trace();
//     }
// };