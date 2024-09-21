import Phaser from 'phaser'
import * as OvercookedMDP from "./mdp.es6"

let Direction = OvercookedMDP.Direction;

export class OvercookedGame {
    constructor({
        layout_name,

        container_id,

        start_grid,
        base_layout_params,

        tileSize = 128,

        gameWidth = tileSize * start_grid[0].length,
        gameHeight = tileSize * start_grid.length,

        ANIMATION_DURATION = 500,
        TIMESTEP_DURATION = 600,
        player_colors = { 0: 'green', 1: 'blue' },
        assets_loc = "./assets/",
        layouts_loc = "./layouts",
        show_post_cook_time = false,
        old_dynamics = false,

        COOK_TIME = 2,
        DELIVERY_REWARD = OvercookedMDP.OvercookedGridworld.DELIVERY_REWARD,
        num_items_for_soup = 3
    }) {
        
        this.container_id = container_id;
        let params = { COOK_TIME, DELIVERY_REWARD, num_items_for_soup };

        let mdp_params = { "layout_name": layout_name, "start_order_list": null };
        let rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        };

        // MARK: use reward shaping
        mdp_params = {
            ...mdp_params,
            "rew_shaping_params": rew_shaping_params,
            "old_dynamics": old_dynamics
        };
        
        
        this.mdp = OvercookedMDP.OvercookedGridworld.from_grid(start_grid, base_layout_params, mdp_params);

        this.gameWidth = gameWidth;
        this.gameHeight = gameHeight;

        // this.mdp = OvercookedMDP.OvercookedGridworld.from_grid(start_grid, params);
        this.state = this.mdp.get_standard_start_state();
        this.joint_action = [OvercookedMDP.Direction.STAY, OvercookedMDP.Direction.STAY];
        this.player_colors = player_colors;
        this.assets_loc = assets_loc;
        this.layouts_loc = layouts_loc;
        this.show_post_cook_time = show_post_cook_time; 

        let gameparent = this;
        this.scene = new Phaser.Class({
            gameparent,
            Extends: Phaser.Scene,
            initialize: function() {
                Phaser.Scene.call(this, {key: "PlayGame"})
            },
            preload: function () {
                this.load.atlas("tiles",
                    this.gameparent.assets_loc+"terrain.png",
                    this.gameparent.assets_loc+"terrain.json");
                this.load.atlas("chefs",
                    this.gameparent.assets_loc+"chefs.png",
                    this.gameparent.assets_loc+"chefs.json");
                this.load.atlas("objects",
                    this.gameparent.assets_loc+"objects.png",
                    this.gameparent.assets_loc+"objects.json");
                this.load.atlas("soups",
                    this.gameparent.assets_loc+"soups.png",
                    this.gameparent.assets_loc+"soups.json");
                
            },
            create: function () {
                // this.gameparent = gameparent;
                this.mdp = gameparent.mdp;
                this.sprites = {};
                this.drawLevel();
                this._drawState(gameparent.state, this.sprites);
                // this.cursors = this.input.keyboard.createCursorKeys(); //this messes with the keys a lot
                // this.player.can_take_input = true;
                // this.animating_transition = false;
            },
            drawLevel: function() {
                //draw tiles
                let terrain_to_img = {
                    ' ': 'floor.png',
                    'X': 'counter.png',
                    'P': 'pot.png',
                    'O': 'onions.png',
                    'T': 'tomatoes.png',
                    'D': 'dishes.png',
                    'S': 'serve.png'
                };
                let pos_dict = this.mdp._get_terrain_type_pos_dict();
                for (let ttype in pos_dict) {
                    if (!pos_dict.hasOwnProperty(ttype)) {continue}
                    for (let i = 0; i < pos_dict[ttype].length; i++) {
                        let [x, y] = pos_dict[ttype][i];
                        let tile = this.add.sprite(
                            tileSize * x,
                            tileSize * y,
                            "tiles",
                            terrain_to_img[ttype]
                        );
                        tile.setDisplaySize(tileSize, tileSize);
                        tile.setOrigin(0);
                    }
                }

            },
            _drawState: function (state, sprites) {
                sprites = typeof(sprites) === 'undefined' ? {} : sprites;

                //draw chefs
                sprites['chefs'] =
                    typeof(sprites['chefs']) === 'undefined' ? {} : sprites['chefs'];
                for (let pi = 0; pi < state.players.length; pi++) {
                    let chef = state.players[pi];
                    let [x, y] = chef.position;
                    let dir = Direction.DIRECTION_TO_NAME[chef.orientation];
                    let held_obj = chef.held_object;
                    let held_object_name = ""
                    if (held_obj !== null) {
                        if (held_obj.name === 'soup') {
                            if(held_obj.ingredients.includes("onion")){
                                held_object_name = "-soup-onion"
                            }else{
                                held_object_name = "-soup-tomato"
                            }
                        }
                        else {
                            held_object_name = "-"+held_obj.name;
                        }
                    }

                    if (typeof(sprites['chefs'][pi]) === 'undefined') {
                        let chefsprite = this.add.sprite(
                            tileSize*x,
                            tileSize*y,
                            "chefs",
                            `${dir}${held_object_name}.png`
                        );
                        chefsprite.setDisplaySize(tileSize, tileSize);
                        chefsprite.depth = 1;
                        chefsprite.setOrigin(0);
                        let hatsprite = this.add.sprite(
                            tileSize*x,
                            tileSize*y,
                            "chefs",
                            `${dir}-${this.gameparent.player_colors[pi]}hat.png`
                        );
                        hatsprite.setDisplaySize(tileSize, tileSize);
                        hatsprite.depth = 2;
                        hatsprite.setOrigin(0);
                        sprites['chefs'][pi] = {chefsprite, hatsprite};
                    }
                    else {
                        let chefsprite = sprites['chefs'][pi]['chefsprite'];
                        let hatsprite = sprites['chefs'][pi]['hatsprite'];
                        chefsprite.setFrame(`${dir}${held_object_name}.png`);
                        hatsprite.setFrame(`${dir}-${this.gameparent.player_colors[pi]}hat.png`);
                        this.tweens.add({
                            targets: [chefsprite, hatsprite],
                            x: tileSize*x,
                            y: tileSize*y,
                            duration: ANIMATION_DURATION,
                            ease: 'Linear',
                            onComplete: (tween, target, player) => {
                                target[0].setPosition(tileSize*x, tileSize*y);
                                //this.animating = false;
                            }
                        })
                    }
                }

                //draw environment objects
                if (sprites['objects'] !== undefined) {
                    for (let objpos in sprites.objects) {
                        let {objsprite, timesprite} = sprites.objects[objpos];
                        objsprite.destroy();
                        if (timesprite !== undefined) {
                            timesprite.destroy();
                        }
                    }
                }
                if (sprites['soups'] !== undefined) {
                    for (let souppos in sprites.soups) {
                        let {objsprite, timesprite} = sprites.soups[souppos];
                        objsprite.destroy();
                        if (timesprite !== undefined) {
                            timesprite.destroy();
                        }
                    }
                }
                sprites['objects'] = {};
                sprites["soups"] = {};

                for (let objpos in state.objects) {
                    if (!state.objects.hasOwnProperty(objpos)) { continue }
                    let obj = state.objects[objpos];
                    let [x, y] = obj.position;
                    let terrain_type = this.mdp.get_terrain_type_at(obj.position);
                    let spriteframe, souptype, n_ingredients;
                    let cooktime = "";
                    if ((obj.name === 'soup') && (terrain_type === 'P')) {

                        cooktime = obj._cooking_tick;
                        
                        let soup_status = obj.is_ready ? "cooked" : "idle";

                        let num_onions = obj.ingredients.filter(item => item == "onion").length;
                        let num_tomatoes = obj.ingredients.filter(item => item == "tomato").length;

                        let frame_name = `soup_${soup_status}_tomato_${num_tomatoes}_onion_${num_onions}`

                        let obj_or_soup;
                        // select pot sprite
                        if (cooktime >= this.mdp.explosion_time) {
                            obj_or_soup = "objects";
                            spriteframe = 'pot-explosion.png';
                        }
                        else {
                            obj_or_soup = "soups";
                            spriteframe = `${frame_name}.png`;
                        }

                        
                        let objsprite = this.add.sprite(
                            tileSize*x,
                            tileSize*y,
                            obj_or_soup,
                            spriteframe
                        );
                        objsprite.setDisplaySize(tileSize, tileSize);
                        objsprite.depth = 1;
                        objsprite.setOrigin(0);
                        let objs_here = {objsprite};

                        // show time accordingly
                        let show_time = true;
                        
                        if ((cooktime == -1) && (cooktime > this.mdp.COOK_TIME) && !this.show_post_cook_time) {
                            show_time = false;
                        }
                        if (show_time) {
                            let timesprite =  this.add.text(
                                tileSize*(x+.5),
                                tileSize*(y+.6),
                                String(cooktime),
                                {
                                    font: "25px Arial",
                                    fill: "red",
                                    align: "center",
                                }
                            );
                            timesprite.depth = 2;
                            objs_here['timesprite'] = timesprite;
                        }

                        sprites[obj_or_soup][objpos] = objs_here
                    }
                    else if (obj.name === 'soup') {
                        
                        let soup_status = "done";

                        let num_onions = obj.ingredients.filter(item => item == "onion").length;
                        let num_tomatoes = obj.ingredients.filter(item => item == "tomato").length;

                        let frame_name = `soup_${soup_status}_tomato_${num_tomatoes}_onion_${num_onions}`

                        spriteframe = `${frame_name}.png`;

                        let objsprite = this.add.sprite(
                            tileSize*x,
                            tileSize*y,
                            "soups",
                            spriteframe
                        );
                        objsprite.setDisplaySize(tileSize, tileSize);
                        objsprite.depth = 1;
                        objsprite.setOrigin(0);
                        sprites['soups'][objpos] = {objsprite};
                    }
                    else {
                        if (obj.name === 'onion') {
                            spriteframe = "onion.png";
                        }
                        else if (obj.name === 'tomato') {
                            spriteframe = "tomato.png";
                        }
                        else if (obj.name === 'dish') {
                            spriteframe = "dish.png";
                        }
                        let objsprite = this.add.sprite(
                            tileSize*x,
                            tileSize*y,
                            "objects",
                            spriteframe
                        );
                        objsprite.setDisplaySize(tileSize, tileSize);
                        objsprite.depth = 1;
                        objsprite.setOrigin(0);
                        sprites['objects'][objpos] = {objsprite};
                    }
                }

                //draw order list
                let order_array = state.all_orders.map(order => "["+order.ingredients.join(", ")+"]");
                let order_list = "Orders: "+order_array.join(", ");
                if (typeof(sprites['order_list']) !== 'undefined') {
                    sprites['order_list'].setText(order_list);
                }
                else {
                    sprites['order_list'] = this.add.text(
                        5, 5, order_list,
                        {
                            font: "15px Arial",
                            fill: "yellow",
                            align: "left"
                        }
                    )
                }
            },
            _drawScore: function(score, sprites) {
                score = "Score: "+score;
                if (typeof(sprites['score']) !== 'undefined') {
                    sprites['score'].setText(score);
                }
                else {
                    sprites['score'] = this.add.text(
                        5, 25, score,
                        {
                            font: "15px Arial",
                            fill: "yellow",
                            align: "left"
                        }
                    )
                }
            },
            _drawTimeLeft: function(time_left, sprites) {
                time_left = "Time Left: "+time_left;
                if (typeof(sprites['time_left']) !== 'undefined') {
                    sprites['time_left'].setText(time_left);
                }
                else {
                    sprites['time_left'] = this.add.text(
                        5, 45, time_left,
                        {
                            font: "10px Arial",
                            fill: "yellow",
                            align: "left"
                        }
                    )
                }
            },
            update: function() {
                // let state, score_;
                // let redraw = false;
                if (typeof(this.gameparent.state_to_draw) !== 'undefined') {
                    let state = this.gameparent.state_to_draw;
                    delete this.gameparent.state_to_draw;
                    // redraw = true;
                    this._drawState(state, this.sprites);
                }
                if (typeof(this.gameparent.score_to_draw) !== 'undefined') {
                    let score = this.gameparent.score_to_draw;
                    delete this.gameparent.score_to_draw;
                    this._drawScore(score, this.sprites);
                }
                if (typeof(this.gameparent.time_left) !== 'undefined') {
                    let time_left = this.gameparent.time_left;
                    delete this.gameparent.time_left;
                    this._drawTimeLeft(time_left, this.sprites);
                }
                // if (!redraw) {
                //     return
                // }

            }
        });
    }

    init () {
        let gameConfig = {
            type: Phaser.WEBGL,
            width: this.gameWidth,
            height: this.gameHeight,
            scene: [this.scene],
            parent: this.container_id,
            pixelArt: true,
            audio: {
                noAudio: true
            }
        };
        this.game = new Phaser.Game(gameConfig);
    }

    drawState(state) {
        this.state_to_draw = state;
    }

    setAction(player_index, action) {

    }

    drawScore(score) {
        this.score_to_draw = String(score);
    }

    drawTimeLeft(time_left) {
        this.time_left = String(time_left);
    }

    close (msg) {
        this.game.renderer.destroy();
        this.game.loop.stop();
        // this.game.canvas.remove();
        this.game.destroy();
    }

}