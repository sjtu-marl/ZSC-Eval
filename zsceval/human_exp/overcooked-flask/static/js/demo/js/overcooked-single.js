import * as Overcooked from "overcooked"
let OvercookedGame = Overcooked.OvercookedGame.OvercookedGame;
let OvercookedMDP = Overcooked.OvercookedMDP;
let Direction = OvercookedMDP.Direction;
let Action = OvercookedMDP.Action;
let [NORTH, SOUTH, EAST, WEST] = Direction.CARDINAL;
let [STAY, INTERACT] = [Action.STAY, Action.INTERACT];

let COOK_TIME = 20;

function getDomData() {
    var userInfo = JSON.parse(sessionStorage.getItem('before_game')) || {}
    var params = {
        name: userInfo.name,
        phone: userInfo.phone,
    }
    // console.log('userInfo', params, userInfo)
    return params
}
export default class OvercookedSinglePlayerTask {
    constructor({
        container_id,
        player_index,
        npc_policies,
        mdp_params,
        task_params,
        algo,
        start_grid,
        base_layout_params,
        layout_name,
        save_trajectory = false,
        TIMESTEP = 200,
        MAX_TIME = 20, //seconds
        init_orders = null,
        completion_callback = () => { console.log("Time up") },
        timestep_callback = (data) => { },
        DELIVERY_REWARD = 20,
        SHAPED_INFOS = ["put_onion_on_X",
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
                        "IDLE_INTERACT",]
    }) {
        //NPC policies get called at every time step
        if (typeof (npc_policies) === 'undefined') {
            // TODO maybe delete this? 
            npc_policies = {
                1:
                    (function () {
                        let action_loop = [
                            SOUTH, WEST, NORTH, EAST
                        ];
                        let ai = 0;
                        let pause = 4;
                        return (s) => {
                            let a = STAY;
                            if (ai % pause === 0) {
                                a = action_loop[ai / pause];
                            }
                            ai += 1;
                            ai = ai % (pause * action_loop.length);
                            return a
                        }
                    })()
            }
        }
        this.npc_policies = npc_policies;
        this.player_index = player_index;
        this.algo = algo;

        let player_colors = this.getHatColor();

        this.game = new OvercookedGame({
            layout_name,
            container_id,
            start_grid : start_grid,
            base_layout_params : base_layout_params,
            assets_loc: "static/assets/",
            layout_loc: "static/layouts/",
            ANIMATION_DURATION: TIMESTEP * 0.2,
            tileSize: 80,
            COOK_TIME: COOK_TIME,
            explosion_time: Number.MAX_SAFE_INTEGER,
            DELIVERY_REWARD: DELIVERY_REWARD,
            player_colors: player_colors,
        });
        this.init_orders = init_orders;
        if (Object.keys(npc_policies).length == 1) {
            // console.log("Single human player vs agent");
            this.game_type = 'human_vs_agent';
        }
        else {
            // console.log("Agent playing vs agent")
            this.game_type = 'agent_vs_agent';
        }

        this.layout_name = layout_name;
        this.TIMESTEP = TIMESTEP;
        this.MAX_TIME = MAX_TIME;
        this.time_left = MAX_TIME;
        this.cur_gameloop = 0;
        this.score = 0;
        this.completion_callback = completion_callback;
        this.timestep_callback = timestep_callback;
        this.mdp_params = mdp_params;
        this.mdp_params['cook_time'] = COOK_TIME;
        this.mdp_params['start_order_list'] = init_orders;
        this.task_params = task_params;
        this.save_trajectory = save_trajectory
        this.trajectory = {
            'ep_states': [[]],
            'ep_actions': [[]],
            'ep_rewards': [[]],
            'mdp_params': [mdp_params]
        }
    }

    init() {
        this.game.init();

        this.start_time = -1;

        this.state = this.game.mdp.get_standard_start_state();
        this.game.drawState(this.state);
        this.joint_action = [STAY, STAY];
        // this.lstm_state = [null, null];
        this.done = 1;
        var agent_settings = this.getAgentSettings()
        var agent_type = this.getAgentType()
        // console.log('agent_settings', agent_settings[agent_type])
        var player_agents = agent_settings[agent_type]['agents']
        var ai = player_agents[0] === 'human' ? player_agents[1] : player_agents[0]
        var layout = agent_settings[agent_type].layout
        this.gameloop = setInterval(() => {
            for (const npc_index of this.npc_policies) {
                // let [npc_a, lstm_state] = this.npc_policies[npc_index](this.state, this.done, this.lstm_state[npc_index], this.game);

                var xhr = new XMLHttpRequest();
                // let predicStr = '/predict/'
                let url = agent_settings[agent_type].url
                // let new_npc_index = !url ?  npc_index : str.slice(url.length - predicStr.length + 1, url.length - predicStr.length)
                // let apiUrl = url || "/predict"
                xhr.open("POST", url, false);
                // xhr.open("POST", "/" + layout + "/" + ai +"/predict/", false); // false for synchronous
                xhr.setRequestHeader('Content-Type', 'application/json');
                var userinfo = getDomData();
                xhr.send(JSON.stringify({
                    state: this.state,
                    npc_index: player_agents[0] === 'human' ? 1 : 0,
                    layout_name: this.layout_name,
                    algo: this.algo,
                    timestep: this.cur_gameloop,
                    user_info: userinfo,
                }));
                var action_idx = JSON.parse(xhr.responseText)["action"];
                let npc_a = Action.INDEX_TO_ACTION[action_idx];
                // console.log(npc_a);

                // this.lstm_state[npc_index] = lstm_state;
                this.joint_action[npc_index] = npc_a;

                if (this.start_time == -1) { // see above
                    this.start_time = new Date().getTime();
                }

            }
            this.joint_action_idx = [Action.ACTION_TO_INDEX[this.joint_action[0]], Action.ACTION_TO_INDEX[this.joint_action[1]]];
            let [[next_state, prob], infos] = this.game.mdp.get_transition_states_and_probs({
                state: this.state,
                joint_action: this.joint_action
            });
            let reward = infos["sparse_reward_by_agent"][0] + infos["sparse_reward_by_agent"][1]; 


            // Apparently doing a Parse(Stringify(Obj)) is actually the most succinct way. 
            // to do a deep copy in JS 
            // let cleanedState = JSON.parse(JSON.stringify(this.state));
            // cleanedState['objects'] = Object.values(cleanedState['objects']);

            const _lodash = require("lodash");

            this.trajectory.ep_states[0].push(JSON.stringify(this.state))
            this.trajectory.ep_actions[0].push(JSON.stringify(this.joint_action_idx))
            this.shaped_infos = _lodash.cloneDeep(infos.shaped_info_by_agent[this.player_index]);
            this.trajectory.ep_rewards[0].push(reward)
            //update next round
            this.game.drawState(next_state);
            this.score += reward;
            this.game.drawScore(this.score);
            let time_elapsed = (new Date().getTime() - this.start_time) / 1000;
            this.time_left = Math.round(this.MAX_TIME - time_elapsed);
            this.game.drawTimeLeft(this.time_left);
            this.done = 0

            //record data
            this.timestep_callback({
                state: this.state,
                joint_action: this.joint_action,
                next_state: next_state,
                reward: reward,
                time_left: this.time_left,
                score: this.score,
                time_elapsed: time_elapsed,
                cur_gameloop: this.cur_gameloop,
                client_id: undefined,
                is_leader: undefined,
                partner_id: undefined,
                datetime: +new Date()
            });

            //set up next timestep
            this.state = next_state;
            this.joint_action = [STAY, STAY];
            this.cur_gameloop += 1;
            this.activate_response_listener();

            //time run out
            if (this.time_left < 0) {
                this.time_left = 0;
                this.close();
            }
        }, this.TIMESTEP);
        this.activate_response_listener();
    }

    close() {
        if (typeof (this.gameloop) !== 'undefined') {
            clearInterval(this.gameloop);
        }

        var today = new Date();
        var traj_time = (today.getMonth() + 1) + '_' + today.getDate() + '_' + today.getFullYear() + '_' + today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds();
        let trajectory = this.trajectory;
        let task_params = this.task_params;

        // Looks like all the internal objects are getting stored as strings rather than actual arrays or objects
        // So it looks like Bodyparser only parses the top levl keys, and keeps everything on the lower level as strings rather 
        // than processing it recursively 

        let parsed_trajectory_data = {
            "ep_states": [[]],
            "ep_rewards": [[]],
            "ep_actions": [[]],
            "mdp_params": []
        }

        parsed_trajectory_data['mdp_params'][0] = trajectory.mdp_params[0];
        ["ep_states", "ep_rewards", "ep_actions"].forEach(function (key, key_index) {
            trajectory[key][0].forEach(function (item, index) {
                parsed_trajectory_data[key][0].push(JSON.parse(item))
            })
        })

        document.getElementById('control').innerHTML = "Updating model...please wait...";
        setTimeout(() => {
            // make dom finishes rendering
            var xhr = new XMLHttpRequest();
            xhr.open("POST", "/finish_episode", false); // false for synchronous
            xhr.setRequestHeader('Content-Type', 'application/json');
            // console.log(this.algo)
            xhr.send(JSON.stringify({
                traj_id: traj_time + "_human=" + this.player_index,
                traj: parsed_trajectory_data,
                layout_name: this.layout_name,
                shaped_infos: this.shaped_infos,
                algo: this.algo,
                agent_type: Number(sessionStorage.getItem('agent_type')),
                user_info: getDomData(),
            }));
            // 前端存session
            this.setQuesData(traj_time + "_human=" + this.player_index)
            var status = JSON.parse(xhr.responseText);
            // console.log("status: " + status);

            this.game.close();
            this.disable_response_listener();
            this.completion_callback();

        }, 15);
    }

    activate_response_listener() {
        $(document).on("keydown", (e) => {
            let action;
            switch (e.which) {
                case 37: // left
                    action = WEST;
                    break;

                case 38: // up
                    action = NORTH;
                    break;

                case 39: // right
                    action = EAST;
                    break;

                case 40: // down
                    action = SOUTH;
                    break;

                case 32: //space
                    action = INTERACT;
                    break;

                default: return; // exit this handler for other keys
            }
            e.preventDefault(); // prevent the default action (scroll / move caret)

            this.joint_action[this.player_index] = action;
            this.disable_response_listener();
        });
    }

    disable_response_listener() {
        $(document).off('keydown');
    }

    getAgentType() {
        let agent_type = sessionStorage.getItem('agent_type')
        let level = agent_type ? Number(agent_type) + 1 : 0
        return level
    }
    getAgentSettings() {
        let agent_settings = []
        try {
            let storedSettings = sessionStorage.getItem('game_setting_list')
            if (storedSettings) {
                agent_settings = JSON.parse(storedSettings)
            }
        } catch (e) {
            console.error("Error parsing game_setting_list from sessionStorage", e)
        }
        return agent_settings
    }

    setQuesData(traj_id) {
        var inGameList = JSON.parse(sessionStorage.getItem('in_game')) || []
        // var agent_type = $("#agent_type").text()
        var agent_type = this.getAgentType()
        // if (agent_type === 'Gameplay trial level') {
        //   agent_type = 0
        // }
        inGameList.push({
            traj_id,
            agent_type,
            questionnaire: {}
        })
        sessionStorage.setItem('in_game', JSON.stringify(inGameList))
    }
    getHatColor() {
        // console.log('this', this, agent_settings)
        let game_setting_list = sessionStorage.getItem('game_setting_list')

        var agent_settings = this.getAgentSettings()
        // var agent_type = $("#agent_type").text()
        var agent_type = this.getAgentType()
        // if (agent_type === 'Gameplay trial level') {
        //   agent_type = 0
        // }
        // console.log(agent_settings)
        let players = agent_settings[agent_type]['agents'];
        let ai_type;
        let human_idx, ai_idx;
        if (players[0] == 'human') {
            human_idx = 0;
            ai_idx = 1;
            ai_type = players[1];
        }
        else if (players[1] == 'human') {
            human_idx = 1;
            ai_idx = 0;
            ai_type = players[0];
        }
        else {
            throw ("Unexpected agent type: " + players.toString());
        }
        // let ai_players = game_setting_list[]
        // let colors = { 0: 'white', 1: 'red' };
        let human_color = 'gray';
        // var colorMap = {
        //     'SP': 'blue',
        //     'E3T': 'green',
        //     'COLE': 'orange',
        //     'FCP': 'red',
        //     'MEP': 'purple',
        //     'TrajeDi': 'black',
        //     'HSP': 'yellow',
        // }
        var colorMap = {
            'AGENT_TYPE_A': 'blue',
            'AGENT_TYPE_B': 'green',
            'AGENT_TYPE_C': 'orange',
            'AGENT_TYPE_D': 'red',
            'AGENT_TYPE_E': 'purple',
            'AGENT_TYPE_F': 'black',
            'AGENT_TYPE_G': 'yellow',
        }
        var agent_colors = {};
        agent_colors[human_idx] = human_color;
        agent_colors[ai_idx] = colorMap[ai_type] || 'white';
        return agent_colors
    }
}
