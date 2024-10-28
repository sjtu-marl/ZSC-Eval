import $ from "jquery"
import _ from "lodash"

import OvercookedSinglePlayerTask from "./js/overcooked-single";
import getOvercookedPolicy from "./js/load_tf_model.js";

import * as Overcooked from "overcooked"
let OvercookedMDP = Overcooked.OvercookedMDP;
let Direction = OvercookedMDP.Direction;
let Action = OvercookedMDP.Action;
let [NORTH, SOUTH, EAST, WEST] = Direction.CARDINAL;
let [STAY, INTERACT] = [Direction.STAY, Action.INTERACT];

// Parameters
let PARAMS = {
  MAIN_TRIAL_TIME: 15, //seconds
  TIMESTEP_LENGTH: 187.5, //milliseconds
  DELIVERY_POINTS: 20,
  PLAYER_INDEX: 1,  // Either 0 or 1
  MODEL_TYPE: 'ppo_bc'  // Either ppo_bc, ppo_sp, or pbt
};

/***********************************
      Main trial order
************************************/


let layouts = {
  "cramped_room": [
    "XXPXX",
    "O  2O",
    "X1  X",
    "XDXSX"
  ],
  "asymmetric_advantages": [
    "XXXXXXXXX",
    "O XSXOX S",
    "X   P 1 X",
    "X2  P   X",
    "XXXDXDXXX"
  ],
  "coordination_ring": [
    "XXXPX",
    "X 1 P",
    "D2X X",
    "O   X",
    "XOSXX"
  ],
  "forced_coordination": [
    "XXXPX",
    "O X1P",
    "O2X X",
    "D X X",
    "XXXSX"
  ],
  "counter_circuit": [
    "XXXPPXXX",
    "X      X",
    "D XXXX S",
    "X2    1X",
    "XXXOOXXX"
  ],
  "fork": [
    "XXXXXXXXXXXXXXX",
    "XXXXX  X    XPX",
    "O1    XXXXX  2S",
    "XXXXX  X    XPX",
    "XXXXXXXXXXXXXXX"
  ]
};

let agent_settings = getAgentSettings()
var agent_type = 0
var playerIndexList = []
// should be in line with app.py

let game;
let agentTypeSelector;

document.addEventListener("DOMContentLoaded", function () {
  // agentTypeSelector = document.getElementById("agent_type");
  // agentTypeSelector.addEventListener('change', event => {
  //   changeModel((0, _jquery.default)("#agent_type").val());
  // });
  var userInfo = sessionStorage.getItem('before_game')
  if (!userInfo) {
    // window.location.href = "/html/before_game"
    // console.log("jump to statement")
    window.location.href = "/html/statement"
    return;
  }
  // playerIndexList = JSON.parse(sessionStorage.getItem('playerIndexList')) || []
  // agent_type = playerIndexList.length ? getRandomAgent() : 0
  agent_type = getAgentType()
  // show agent_type and layout on the page
  console.log("agent_type:" + agent_type)
  console.log("agent_settings length:" + agent_settings.length)
  if (agent_type >= agent_settings.length) {
    window.location.href = "/html/after_game";
    return;
  }

  let layout_saved = agent_settings[agent_type]['layout']
  if (agent_type === -1) {
    $("#agent_type").text('Warmup trial')
    $("#agent_type_in_a_layout").text(`Warmup trial`)
  } else {
    $("#agent_type").append(`${agent_type} of ${agent_settings.length - 1}`)
    $("#agent_type_in_a_layout").text(`${agent_settings[agent_type]['run_id']} of ${agent_settings[agent_type]['n_runs']}`)
  }
  // console.log("qqq", agent_settings[agent_type]['layout'])
  // 
  changeModel(agent_type);
  $("#layout").text(layout_saved)
  // console.log(paid)
  // console.log(layout_saved)
  let paid_bounds = {
    "random1": "200",
    "random3": "160",
    "unident_s": "400",
    "simple": "160",
    "random0": "60",
  }
  if (agent_type === 0) {
    document.getElementById("trail").style.display = "block"
    document.getElementById("bonus").style.display = "none"
  }
  let paids = paid_bounds[layout_saved]

  // (0, _jquery.default)("#paid").text(paids)
  let x = document.getElementById("paid")
  x.innerHTML = paids
});

function sleep(time) {
  var timeStamp = new Date().getTime();
  var endTime = timeStamp + time;
  while (true) {
    if (new Date().getTime() > endTime) {
      return;
    }
  }
}
function startGame(endOfGameCallback) {
  let AGENT_INDEX = 1 - PARAMS.PLAYER_INDEX;
  /***********************************
        Set up websockets server
  ***********************************/
  // let HOST = "https://lit-mesa-15330.herokuapp.com/".replace(/^http/, "ws");
  // let gameserverio = new GameServerIO({HOST});

  // let players = [$("#playerZero").val(), $("#playerOne").val()];
  // let players = agent_settings[(0, _jquery.default)("#agent_type").val()];
  let players = agent_settings[agent_type].agents
  // console.log(players);
  // let algo = $("#algo").val();
  let algo = agent_settings[agent_type].algo
  let deterministic = $("#deterministic").is(':checked');
  let saveTrajectory = $("#saveTrajectories").is(':checked');
  if (players[0] == 'human' && players[1] == 'human') {
    $("#overcooked").html("<h3>Sorry, we can't support humans as both players.  Please make a different dropdown selection and hit Enter!</h3>");
    endOfGameCallback();
    return;
  }
  let player_index = null;
  let npc_policies = [0, 1];
  if (players[0] == 'human') {
    player_index = 0;
    npc_policies = [1];
  }
  if (players[1] == 'human') {
    player_index = 1;
    npc_policies = [0];
  }
  // let layout_name = (0, _jquery.default)("#layout").val();
  let layout_name = getLayoutName(agent_type, agent_settings)
  let game_time = $("#gameTime").val();

  if (agent_type > 0) {
    game_time = 90;
    $("#gameTime").val(game_time);
  } else {  // tutorial
    game_time = 90;
    $("#gameTime").val(game_time);
  }

  if (game_time > 9999) {
    $("#overcooked").html("<h3>Sorry, please choose a shorter amount of time for the game!</h3>");
    endOfGameCallback();
    return;
  }

  // let layout = layouts[layout_name];

  let [grid, base_layout_params] = get_layout_params(agent_settings);


  // console.log(layout);
  // console.log(layout_name);

  $("#overcooked").empty();
  let mdp_params = {
    "layout_name": layout_name,
    "num_items_for_soup": 3,
    "rew_shaping_params": null
  };



  game = new OvercookedSinglePlayerTask({
    container_id: "overcooked",
    player_index: player_index,
    layout_name: layout_name,
    start_grid : grid,
    base_layout_params : base_layout_params,
    npc_policies: npc_policies,
    algo: algo,
    mdp_params: mdp_params,
    task_params: PARAMS,
    save_trajectory: saveTrajectory,
    TIMESTEP: PARAMS.TIMESTEP_LENGTH,
    MAX_TIME: game_time,
    //seconds
    init_orders: null,
    completion_callback: () => {
      // console.log("Time up");
      endOfGameCallback();
      // 记录本次Agent
      // setRandomAgent()
      setAgentType(agent_type)
      // 在这里跳转并填写问卷
      sleep(3000);
      if (agent_type === 0) {
        window.location.href = "/"
      } else {
        window.location.href = "/html/in_game"
      }
    },
    DELIVERY_REWARD: PARAMS.DELIVERY_POINTS
  });
  game.init();
}
function endGame() {
  game.close();
}

// Handler to be added to $(document) on keydown when a game is not in
// progress, that will then start the game when Enter is pressed.
function startGameOnEnter(e) {
  // Do nothing for keys other than Enter
  if (e.which !== 13) {
    return;
  }
  disableEnter();
  // Reenable enter handler when the game ends
  startGame(enableEnter);
}
function enableEnter() {
  $(document).keydown(startGameOnEnter);
  $("#control").html("<p>Press enter to begin!</p><p>(make sure you've deselected all input elements first!)</p>");
}
function disableEnter() {
  $(document).off("keydown");
  $("#control").html('');
  // (0, _jquery.default)("#control").html('<button id="reset" type="button" class="btn btn-primary">Reset</button>');
  // (0, _jquery.default)("#reset").click(endGame);
}
function getDomData() {
  var userInfo = JSON.parse(sessionStorage.getItem('before_game')) || {}
  var params = {
    name: userInfo.name,
    phone: userInfo.phone,
  }
  // console.log('userInfo', params, userInfo)
  return params
}
function changeModel(agent_type) {
  // console.log("Changed Model to: " + agent_type);
  var infodata = getDomData()
  var xhr = new XMLHttpRequest();
  xhr.open("POST", "/changemodel", false); // false for synchronous
  xhr.setRequestHeader('Content-Type', 'application/json');
  var game_setting_list = getAgentSettings()
  xhr.send(JSON.stringify({
    agent_type: agent_type,
    user_info: infodata,
    game_settings: game_setting_list,
  }));
  var response = JSON.parse(xhr.response);
  // console.log("response:");
  // console.log(response);
}
function getRandomAgent() {
  var agent = Math.floor(Math.random() * 25) + 1
  if (playerIndexList.includes(agent)) {
    agent = getRandomAgent()
  }
  // console.log('agent', agent, playerIndexList)

  // [1, 25]
  return agent
}

function setRandomAgent() {
  if (playerIndexList.length >= 26) {
    return
  }
  var list = playerIndexList.concat(agent_type)
  // console.log('list', list)
  sessionStorage.setItem('playerIndexList', JSON.stringify(list));
}

function getAgentType() {
  let agent_type = sessionStorage.getItem('agent_type')
  let level = agent_type ? Number(agent_type) + 1 : 0
  return level
}

function setAgentType(agent_type) {
  if (agent_type > agent_settings.length - 1) {
    return
  }
  sessionStorage.setItem('agent_type', agent_type);
}

function getAgentSettings() {
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

function getLayoutName(agent_type, agent_settings) {
  let layoutMap = {
    "simple": "cramped_room",
    "unident_s": "asymmetric_advantages",
    "random1": "coordination_ring",
    "random0": "forced_coordination",
    "random3": "counter_circuit",
  }
  let layout = agent_settings[agent_type].layout;
  if (layout in layoutMap){
    return layoutMap[layout];
  }else{
    return layout;
  }
}

function get_layout_params(game_setting_list) {

  let base_layout_params = game_setting_list[0]["base_layout_params"];

  let grid = base_layout_params["grid"];
  delete base_layout_params["grid"];
  // base_layout_params["layout"] = layout_name;
  if (base_layout_params["start_state"]) {
      base_layout_params["start_state"] = OvercookedState.from_dict(base_layout_params["start_state"]);
  }
  // Clean grid
  grid = grid.split("\n").map(layout_row => layout_row.trim());

  // Convert Objects to Arrays
  var toString = Object.prototype.toString;
  base_layout_params["start_bonus_orders"] = Object.values(base_layout_params["start_bonus_orders"]);
  console.log(toString.call(base_layout_params["start_bonus_orders"]));
  /*
  for(let i = 0; i < base_layout_params["start_all_orders"].length; i++){
    base_layout_params["start_all_orders"][i] = Object.entries(base_layout_params["start_all_orders"][i]);
    console.log(base_layout_params["start_all_orders"]);
  }
  */
  return [grid, base_layout_params];
}

$(document).ready(() => {
  enableEnter();
});



