import argparse
import json
import os
import shutil
import random
import time
from collections import defaultdict
from pprint import pformat
from typing import Callable, Dict
import threading


import imageio
import yaml
from flask import Flask, jsonify, request
from flask_cors import CORS
from loguru import logger
from markdown import markdown

from zsceval.envs.overcooked_new.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action

from zsceval.human_exp.overcooked_utils import (
    NAME_TRANSLATION,
    LAYOUT_LIST,
    generate_balanced_permutation,
    read_layout_dict,
)

app = Flask(__name__)
cors = CORS()
cors.init_app(app, resources={r"/*": {"origins": "*"}})
NAME_TRANSLATION_REVERSE = {v: k for k, v in NAME_TRANSLATION.items()}

OLD_LAYOUTS = ["random0","random1","random3",]

ALL_LAYOUTS = ["random3_m"]
#ALL_LAYOUTS = ["random3"]
USER_AGENTS = {}
""" 
{
    username_phone:
        - agent_call
}
"""
HUMAN_NAME = "human"

ARGS = None
AGENT_POOLS = None
ALL_ALGOS = None
CODE = None
CODE_2_ALGO = None
ALGO_2_CODE = None
CURRENT_LAYOUT = None


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--port", type=int, default=int(os.getenv("FLASK_PORT", 8088)), help="port to run flask")
    parser.add_argument("--ip", type=str, default=os.getenv("FLASK_HOST", "localhost"), help="your public network IP")
    parser.add_argument(
        "--access_ip", type=str, default=os.getenv("FLASK_ACCESS_HOST", "localhost"), help="your public network IP"
    )
    parser.add_argument(
        "--population_yml_path",
        type=str,
        default=os.getenv("POPULATION_YML_PATH", "./zsceval/human_exp/configs/benchmark_configs"),
        help="path of policy population config",
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", 1)), help="seed for all randomness")
    parser.add_argument(
        "--trajs_save_path",
        type=str,
        default=os.getenv("TRAJs_SAVE_PATH", "./zsceval/human_exp/data/trajs"),
        help="optional trajectory save path",
    )
    parser.add_argument(
        "--infos_save_path",
        type=str,
        default=os.getenv("INFOs_SAVE_PATH", "./zsceval/human_exp/data/infos"),
        help="optional info save path",
    )
    parser.add_argument(
        "--progress_save_path",
        type=str,
        default=os.getenv("PROGRESS_SAVE_PATH", "./zsceval/human_exp/data/progress.json"),
        help="optional game progress save path",
    )
    parser.add_argument(
        "--questionnaire_save_path",
        type=str,
        default=os.getenv("QUESTIONNAIRE_SAVE_PATH", "./zsceval/human_exp/data/questionnaires"),
        help="optional questionnaire save path",
    )
    parser.add_argument(
        "--gifs_save_path",
        type=str,
        default=os.getenv("GIFs_SAVE_PATH", "./zsceval/human_exp/data/gifs"),
        help="optional info save path",
    )
    return parser


@app.route("/")
def root():
    return app.send_static_file("index_" + "1" + ".html")


@app.route("/html/<page>")
def return_html(page):
    return app.send_static_file(f"{page}.html")


@app.route("/beforegame", methods=["POST"])
def beforegame():
    if request.method == "POST":
        script_path = os.path.abspath(__file__)
        script_directory = os.path.dirname(script_path)
        f = open(os.path.join(script_directory, "..", "configs", "before_game.yaml"), "r", encoding="utf-8")
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


@app.route("/statement", methods=["POST"])
def statement():
    if request.method == "POST":
        script_path = os.path.abspath(__file__)
        script_directory = os.path.dirname(script_path)
        f = open(os.path.join(script_directory, "..", "configs", "statement.md"), "r", encoding="utf-8").read()
        html = markdown(f)
        return html


def init_game_settings(
    random_pos: bool = True,
):
    """
    random_start_index : whether to use randomized starting index in overcooked
    """
    with open(ARGS.progress_save_path, "r", encoding="utf-8") as f:
        progress_dict = json.load(f)
    layout_game_algo_lists = defaultdict(list)
    for g_settings in progress_dict["used_game_settings"]:
        _layout_game_algos = defaultdict(list)
        for g_setting in g_settings:
            if g_setting["algo"] == "dummy":
                continue
            _layout_game_algos[g_setting["layout"]].append(g_setting["algo"])
        for k, v in _layout_game_algos.items():
            layout_game_algo_lists[k].append(v)

    logger.info(f"used layout_algos\n{pformat(dict(layout_game_algo_lists))}")

    game_settings = []

    layout_ids = list(range(len(ALL_LAYOUTS)))
    random.shuffle(layout_ids)

    for layout_i in layout_ids:
        layout = ALL_LAYOUTS[layout_i]
        algos = list(AGENT_POOLS[layout].policy_pool.keys())
        used_algo_lists = layout_game_algo_lists[layout]
        algo_list = generate_balanced_permutation(used_algo_lists, algos)
        for algo_i, algo_control in enumerate(algo_list):
            pos = 0
            algo_control = algo_control
            if random_pos:
                pos = random.choice([0, 1])
            if pos == 0:
                human_algo_pair = [HUMAN_NAME, algo_control]
            else:
                human_algo_pair = [algo_control, HUMAN_NAME]

            #layout_alias = NAME_TRANSLATION_REVERSE[layout] if layout in NAME_TRANSLATION_REVERSE.values() else layout
            layout_alias = layout 

            base_layout_params = read_layout_dict(layout)
            CURRENT_LAYOUT = base_layout_params

            game_settings.append(
                {
                    "agents": human_algo_pair,
                    "algo": algo_control,
                    "layout": layout,
                    "base_layout_params" : base_layout_params,
                    "run_id": algo_i + 1,
                    "n_runs": len(algo_list),
                    "layout_alias": layout_alias,
                    "url": f"http://{ARGS.access_ip}:{ARGS.port}/{algo_control}/predict/",
                }
            )
    return game_settings

@app.route("/randomize_game_settings", methods=["POST"])
def randomize_game_settings():
    if request.method == "POST":
        user_info = json.loads(request.data)
        user_id = f"{user_info['name']}_{user_info['phone']}"
        logger.debug(f"data json {user_info}")
        with open(ARGS.progress_save_path, "r", encoding="utf-8") as f:
            progress_dict = json.load(f)
        if user_id in progress_dict["user_progress"]:
            game_settings, agent_type, code_order = progress_dict["user_progress"][user_id]
        else:
            game_settings, agent_type, code_order = init_game_settings(True), -1, {}
            progress_dict["user_progress"][user_id] = (game_settings, agent_type, code_order)
            progress_dict["used_game_settings"].append(game_settings)
            with open(ARGS.progress_save_path, "w", encoding="utf-8") as f:
                json.dump(progress_dict, f)
        logger.debug(f"Games generate for {user_id}:\n{pformat(game_settings)}, agent_type {agent_type}")
        return jsonify({"game_setting_list": game_settings, "agent_type": agent_type, "algo_order": code_order},)

def render_traj(joint_actions, save_dir, layout_name, traj_id):
    
    try:
        translated_joint_actions = []
        for joint_action in joint_actions[0]:
            
            trans = (Action.INDEX_TO_ACTION[joint_action[0]], Action.INDEX_TO_ACTION[joint_action[1]])
            translated_joint_actions.append(trans)
            
        filename = f"{traj_id}.gif".replace(":", "_")
        traj_dir = os.path.normpath(os.path.join(save_dir, layout_name))
        os.makedirs(traj_dir, exist_ok=True)
        temp_dir = os.path.normpath(os.path.join(traj_dir, "temp"))
        
        StateVisualizer().render_from_actions(translated_joint_actions, layout_name, img_directory_path=temp_dir)
        
        for img_path in os.listdir(temp_dir):
            img_path = temp_dir + "/" + img_path
        imgs = []
        imgs_dir = os.listdir(temp_dir)
        imgs_dir = sorted(imgs_dir, key=lambda x: int(x.split(".")[0]))
        for img_path in imgs_dir:
            img_path = temp_dir + "/" + img_path
            imgs.append(imageio.imread(img_path))
        save_path = traj_dir + f'/{filename}'
        imageio.mimsave(
            save_path,
            imgs,
            duration=0.05,
        )
        logger.info(f"save gifs in {save_path}")
        
        # delete pngs
        imgs_dir = os.listdir(temp_dir)
        for img_path in imgs_dir:
            img_path = temp_dir + "/" + img_path
            if "png" in img_path:
                os.remove(img_path)
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        logger.exception(e)
            
    return 


@app.route("/finish_episode", methods=["POST"])
def finish_episode():
    """
    Save game trajectory to file
    """
    if request.method == "POST":
        # dict_keys(['traj_id', 'traj', 'layout_name', 'shaped_infos', 'algo', 'agent_type', 'user_info'])
        data_json = json.loads(request.data)
        if data_json["algo"] == "dummy":
            return jsonify({"status": True})
        traj_dict, traj_id, server_layout_name = data_json["traj"], data_json["traj_id"], data_json["layout_name"]
        
        agent_infos = data_json["shaped_infos"]
        
        
        layout_name = NAME_TRANSLATION[server_layout_name] if server_layout_name in NAME_TRANSLATION else server_layout_name

        if ARGS.trajs_save_path:
            # Save trajectory (save this to keep reward information)
            filename = f"{traj_id}.json".replace(":", "_")
            save_path = os.path.normpath(os.path.join(ARGS.trajs_save_path, layout_name))
            os.makedirs(save_path, exist_ok=True)
            with open(
                os.path.normpath(os.path.join(save_path, filename)).replace("\\", "/"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(traj_dict, f)
            logger.info(f"saving traj: {traj_id} in {save_path}")
            
        if ARGS.infos_save_path:
            # Save infos
            filename = f"{traj_id}.json".replace(":", "_")
            save_path = os.path.normpath(os.path.join(ARGS.infos_save_path, layout_name))
            os.makedirs(save_path, exist_ok=True)
            with open(
                os.path.normpath(os.path.join(save_path, filename)).replace("\\", "/"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(agent_infos, f)
            logger.info(f"saving info: {traj_id} in {save_path}")
            
        if ARGS.gifs_save_path:
            threading.Thread(target=render_traj, args=(data_json["traj"]["ep_actions"], ARGS.gifs_save_path, layout_name, traj_id)).start()
            #render_traj(data_json["traj"]["ep_actions"], ARGS.gifs_save_path, layout_name, traj_id)

        return jsonify({"status": True})


@app.route("/changemodel", methods=["POST"])
def changemodel():
    """
    Change both agents to allow changing agents on the fly.
    """
    if request.method == "POST":
        data_json = json.loads(request.data)
        user_id = f"{data_json['user_info']['name']}_{data_json['user_info']['phone']}"
        agent_type = int(data_json["agent_type"])
        game_setting = data_json["game_settings"][agent_type]
        logger.info(f"{user_id} with {game_setting['agents']}")
        if game_setting["algo"] != "dummy":
            agent_call = AGENT_POOLS[game_setting["layout"]].get_agent(game_setting["algo"])
            USER_AGENTS[user_id] = agent_call

        result = {"status": True}

        return jsonify(result)


@app.route("/create_questionnaire_before_game", methods=["POST"])
def create_questionnaire_before_game():
    """
    {
        "name": "Bob",
        "gender": "male",
        "phone": "123456",
        "email":"abc@gmail.com",
        "age" : 20
    }
    Returns:

    """
    os.makedirs(ARGS.questionnaire_save_path, exist_ok=True)
    data_json = json.loads(request.data)
    user_id = f"{data_json.get('name')}_{data_json.get('phone')}"
    questionnaire_path = os.path.join(ARGS.questionnaire_save_path, f"{user_id}.json")
    if os.path.exists(questionnaire_path):
        with open(f"{questionnaire_path}", "r", encoding="utf-8") as f:
            prev_data_json: Dict = json.load(f)
        data_json |= prev_data_json
        with open(ARGS.progress_save_path, "r", encoding="utf-8") as f:
            progress_dict = json.load(f)
            logger.info(f"user {user_id} resume run {pformat(progress_dict['user_progress'][user_id][1])}")
    with open(f"{questionnaire_path}", "w", encoding="utf-8") as f:
        f.write(json.dumps(data_json))

    return data_json


@app.route("/update_questionnaire_in_game", methods=["POST"])
def create_questionnaire_in_game():
    """
    {
        "name": "Bob",
        "phone": "123456",
        "traj_id":"3_2_2023_9:30:44_human=0",
        "agent_type":"1",
        "questionnaire":{
            "I am playing well.": "I am playing well.",
            "The agent is playing poorly.": "The agent is playing poorly.",
            "The team is playing well.": "The team is playing well.",
    }
    Returns:

    """
    data_json = json.loads(request.data)
    user_id = f"{data_json.get('name')}_{data_json.get('phone')}"
    questionnaire_path = os.path.join(ARGS.questionnaire_save_path, f"{user_id}.json")
    with open(f"{questionnaire_path}", "r", encoding="utf-8") as f:
        questionnaire = json.load(f)
    if "in_game" not in questionnaire.keys():
        in_game = []
    else:
        in_game = questionnaire["in_game"]
    traj_id = data_json["traj_id"]
    agent_settings_list = list(data_json["agent_settings_list"])
    agent_type_idx = int(data_json["agent_type"])
    layout = agent_settings_list[agent_type_idx]["layout"]
    save_path = os.path.normpath(os.path.join(ARGS.trajs_save_path, layout))
    filename = f"{traj_id}.json".replace(":", "_")
    if os.path.normpath(os.path.join(save_path, filename)).replace("\\", "/") in [i_g["traj_path"] for i_g in in_game]:
        return questionnaire
    try:
        agent, human = agent_settings_list[agent_type_idx]["agents"]
    except KeyError as e:
        logger.error(e)
        agent, human = None, None

    if human != HUMAN_NAME:
        agent, human = human, agent
        human_pos = 0
    else:
        human_pos = 1
    agent_count = 0
    for in_game_item in in_game:
        if in_game_item.get("teammate") == agent:
            agent_count += 1

    in_game.append(
        {
            "traj_path": os.path.normpath(os.path.join(save_path, filename)).replace("\\", "/"),
            "questionnaire": {k: v for k, v in data_json["questionnaire"].items()},
            "teammate": agent,
            "human_pos": human_pos,
            "run_id": agent_settings_list[agent_type_idx]["run_id"],
            "n_runs": agent_settings_list[agent_type_idx]["n_runs"],
            "layout": agent_settings_list[agent_type_idx]["layout"],
            "game_id": f"{agent}_vs_human_{agent_count}",
        }
    )
    questionnaire["in_game"] = in_game
    with open(f"{questionnaire_path}", "w", encoding="utf-8") as fw:
        fw.write(json.dumps(questionnaire))

    with open(ARGS.progress_save_path, "r", encoding="utf-8") as f:
        progress_dict = json.load(f)
    progress_dict["user_progress"][user_id] = (
        progress_dict["user_progress"][user_id][0],
        min(agent_type_idx, len(progress_dict["user_progress"][user_id][0])),
        data_json["questionnaire"],
    )
    with open(ARGS.progress_save_path, "w", encoding="utf-8") as f:
        json.dump(progress_dict, f)

    return questionnaire


def get_action(state: Dict, agent: Callable, pos: int, algo: str) -> int:
    """
    get agent action from observed state s and policy
    """
    return int(agent(state, pos).item())

def rename_keys(a):
    renamed_dict = a
    for key, value in list(renamed_dict.items()):
        if key[0] == '_':
            renamed_dict[key[1:]] = renamed_dict[key]
            del renamed_dict[key]
            key = key[1:]
        if isinstance(value, dict):
            renamed_dict[key] = rename_keys(renamed_dict[key])
        elif isinstance(value, list):
            for i, n in enumerate(value):
                if isinstance(n, dict):
                    renamed_dict[key][i] = rename_keys(renamed_dict[key][i])
            
    return renamed_dict

@app.route(f"/<algo>/predict/", methods=["POST"])
def predict(algo):
    """
    Each timestep, frontend POSTs to /predict to get agent action from backend.
    The state info is got from front end, then we return action by passing state s
    into both agents.
    With each POST we process exactly one agent, so npc_index=0 or 1.

    For details see overcookedgym/overcooked-flask/static/js/demo/js/overcooked-single.js and
    overcookedgym/overcooked-flask/static/js/demo/index.js
    """
    assert request.method == "POST"
    if algo == "dummy":
        return jsonify({"action": 4})
    data_json = json.loads(request.data)
    state_dict, player_id = (
        data_json["state"],
        data_json["npc_index"],
    )
    
    state_dict_renamed = rename_keys(state_dict)
    
    #print(state_dict_renamed)
    
    pos = int(player_id)

    user_id = f"{data_json['user_info']['name']}_{data_json['user_info']['phone']}"

    if user_id not in USER_AGENTS:
        with open(ARGS.progress_save_path, "r", encoding="utf-8") as f:
            progress_dict = json.load(f)
        game_settings, agent_type, _ = progress_dict["user_progress"][user_id]
        g_setting = game_settings[agent_type]
        USER_AGENTS[user_id] = AGENT_POOLS[g_setting["layout"]].get_agent(algo)
    
    a = get_action(state_dict_renamed, USER_AGENTS[user_id], pos, algo)
    #print(a)
    return jsonify({"action": a})


def main(args: argparse.Namespace):
    
    is_old = False
    for layout in ALL_LAYOUTS:
        if layout in OLD_LAYOUTS:
            is_old = True
    
    if is_old:
        from zsceval.human_exp.agent_pool import ZSCEvalAgentPool
    else:
        from zsceval.human_exp.agent_pool_new import ZSCEvalAgentPool
        
    global ARGS, AGENT_POOLS, ALL_ALGOS, CODE, CODE_2_ALGO, ALGO_2_CODE

    ARGS = args

    debug = os.getenv("FLASK_ENV", "development") == "development"
    if not debug:
        import logging
        import sys

        logger.info("Production Mode")
        logging.getLogger("werkzeug").disabled = True
        app.logger.disabled = True
        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.add(os.path.join(os.path.dirname(ARGS.progress_save_path), "loguru.log"), level="INFO")

    logger.info("Args:\n" + pformat(ARGS.__dict__))

    t0 = time.time()
    logger.info("Loading agents...")
    AGENT_POOLS = {
        layout: ZSCEvalAgentPool(
            os.path.join(ARGS.population_yml_path, f"{layout}_benchmark.yml"),
            layout,
            deterministic=True,
            epsilon=0.5,
        )
        for layout in ALL_LAYOUTS
    }
    logger.info(f"Loaded agents in {time.time() - t0:.2f}s")
    ALL_ALGOS = []
    for layout in ALL_LAYOUTS:
        ALL_ALGOS.extend(list(AGENT_POOLS[layout].policy_pool.keys()))
    ALL_ALGOS = sorted(list(set(ALL_ALGOS)))
    CODE = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]
    CODE_2_ALGO = {f"AGENT_TYPE_{CODE[i]}": algo for i, algo in enumerate(ALL_ALGOS)}
    ALGO_2_CODE = {algo: f"AGENT_TYPE_{CODE[i]}" for i, algo in enumerate(ALL_ALGOS)}

    logger.info(f"algo {ALL_ALGOS} and algo map:\n {pformat(ALGO_2_CODE)}")

    for layout in ALL_LAYOUTS:
        logger.info(f"Layout {layout}")
        agent_pool = AGENT_POOLS[layout]
        logger.info("Agents:\n" + pformat(agent_pool.agent_names))

    if not os.path.exists(ARGS.progress_save_path):
        os.makedirs(os.path.dirname(ARGS.progress_save_path), exist_ok=True)
        with open(ARGS.progress_save_path, "w", encoding="utf-8") as f:
            json.dump({"user_progress": {}, "used_game_settings": []}, f)
            """ 
            user_progress:
                {
                    user_id:
                        - game_settings
                        - agent_type (game_index)
                        - code (algo) order
                }
            used_game_settings:
                [
                    - 0 game_settings
                    - 1 game_settings
                ]
            """

    random.seed(ARGS.seed)


app_args = get_args().parse_args()
main(app_args)

if __name__ == "__main__":
    host = ARGS.ip
    port = ARGS.port
    debug = os.getenv("FLASK_ENV", "development") == "development"
    app.run(debug=debug, host=host, port=port, threaded=True)
