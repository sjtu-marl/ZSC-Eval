import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import math
import time

from adjustText import adjust_text

import sys
import json
import pickle
import os
import glob
import shutil
from pathlib import Path
import re

from multiprocessing import Pool

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from loguru import logger

LABELS_NEW_CORE =[
    
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
    "onions_placed_on_X",
    "tomatoes_placed_on_X",
    "dishes_placed_on_X",
    "soups_placed_on_X",
    "sparse_r",
    "shaped_r",
]

LABELS_NEW = [
    "put_onion_on_X_by_agent0",
    "put_tomato_on_X_by_agent0",
    "put_dish_on_X_by_agent0",
    "put_soup_on_X_by_agent0",
    "pickup_onion_from_X_by_agent0",
    "pickup_onion_from_O_by_agent0",
    "pickup_tomato_from_X_by_agent0",
    "pickup_tomato_from_T_by_agent0",
    "pickup_dish_from_X_by_agent0",
    "pickup_dish_from_D_by_agent0",
    "pickup_soup_from_X_by_agent0",
    "USEFUL_DISH_PICKUP_by_agent0", 
    "SOUP_PICKUP_by_agent0", 
    "PLACEMENT_IN_POT_by_agent0",
    "viable_placement_by_agent0",
    "optimal_placement_by_agent0",
    "catastrophic_placement_by_agent0",
    "useless_placement_by_agent0", 
    "potting_onion_by_agent0",
    "potting_tomato_by_agent0",
    "cook_by_agent0",
    "delivery_by_agent0",
    "deliver_size_two_order_by_agent0",
    "deliver_size_three_order_by_agent0",
    "deliver_useless_order_by_agent0",
    "STAY_by_agent0",
    "MOVEMENT_by_agent0",
    "IDLE_MOVEMENT_by_agent0",
    "IDLE_INTERACT_by_agent0",
    "sparse_r_by_agent0",
    "shaped_r_by_agent0",
    "place_onion_on_X_by_agent0",
    "place_tomato_on_X_by_agent0",
    "place_dish_on_X_by_agent0",
    "place_soup_on_X_by_agent0",
    "recieve_onion_via_X_by_agent0",
    "recieve_tomato_via_X_by_agent0",
    "recieve_dish_via_X_by_agent0",
    "recieve_soup_via_X_by_agent0",
    "onions_placed_on_X_by_agent0",
    "tomatoes_placed_on_X_by_agent0",
    "dishes_placed_on_X_by_agent0",
    "soups_placed_on_X_by_agent0",
    "utility_r_by_agent0",
    
    "put_onion_on_X_by_agent1",
    "put_tomato_on_X_by_agent1",
    "put_dish_on_X_by_agent1",
    "put_soup_on_X_by_agent1",
    "pickup_onion_from_X_by_agent1",
    "pickup_onion_from_O_by_agent1",
    "pickup_tomato_from_X_by_agent1",
    "pickup_tomato_from_T_by_agent1",
    "pickup_dish_from_X_by_agent1",
    "pickup_dish_from_D_by_agent1",
    "pickup_soup_from_X_by_agent1",
    "USEFUL_DISH_PICKUP_by_agent1", 
    "SOUP_PICKUP_by_agent1", 
    "PLACEMENT_IN_POT_by_agent1",
    "viable_placement_by_agent1",
    "optimal_placement_by_agent1",
    "catastrophic_placement_by_agent1",
    "useless_placement_by_agent1", 
    "potting_onion_by_agent1",
    "potting_tomato_by_agent1",
    "cook_by_agent1",
    "delivery_by_agent1",
    "deliver_size_two_order_by_agent1",
    "deliver_size_three_order_by_agent1",
    "deliver_useless_order_by_agent1",
    "STAY_by_agent1",
    "MOVEMENT_by_agent1",
    "IDLE_MOVEMENT_by_agent1",
    "IDLE_INTERACT_by_agent1",
    "sparse_r_by_agent1",
    "shaped_r_by_agent1",
    "place_onion_on_X_by_agent1",
    "place_tomato_on_X_by_agent1",
    "place_dish_on_X_by_agent1",
    "place_soup_on_X_by_agent1",
    "recieve_onion_via_X_by_agent1",
    "recieve_tomato_via_X_by_agent1",
    "recieve_dish_via_X_by_agent1",
    "recieve_soup_via_X_by_agent1",
    "onions_placed_on_X_by_agent1",
    "tomatoes_placed_on_X_by_agent1",
    "dishes_placed_on_X_by_agent1",
    "soups_placed_on_X_by_agent1",
    "utility_r_by_agent1",
    
    "sparse_r",
    "shaped_r",
    "utility_r"


]

LABELS_OLD = [
    "put_onion_on_X_by_agent0",
    "put_dish_on_X_by_agent0",
    "put_soup_on_X_by_agent0",
    "pickup_onion_from_X_by_agent0",
    "pickup_onion_from_O_by_agent0",
    "pickup_dish_from_X_by_agent0",
    "pickup_dish_from_D_by_agent0",
    "pickup_soup_from_X_by_agent0",
    "USEFUL_DISH_PICKUP_by_agent0",
    "SOUP_PICKUP_by_agent0",
    "PLACEMENT_IN_POT_by_agent0",
    "delivery_by_agent0",
    "STAY_by_agent0",
    "MOVEMENT_by_agent0",
    "IDLE_MOVEMENT_by_agent0",
    "IDLE_INTERACT_X_by_agent0",
    "IDLE_INTERACT_EMPTY_by_agent0",
    "sparse_r_by_agent0",
    "shaped_r_by_agent0",
    
    "put_onion_on_X_by_agent1",
    "put_dish_on_X_by_agent1",
    "put_soup_on_X_by_agent1",
    "pickup_onion_from_X_by_agent1",
    "pickup_onion_from_O_by_agent1",
    "pickup_dish_from_X_by_agent1",
    "pickup_dish_from_D_by_agent1",
    "pickup_soup_from_X_by_agent1",
    "USEFUL_DISH_PICKUP_by_agent1",
    "SOUP_PICKUP_by_agent1",
    "PLACEMENT_IN_POT_by_agent1",
    "delivery_by_agent1",
    "STAY_by_agent1",
    "MOVEMENT_by_agent1",
    "IDLE_MOVEMENT_by_agent1",
    "IDLE_INTERACT_X_by_agent1",
    "IDLE_INTERACT_EMPTY_by_agent1",
    "sparse_r_by_agent1",
    "shaped_r_by_agent1",
    
    "sparse_r",
    "shaped_r",
    "either_sparse_r",
    "either_shaped_r",
    "either_sparse_r_agent0",
    "either_shaped_r_agent0",
    "either_sparse_r_agent1",
    "either_shaped_r_agent1",

]

def summon_gif(save_path, layout, exp, index):
    
    data_dir = Path(os.path.expanduser("~") + "/ZSC/results/Overcooked/" + layout + "/population/eval-" + exp + "-" + str(index))
    
    gifs =  glob.glob(str(data_dir) + "/**/*.gif" , recursive=True)
    
    if(len(gifs)==0):
        logger.debug(f"No gif for {exp}{str(index)}")
        return
    
    reward = re.findall(r'\d+', gifs[0])
    
    if not os.path.exists(f"{save_path}/gifs"):
        os.makedirs(f"{save_path}/gifs")
    
    shutil.copy(gifs[0], f"{save_path}/gifs/{exp}_{str(index)}-reward_{reward[-1]}.gif")
    logger.debug(f"gif saved at {save_path}/gifs/{exp}_{str(index)}-reward_{reward[-1]}.gif")  
        
        
def render_traj(joint_actions, save_dir, layout_name, traj_label):
    
    from zsceval.envs.overcooked_new.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
    from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action
    import imageio
    
    try:
        translated_joint_actions = []
        for joint_action in joint_actions:
            trans = []
            for player_action in joint_action:
                if player_action =="interact":
                    trans.append(player_action)
                else:
                    trans.append((player_action[0], player_action[1]))
            translated_joint_actions.append(trans)
            
        filename = f"{traj_label}.gif".replace(":", "_")
        os.makedirs(save_dir, exist_ok=True)
        temp_dir = os.path.normpath(os.path.join(save_dir, f"temp_{traj_label}"))
        
        StateVisualizer().render_from_actions(translated_joint_actions, layout_name, img_directory_path=temp_dir)
        
        for img_path in os.listdir(temp_dir):
            img_path = temp_dir + "/" + img_path
        imgs = []
        imgs_dir = os.listdir(temp_dir)
        imgs_dir = sorted(imgs_dir, key=lambda x: int(x.split(".")[0]))
        for img_path in imgs_dir:
            img_path = temp_dir + "/" + img_path
            imgs.append(imageio.imread(img_path))
        save_path = save_dir + f'/{filename}'
        imageio.mimsave(
            save_path,
            imgs,
            duration=0.05,
        )
        logger.info(f"saved gif in {save_path}")
        
        # delete pngs
        imgs_dir = os.listdir(temp_dir)
        for img_path in imgs_dir:
            img_path = temp_dir + "/" + img_path
            if "png" in img_path:
                os.remove(img_path)
        shutil.rmtree(temp_dir)
        logger.info(f"deleted temp dir : {temp_dir}")
        
    except Exception as e:
        logger.exception(e)
         
    return 

def render_traj_wrapper(args):
    return render_traj(*args)

def render_gif_from_traj(exp_save_dir, layout, exp, index, threads=4):
    
    save_dir = f"{exp_save_dir}/gifs"
    
    data_dir = Path(os.path.expanduser("~") + "/ZSC/results/Overcooked/" + layout + "/population/eval_cp-" + exp + "-" + str(index))
    logger.debug(exp)
    traj_jsons =  glob.glob(str(data_dir) + "/**/*.json" , recursive=True)
    
    args = []
    label_log = []
    for i, json_path in enumerate(sorted(traj_jsons)):
        #logger.debug(json_path)
        traj_label = re.findall(r'.*_(\d+)_\d+.json', json_path)
        
        if(len(traj_label)==0):
            logger.info(f"Can't find traj json in {json_path}")
            continue
        # only sample one gifs per eval settings
        if traj_label[0] not in label_log:
            json_open = open(json_path, 'r')
            json_load = json.load(json_open)
            args.append([json_load["ep_action"], save_dir, layout, f"{exp}_{index}_{traj_label}"])
            label_log.append(traj_label)
    
    p = Pool(threads)
    p.map(render_traj_wrapper, args)
    
    
def load_behavior(layout, exp, seeds, match_ups, use_new, save_root, collect_gif, gif_from_traj):
    
    logger.debug(f"Loading {exp}")

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]
    
    
    LABELS = sorted(LABELS_NEW) if use_new else sorted(LABELS_OLD)
    
    
    #for label in LABELS:
    #    for match_up in match_ups:
    #        keys.append(f"{match_up}-eval_ep_{label}")
    
    files = glob.glob(f"eval/results/{layout}/{exp}/*.json")
    sorted_files = sorted(files, key=natural_keys)
    logger.debug(sorted_files)
    
    keys = None
    with open(sorted_files[0], mode="rt", encoding="utf-8") as f:
        json_dict = json.load(f)
        keys = json_dict.keys()
    
    #logger.debug(match_ups)
    shaped_dict = {}
    for label in sorted(LABELS):
        for match_up in match_ups:
            for i in range(len(match_up)):
                shaped_dict[f"{match_up[i]}_{label}"] = []

    save_dir = save_root + exp
    if not os.path.exists(f"{save_dir}/gifs"):
        os.makedirs(f"{save_dir}/gifs")
    else:
        shutil.rmtree(f"{save_dir}/gifs")
        os.makedirs(f"{save_dir}/gifs")

    index = []
    for i in range(seeds):
        #if(i is 25 or i is 26):
        #    continue
        file = sorted_files[i]
        index.append(f"{exp}_{i+1}")
        with open(file, mode="rt", encoding="utf-8") as f:
            json_dict = json.load(f)
            json_dict = {k: v for k, v in json_dict.items() if "either" not in k}
            logger.debug(len(json_dict))
            logger.debug(len(shaped_dict.keys()))
            ordered_json_keys = sorted(list(json_dict.keys()))
            #logger.debug(shaped_dict.keys())
            #logger.debug(ordered_json)
            assert(len(json_dict)==len(shaped_dict.keys()))
            
            for j, (key_s, key_o) in enumerate(zip(shaped_dict.keys(), ordered_json_keys)):
                shaped_dict[key_s].append(json_dict[key_o])
                
        if collect_gif:
            if gif_from_traj:
                render_gif_from_traj(save_dir, layout, exp, i+1)
            else:
                summon_gif(save_root + exp, layout, exp, i+1)
            

    
    #logger.debug(len(LABELS))
    #logger.debug(f"{len(json_values)}, {len(json_values[0])}")

    df = pd.DataFrame(data=shaped_dict, index=index)
    #logger.debug(df)
    
    #df = df.rename(index=index)
    
    return df

def load_human_ai():
    json_values = []
    index = []
    for i in range(1,seed_max+1):
        if(i is 25 or i is 26):
            continue
        file = f"eval-{exp}{i}"
        index.append(f"{exp}{i}")
        path = f"eval/results/{layout}/{alg}/{file}.json"
        with open(path, mode="rt", encoding="utf-8") as f:
            json_dict = json.load(f)
            orderedNames = sorted(list(json_dict.keys()))
            values = []
            for name in orderedNames:
                #logger.debug(name)
                if not "either" in name:
                   values.append(json_dict[name])
            json_values.append(values)
            
        summon_gif(layout, exp, i)
    
    LABELS = sorted(LABELS_NEW) if use_new else sorted(LABELS_OLD)
    
    logger.debug(len(LABELS))
    logger.debug(f"{len(json_values)}, {len(json_values[0])}")
    
    shaped_dict = {}
    for label in LABELS:
        shaped_dict[label] = []
    for values in json_values:
        for j, v in enumerate(values):
            if(j >= len(LABELS)):
                break
            shaped_dict[LABELS[j]].append(v)
                        
    df = pd.DataFrame(data=shaped_dict, index=index)
    #df = df.rename(index=index)
    
    return df
    
def load_trajectory(layout_name, alg, exps, ranks, run, traj_num, use_new):
    
    for i, exp in enumerate(exps):
        run_dir = Path(os.path.expanduser("~") + "/ZSC/results/Overcooked/" + layout_name + "/population/eval-" + alg + str(exp) + "/run" + str(run)) 
        for rank in ranks:
            traj = None
            with open(f"{run_dir}/trajs_store/{layout_name}/traj_{rank}_{traj_num}.pkl",'rb') as f:
                traj = pickle.load(f)
            
            logger.debug(len(traj))
    
def plot_pca(df_all, exps, layout, plot_label, save_root):
    
    std_scaler = MinMaxScaler()
    std_scaler.fit(df_all.to_numpy())
    
    df_std = std_scaler.transform(df_all)
    
    df_std = pd.DataFrame(df_std, index=df_all.index, columns=df_all.columns)
    
    #logger.debug(df_std)
    
    pca = PCA()
    pca.fit(df_std)
    
    score = pd.DataFrame(pca.transform(df_std))

    #logger.debug(score)
    
    num = len(score)  # 可視化するデータ数を指定
    
    cmap = plt.get_cmap("tab20")
    
    plt.rcParams["font.size"] = 5
    
    texts = []
    colors = []
    markers=["8", "*", "s", "p",  "h", "H", "D"]
    for i in range(10):
        # no orange
        if i == 1:
            continue
        if i % 2 == 1:
            colors.append(cmap(2*i))
        else:
            colors.append(cmap(2*i))
    
    for i in range(num):
        alg_index = None
        for j, alg in enumerate(exps):
            alg_index = j if alg in df_all.index.values[i] else alg_index
        
        plt.scatter(score.iloc[i][0], score.iloc[i][1], c=colors[alg_index], marker=markers[alg_index]) 
        if plot_label:
            texts.append(plt.text(score.iloc[i,0], score.iloc[i,1], df_all.index[i], horizontalalignment="center", verticalalignment="bottom"))
    
    if plot_label:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
        
    handles = []
    for i, exp in enumerate(exps):
        handles.append(mlines.Line2D([],[],color=colors[i],marker=markers[i],linestyle='None',markersize=5, label=exp))
            
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), handles=handles)
    plt.xlabel("1st principal component")
    plt.xlabel("2nd principal component")
    plt.grid()
    plt.savefig(f"{save_root}/pca.jpg", dpi=300, bbox_inches='tight')
    logger.debug(f"saved pca plot at {save_root}/pca.jpg")
    plt.close()
    
    
def compute_cos_similarity(df_all):
    
    std_scaler = StandardScaler()
    std_scaler.fit(df_all.to_numpy())

    df_1 = pd.DataFrame(std_scaler.transform(df_all), index=df_all.index, columns=df_all.columns)
    df_2 = pd.DataFrame(std_scaler.transform(df_all), index=df_all.index, columns=df_all.columns)
    
    #logger.debug(df_1)
    
    # 行列ベクトルの片方を転置して積を求める
    df_dot = df_1.dot(df_2.T)

    # 行列ノルムを求める
    df1_norm = pd.DataFrame(np.linalg.norm(df_1.values, axis=1), index = df_1.index)
    df2_norm = pd.DataFrame(np.linalg.norm(df_2.values, axis=1), index = df_2.index)
    
    # 行列ノルムの片方を転置して積を求める
    df_norm = df1_norm.dot(df2_norm.T)

    # コサイン類似度を算出
    df_cos = df_dot/df_norm

    return df_cos
    
def plot_radar(df, exps, seed_max, subscript, save_root):
    
    #df_no_zero = df.loc[:, (df != 0).any(axis=0)]
    df_no_zero = df
    
    label_list = df_no_zero.columns.values
    index_list = df_no_zero.index
    
    scaler = MinMaxScaler()
    #logger.debug(scaler.fit_transform(df_no_zero))
    df_scaler = pd.DataFrame(scaler.fit_transform(df_no_zero), columns=label_list)
    
    df_scaler.to_csv(f"{save_root}/min_max_scale.csv")
    
    logger.debug("plotting radars...")
    
    exp_index = 0
    for i, series in df_scaler.iterrows():
        
        value_list = series.tolist()
        value_list += value_list[:1]
        
        angle_list = [n / float(len(label_list)) * 2 * np.pi for n in range(len(label_list))]
        angle_list += angle_list[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        plt.xticks(angle_list[:-1], label_list, color='grey', size=12)
        
        ax.plot(angle_list, value_list, linewidth=1, linestyle='solid')
        ax.fill(angle_list, value_list, 'blue', alpha=0.1)
        ax.set_ylim(ymin=0, ymax=1.0)
        
        logger.debug(index_list[i])

        rows = i
        for j, seeds in enumerate(seed_max):
            rows = rows - seeds 
            exp_index = j if rows == 0 else exp_index
        
        if not os.path.exists(f"{save_root}/{exps[exp_index]}/radars"):
            os.makedirs(f"{save_root}/{exps[exp_index]}/radars")
        
        plt.savefig(f"{save_root}/{exps[exp_index]}/radars/radar_{index_list[i]}_{subscript}.jpg")
        plt.close()
        
    
    
def dump_hist(df, df_atr, bins, xlabel, ylabel, save_dir, subscript, color):
    plt.rcParams["font.size"] = 20
    
    #if(np.isnan(df_atr.min()) or np.isnan(df_atr.max()) or df_atr.min() >= df_atr.max()):
    #   return
    
    plt.hist(df_atr, bins=bins, range=(df_atr.min(), df_atr.max()), color=color)
    
    plt.ylim(0, len(df_atr))
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    
    #plt.xticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{subscript}.jpg", dpi=300)
    plt.close() 

def dump_hist_comp(df_atr, exps, bins, xlabel, ylabel, save_dir, subscript, colors):
            
    #if(not np.isnan(x_min) or not np.isnan(x_max) or x_min >= x_max):
    #    return
    # logger.debug(df_atr.columns)
    for df_match_up_atr_key in df_atr.columns:
        fig = plt.figure(figsize=(20,10))
        fig.suptitle("")
        
        df_match_up_atr = df_atr[df_match_up_atr_key]
        logger.debug(exps)
        for i, exp in enumerate(exps):
            df_exp_atr = df_match_up_atr.filter(regex=f"^{exp}_(\d)+$", axis=0)
            print(df_exp_atr)
            logger.debug(df_exp_atr.min())
            logger.debug(df_exp_atr.max())
            ax = fig.add_subplot(2, 2, i+1)
            ax.hist(df_exp_atr, bins=bins, range=(df_exp_atr.min(), df_exp_atr.max()), color=colors[i])
            ax.set_title(exp)
            ax.set_ylim(0, len(df_exp_atr))
            ax.set_xlabel(xlabel=xlabel)
            ax.set_ylabel(ylabel=ylabel)
            
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{subscript}_{df_match_up_atr_key}.jpg", dpi=300)
        plt.close() 
    
    
def get_atr_params(key, value, filter):
    
    logger.debug(key)
    logger.debug(filter)
    
    df_dish_remain = value.filter(like=f"{filter}dishes_placed_on_X_{key}",axis=1)
    df_soup_remain = value.filter(like=f"{filter}soups_placed_on_X_{key}",axis=1)
    df_plate_remain = df_dish_remain.add(df_soup_remain.values, axis=0)
    df_onion_remain = value.filter(like=f"{filter}onions_placed_on_X_{key}",axis=1)
    df_tomato_remain = value.filter(like=f"{filter}tomatoes_placed_on_X_{key}",axis=1)
    
    df_stay = value.filter(like=f"{filter}STAY_by_{key}",axis=1)
    df_movement = value.filter(like=f"{filter}MOVEMENT_by_{key}",axis=1)
    df_onion = value.filter(like=f"{filter}potting_onion_by_{key}",axis=1)
    df_tomato = value.filter(like=f"{filter}potting_tomato_by_{key}",axis=1)
    df_size_2 = value.filter(like=f"{filter}deliver_size_two_order_by_{key}",axis=1)
    df_size_3 = value.filter(like=f"{filter}deliver_size_three_order_by_{key}",axis=1)
    df_sparse_reward = value.filter(like=f"{filter}sparse_r_by_{key}",axis=1)
    
    
    df_put_onion_on_X = value.filter(like=f"{filter}put_onion_on_X_by_{key}",axis=1)
    df_put_tomato_on_X = value.filter(like=f"{filter}put_tomato_on_X_by_{key}",axis=1)
    df_put_dish_on_X = value.filter(like=f"{filter}put_dish_on_X_by_{key}",axis=1)
    df_put_soup_on_X = value.filter(like=f"{filter}put_soup_on_X_by_{key}",axis=1)
    
    df_pickup_onion_from_X = value.filter(like=f"{filter}pickup_onion_from_X_by_{key}",axis=1)
    df_pickup_tomato_from_X = value.filter(like=f"{filter}pickup_tomato_from_X_by_{key}",axis=1)
    df_pickup_dish_from_X = value.filter(like=f"{filter}pickup_dish_from_X_by_{key}",axis=1)
    df_pickup_soup_from_X = value.filter(like=f"{filter}pickup_soup_from_X_by_{key}",axis=1)
    
    df_pickup_onion_from_O = value.filter(like=f"{filter}pickup_onion_from_O_by_{key}",axis=1)
    df_pickup_tomato_from_T = value.filter(like=f"{filter}pickup_tomato_from_T_by_{key}",axis=1)
    df_pickup_dish_from_D = value.filter(like=f"{filter}pickup_dish_from_D_by_{key}",axis=1)
    df_SOUP_PICKUP = value.filter(like=f"{filter}SOUP_PICKUP_by_{key}",axis=1)
    
    df_onion_placement = value.filter(like=f"{filter}place_onion_on_X_{key}",axis=1)
    df_tomato_placement = value.filter(like=f"{filter}place_tomato_on_X_{key}",axis=1)
    df_dish_placement = value.filter(like=f"{filter}place_dish_on_X_{key}",axis=1)
    df_soup_placement = value.filter(like=f"{filter}place_soup_on_X_{key}",axis=1)
    
    #pd.set_option('display.max_rows', None)
    
    
    df_atrs = {"dish_remain" : df_dish_remain, "soup_remain" : df_soup_remain,
                "plate_remain" : df_plate_remain, "movement" : df_movement, 
                #"onion" : df_onion,
                #"tomato" : df_tomato,  "size_2" : df_size_2,
                "size_3" : df_size_3,
                "sparse_reward" : df_sparse_reward, "stay" : df_stay,
                "put_onion_on_X" : df_put_onion_on_X, "put_tomato_on_X" : df_put_tomato_on_X,
                "put_dish_on_X" : df_put_dish_on_X, "put_soup_on_X" : df_put_soup_on_X,
                "pickup_onion_from_X" : df_pickup_onion_from_X, "pickup_tomato_from_X"  : df_pickup_tomato_from_X,
                "pickup_dish_from_X" : df_pickup_dish_from_X, "pickup_soup_from_X" : df_pickup_soup_from_X,
                "pickup_onion_from_O" : df_pickup_onion_from_O, "pickup_tomato_from_T"  : df_pickup_tomato_from_T,
                "pickup_dish_from_D" : df_pickup_dish_from_D, "pickup_soup_from_P" : df_SOUP_PICKUP,
            }
    atrs_min = {atr_label : atr_value.min() for atr_label, atr_value in df_atrs.items()}
    atrs_max = {atr_label : atr_value.max() for atr_label, atr_value in df_atrs.items()}
    atrs_disc = {"dish_remain" : "Counts for dish remaining",
                "soup_remain" : "Counts for soup remaining",
                "plate_remain" : "Counts for plate (placement - pickup)",
                "movement" : "Counts for movement", 
                #"onion" : "Counts for cooking onions",
                #"tomato" : "Counts for cooking tomatos",
                #"size_2" : "Counts for delivering size 2 recipe",
                #"size_3" : "Counts for delivering size 3 recipe",
                "sparse_reward" : "Final scores",
                "stay" : "Counts for Staying",
                "put_onion_on_X" : "Counts for putting a onion on the counter",
                "put_tomato_on_X" : "Counts for putting a tomato on the counter",
                "put_dish_on_X" : "Counts for putting a plate on the counter",
                "put_soup_on_X" : "Counts for putting a plate of soup on the counter",
                
                "pickup_onion_from_X" : "Counts for picking up a onion from the counter",
                "pickup_tomato_from_X" : "Counts for picking up a tomato from the counter",
                "pickup_dish_from_X" : "Counts for picking up a plate from the counter",
                "pickup_soup_from_X" : "Counts for picking up a plate of soup from the counter",
                
                "pickup_onion_from_O" : "Counts for picking up a onion from the source",
                "pickup_tomato_from_T" : "Counts for picking up a tomato from the source",
                "pickup_dish_from_D" : "Counts for picking up a plate from the source",
                "pickup_soup_from_P" : "Counts for picking up a plate of soup from the pot",
            }
    
    atrs_bins = {"dish_remain" : 20, "soup_remain" : 20,
                "plate_remain" : 20, 
                "movement" : 20,
                "onion" : 20,
                "tomato" : 20, 
                "size_2" : 20,
                "size_3" : 20, 
                "sparse_reward" : 20,
                "stay" : 20,
                "put_onion_on_X" : 20, "put_tomato_on_X" :20,  "put_dish_on_X" :20, "put_soup_on_X" :20,
                "pickup_onion_from_X" : 20, "pickup_tomato_from_X" : 20, "pickup_dish_from_X" : 20,
                "pickup_soup_from_X" : 20, "pickup_onion_from_O" : 20, "pickup_tomato_from_T" : 20,
                "pickup_dish_from_D" : 20, "pickup_soup_from_P" : 20,
                
                }
        
    ylabel = "Number of AIs"
    
    if key == "all_agents":
                
        #plates_remain_agent0 = df_agent0.filter(like=f"put_dish_on_X_by_agent0",axis=1) - df_agent0.filter(like=f"pickup_dish_from_X_by_agent0",axis=1)
        #plates_remain_agent1 = df_agent1.filter(like=f"put_dish_on_X_by_agent1",axis=1) - df_agent1.filter(like=f"pickup_dish_from_X_by_agent1",axis=1)
        
        plates_pickup_agent0 = df_agent0.filter(like=f"pickup_dish_from_X_by_agent0",axis=1).subtract(df_agent0.filter(like=f"put_dish_on_X_by_agent0",axis=1).values, axis=0)
        plates_pickup_agent1 = df_agent1.filter(like=f"pickup_dish_from_X_by_agent1",axis=1).subtract(df_agent1.filter(like=f"put_dish_on_X_by_agent1",axis=1).values, axis=0)
        
        df_atrs["plates_coord_0_to_1"] = plates_pickup_agent1
        df_atrs["plates_coord_1_to_0"] = plates_pickup_agent0
        df_atrs["plates_coord_total"] = df_atrs["plates_coord_0_to_1"].add(df_atrs["plates_coord_1_to_0"].values, axis=0)
        
        atrs_disc["plates_coord_0_to_1"] = "Counts for plate pickup - plate placement"
        atrs_disc["plates_coord_1_to_0"] = "Counts for plate pickup - plate placement"
        atrs_disc["plates_coord_total"] = "Counts for plate pickup - plate placement"
        
        atrs_bins["plates_coord_0_to_1"] = 20
        atrs_bins["plates_coord_1_to_0"] = 20
        atrs_bins["plates_coord_total"] = 20  
    
    return df_atrs, atrs_min, atrs_max, atrs_disc, atrs_bins, ylabel
    





def plot_histgrams(df, layout, exps, exp_match_ups, save_root, use_new=True):
    
    df_agent0 = df.filter(like="agent0", axis=1)
    df_agent1 = df.filter(like="agent1", axis=1)
    
    df_all_agents = pd.DataFrame()
    for j, column in enumerate(df_agent0.columns):
        df_all_agents[column.replace("agent0", "all_agents")] = (df_agent0.iloc[:,j].add(df_agent1.iloc[:,j].values, axis=0)) / 2

    # df_agents = {"all_agents":df_all_agents,"agent0":df_agent0, "agent1":df_agent1}
    df_agents = {}
        
    cmap = plt.get_cmap("tab20")
    colors = []
    for j in range(10):
        # no orange
        if j == 1:
            continue
        if j % 2 == 1:
            colors.append(cmap(2*j))
        else:
            colors.append(cmap(2*j))
    
    # define atrs in a match_up division
    for i, (exp, match_ups) in enumerate(zip(exps, exp_match_ups)):
        
        logger.debug(match_ups)
        logger.debug(df_agent0.shape)
        logger.debug(len(match_ups))
        df_match_up_agent0_sum = pd.DataFrame(np.zeros((df_agent0.shape[0],int(df_agent0.shape[1]/(len(match_ups))))), index=df_agent0.index)
        df_match_up_agent1_sum = pd.DataFrame(np.zeros((df_agent1.shape[0],int(df_agent1.shape[1]/(len(match_ups))))), index=df_agent1.index)
        df_all_match_up_sum = pd.DataFrame(np.zeros((df_all_agents.shape[0],int(df_all_agents.shape[1]/(len(match_ups))))), index=df_all_agents.index)
        
        for i, match_up in enumerate(match_ups):
            key = f"{match_up}"
            # logger.debug(key)
            
            df_match_up_agent0 = df_agent0.filter(regex=fr'^{key}_.*', axis=1)
            df_match_up_agent1 = df_agent1.filter(regex=fr'^{key}_.*', axis=1)
            df_match_up_all_agents = df_all_agents.filter(regex=fr'^{key}_.*', axis=1)
            
            df_agents[f"{key}:agent0"] = df_match_up_agent0
            df_agents[f"{key}:agent1"] = df_match_up_agent1
            df_agents[f"{key}:all_agents"] = df_match_up_all_agents
            
            # logger.debug(df_match_up_agent0.columns)
            
            df_match_up_agent0_sum = df_match_up_agent0_sum.add(df_match_up_agent0.values, axis=0)
            df_match_up_agent1_sum = df_match_up_agent1_sum.add(df_match_up_agent1.values, axis=0)
            df_all_match_up_sum = df_all_match_up_sum.add(df_match_up_all_agents.values, axis=0)
            
        df_match_up_agent0_sum = df_match_up_agent0_sum / (len(match_ups))
        df_match_up_agent1_sum = df_match_up_agent1_sum / (len(match_ups))
        df_all_match_up_sum = df_all_match_up_sum / (len(match_ups))
        
        LABELS = sorted(LABELS_NEW) if use_new else sorted(LABELS_OLD)
        agent0_columns = []
        agent1_columns = []
        all_agents_columns = []
        for label in [l for l in LABELS if "agent0" in l]:
            agent0_columns.append(label + "_all_match_ups_average")
        for label in [l for l in LABELS if "agent1" in l]:
            agent1_columns.append(label + "_all_match_ups_average")
        for label in [l for l in LABELS if "agent0" in l]:
            label = label.replace("agent0","all_agents")
            all_agents_columns.append(label + "_all_match_ups_average")
            
        df_match_up_agent0_sum.columns = agent0_columns
        df_match_up_agent1_sum.columns = agent1_columns
        df_all_match_up_sum.columns = all_agents_columns
        
        df_agents[f"all_match_ups:agent0"] = df_match_up_agent0_sum
        df_agents[f"all_match_ups:agent1"] = df_match_up_agent1_sum
        df_agents[f"all_match_ups:all_agents"] = df_all_match_up_sum
        
        
    # dump for each dfs
    for key, value in df_agents.items():
        
        logger.debug(key)
        logger.debug(value)
        
        if len(key.split(":")) == 1:
            match_up_filter = ""
            agent_type = key
        else:
            match_up_filter = key.split(":")[0]+"_"
            agent_type = key.split(":")[1]
            
        df_atrs, atrs_min, atrs_max, atrs_disc, atrs_bins, ylabel = get_atr_params(agent_type, value, match_up_filter)
        
        
        ranking_all_dir = f"{save_root}/rankings"
        if not os.path.exists(ranking_all_dir):
            os.makedirs(ranking_all_dir)
            
        hist_dir = f"{save_root}histgrams"
        if not os.path.exists(hist_dir):
            os.makedirs(hist_dir)

        df_all_sorted= pd.DataFrame()
        
        for atr_label, df_atr in df_atrs.items():
            
            for df_match_up_atr_key in df_atr.columns:
                
                df_match_up_atr = df_atr[df_match_up_atr_key]
                
                df_sorted = df_match_up_atr.sort_values(ascending=False)
                df_all_sorted[f"{atr_label}_{df_match_up_atr_key}_i"] = df_sorted.index
                df_all_sorted[f"{atr_label}_{df_match_up_atr_key}_v"] = df_sorted.values

            dump_hist_comp(df_atr, exps, bins=atrs_bins[atr_label],
                        xlabel=atrs_disc[atr_label], ylabel=ylabel,
                        save_dir=hist_dir, subscript=f"hist_{atr_label}_{key}",
                        colors=colors)
            
        df_all_sorted.to_csv(f"{ranking_all_dir}/ranking_all-exp_{key}.csv")

        
        
        for i, exp in enumerate(exps):
            
            df_sorted = pd.DataFrame()
            
            hist_dir = f"{save_root}{exp}/histgrams"
            ranking_dir = f"{save_root}{exp}/rankings"
            
            if not os.path.exists(hist_dir):
                os.makedirs(hist_dir)
            if not os.path.exists(ranking_dir):
                os.makedirs(ranking_dir)
            
            for atr_label, df_atr in df_atrs.items():
                
                for df_match_up_atr_key in df_atr.columns:
                    
                    df_match_up_atr = df_atr[df_match_up_atr_key]
                    
                    # need to filter
                    df_atr_filtered = df_match_up_atr.filter(regex=f"^{exp}_(\d)+$", axis=0)
                    
                    logger.debug(df_atr_filtered.shape)
                    
                    dump_hist(value, df_atr_filtered, bins=atrs_bins[atr_label],
                            xlabel=atrs_disc[atr_label], ylabel=ylabel,
                            save_dir=hist_dir, subscript=f"hist_{atr_label}_{exp}_{key}_{df_match_up_atr_key}",
                            color=colors[i])
                
                    df_sort = df_atr_filtered.sort_values(ascending=False)

                    df_sorted[f"{atr_label}_{df_match_up_atr_key}_i"] = df_sort.index
                    df_sorted[f"{atr_label}_{df_match_up_atr_key}_v"] = df_sort.values

            df_sorted.to_csv(f"{ranking_dir}/ranking_{exp}_{key}.csv")
        
    
def init_dir(root_path, exps):
    
    
    if not os.path.exists(root_path):
        os.makedirs(root_path)
        
    for exp in exps:
        if not os.path.exists(root_path + exp):
            os.makedirs(root_path + exp)

if __name__ == "__main__":
    
    layout = sys.argv[1]
    
    use_new = True
    
    _collect_gif = True
    
    _gif_from_traj = True
    
    _plot_radar = False
    
    _plot_hist = False
    
    _plot_pca = False
    
    _plot_label = False
    
    save_root = f"./eval/results/{layout}/analysis/"
    
    exps = [
    #    "hsp","adaptive_hsp_plate","adaptive_hsp_plate_vs_hsp_plate",
    #    "hsp_plate_shared", "hsp_plate_shared-pop_cross_play", "adaptive_hsp_plate_shared-pop_cross_play",
    #    "hsp_plate_shared-pop_cross_play", "hsp_plate-S2-s36-adp_cp-s5" , "adaptive_hsp_plate-S2-s36-adp_cp-s5",
    #    "hsp_all_shared", 
    #    "hsp_plate-S2-s36-adp_cp-s5","adaptive_hsp_plate-S2-s36-adp_cp-s5",
    #    "mep-S2-s36-adp_cp-s5", "adaptive_mep-S2-s36-adp_cp-s5"
    #     "adaptive_hsp_plate_shared-pop_cp-s60", "hsp_plate_shared-pop_cp-s60",
         #"hsp_plate_placement", "reactive_hsp_placement",
         "hsp_plate_placement_shared-S2-s12", 
         #"reactive_hsp_plate_placement_shared-S3-s12"
        ]
    #exps = ["hsp_plate_shared", "hsp_plate_shared-pop_cross_play", "adaptive_hsp_plate_shared-pop_cross_play"]
    #exps = ["hsp_plate_shared-pop_cross_play", "hsp_plate-S2-s36-adp_cp-s5" , "adaptive_hsp_plate-S2-s36-adp_cp-s5"]
    #exps = ["hsp_all_shared"]
    #exps = ["hsp_plate-S2-s36-adp_cp-s5","adaptive_hsp_plate-S2-s36-adp_cp-s5",
    #        "mep-S2-s36-adp_cp-s5", "adaptive_mep-S2-s36-adp_cp-s5"]
    #algs = ["bias"]
    #exps = ["hsp"]
    seed_max = [5]
    # seed_max = [72, 72, 72]
    #seed_max = [20, 10, 10]
    #seed_max = [1, 1, 1]
    #exps = [e for e in range(1,4)]
    #ranks = [r for r in range(1)]
    #run = 1
    #traj_num = 1
    
    #is_self = [True , True, True]
    is_self = [False, False]
    #partner_agents = [30, 30, 30, 30]
    partner_agents = [10]
    #self_agent_name = ["hsp_cp", "hsp_cp", "hsp_cp", "hsp_cp"]
    self_agent_name = ["hsp"]
    
    #partner_agents = [1, 1, 1]
    #self_agents = ["hsp","hsp_cp","hsp_cp"]
    
    init_dir(save_root, exps)
    
    dfs = [] # index : alg
    index = 0
    
    all_match_ups = []
    all_partners = []
    all_match_up_for_hist = []
     
    for seeds, self_play, self_agent, partner_num in zip(seed_max, is_self, self_agent_name, partner_agents):
        match_ups = []
        partners = []
        match_up_for_hist = []
        
        if self_play:
            for i in range(seeds):
                match_ups.append([
                    f"{self_agent}{i}_final_w0-{self_agent}{i}_final_w1",
                    f"{self_agent}{i}_final_w1-{self_agent}{i}_final_w0",
                ])
                partners.append(["self_w0-self_w1", "self_w1-self_w0"])
            match_up_for_hist.append("self_w0-self_w1")
            match_up_for_hist.append("self_w1-self_w0")
        else:
            for i in range(seeds):
                m_per_seed = []
                p_per_seed = []
                for j in range(int(partner_num/2)):
                    m_per_seed.append(f"{self_agent}-{i}-bias{j}_final")
                    p_per_seed.append(f"self-bias{j}")
                for j in range(int(partner_num/2), partner_num):
                    m_per_seed.append(f"bias_final{j}-{self_agent}-{i}")
                    p_per_seed.append(f"bias{j}-self")
                match_ups.append(m_per_seed)
                partners.append(p_per_seed)
            match_up_for_hist = partners[0].copy()
            
        all_match_ups.append(match_ups)
        all_partners.append(partners)
        all_match_up_for_hist.append(match_up_for_hist)
        
    logger.debug(all_match_up_for_hist)
    
    for seeds, exp, match_ups in zip(seed_max, exps, all_partners):
        
        df = load_behavior(layout, exp, seeds, match_ups, use_new, save_root, _collect_gif, _gif_from_traj)
        #df = load_trajectory(layout, alg, exps, ranks, run, traj_num, True)
        # logger.debug(df)
        dfs.append(df)
        
        index += 1
    
    df_all = pd.concat(dfs, axis=0)
    df_all.to_csv(f"{save_root}/preprocess.csv")
    
    logger.debug(df_all)

    # df_cs = compute_cos_similarity(df_all)
    
    # logger.debug(df_cs)
    
    # sns.heatmap(df_cs)
    # plt.savefig(f"eval/results/{layout}/cos_similarity.png", dpi=300)
    
    plt.close()
    
    df_agent0 = df_all.filter(like="agent0", axis=1)
    df_agent1 = df_all.filter(like="agent1", axis=1)
    
    df_agent0.to_csv(f"{save_root}/agent_0_behavior_metrics.csv")
    df_agent1.to_csv(f"{save_root}/agent_1_behavior_metrics.csv")
    
    if _plot_radar:
        plot_radar(df_agent0, exps, seed_max,"agent0", save_root)
        plot_radar(df_agent1, exps, seed_max,"agent1", save_root)
    
    if _plot_hist:
        plot_histgrams(df_all, layout, exps, all_match_up_for_hist, save_root)
    
    if _plot_pca:
        plot_pca(df_all, exps, layout, _plot_label, save_root)
    
    
    
    
    
    
