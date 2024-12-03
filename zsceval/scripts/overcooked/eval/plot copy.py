import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import math

from adjustText import adjust_text

import sys
import json
import pickle
import os
import glob
import shutil
from pathlib import Path
import re

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from loguru import logger



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
    
    "sparse_r",
    "shaped_r",
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
    
    data_dir = Path(os.path.expanduser("~") + "/ZSC/results/Overcooked/" + layout + "/population/eval-" + exp + "_" + str(index))
    
    gifs =  glob.glob(str(data_dir) + "/**/*.gif" , recursive=True)
    
    if(len(gifs)==0):
        logger.debug(f"No gif for {exp}{str(index)}")
        return
    
    reward = re.findall(r'\d+', gifs[0])
    
    if not os.path.exists(f"{save_path}/gifs"):
        os.makedirs(f"{save_path}/gifs")
    
    shutil.copy(gifs[0], f"{save_path}/gifs/{exp}_{str(index)}-reward_{reward[-1]}.gif")
    logger.debug(f"gif saved at {save_path}/gifs/{exp}_{str(index)}-reward_{reward[-1]}.gif")  

def render_gif_from_traj(save_path, layout, exp, index):
    data_dir = Path(os.path.expanduser("~") + "/ZSC/results/Overcooked/" + layout + "/population/eval-" + exp + "_" + str(index))
    trajs =  glob.glob(str(data_dir) + "/**/*.json" , recursive=True)

    
    

def load_behavior(layout, exp, match_ups, use_new, save_root, collect_gif, gif_from_traj):
    
    json_values = []
    index = []
    
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]
    
    files = glob.glob(f"eval/results/{layout}/{exp}/*.json")
    sorted_files = sorted(files, key=natural_keys)
    # logger.debug(sorted_files)
    
    for i in range(0,len(match_ups)):
        #if(i is 25 or i is 26):
        #    continue
        file = sorted_files[i]
        index.append(f"{exp}_{i+1}")
        with open(file, mode="rt", encoding="utf-8") as f:
            json_dict = json.load(f)
            orderedNames = sorted(list(json_dict.keys()))
            logger.debug(len(json_dict.keys()))
            values = []
            for name in orderedNames:
                #logger.debug(name)
                if not "either" in name:
                   values.append(json_dict[name])
            logger.debug(len(values))
            json_values.append(values)
        #logger.debug(save_root + exp)
        if collect_gif:
            summon_gif(save_root + exp, layout, exp, i+1)
    
    LABELS = sorted(LABELS_NEW) if use_new else sorted(LABELS_OLD)
    
    #logger.debug(len(LABELS))
    #logger.debug(f"{len(json_values)}, {len(json_values[0])}")
    
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
    plt.savefig(f"{save_root}/pca.png", dpi=300, bbox_inches='tight')
    logger.debug(f"saved pca plot at {save_root}/pca.png")
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
        
        plt.savefig(f"{save_root}/{exps[exp_index]}/radars/radar_{index_list[i]}_{subscript}.png")
        plt.close()
        

    
    
def dump_hist(df, df_atr, bins, xlabel, ylabel, save_dir, subscript, color):
    plt.rcParams["font.size"] = 20
    
    if(np.isnan(df_atr.min()) or np.isnan(df_atr.max()) or df_atr.min() >= df_atr.max()):
        return
            
    plt.hist(df_atr, bins=bins, range=(df_atr.min(), df_atr.max()), color=color)
    
    plt.ylim(0, len(df_atr))
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    
    #plt.xticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{subscript}.png", dpi=300)
    plt.close() 

def dump_hist_comp(df_atrs, exps, bins, xlabel, ylabel, save_dir, subscript, colors):
    fig = plt.figure(figsize=(20,10))
    fig.suptitle("")
    
    x_max = -10000000000
    x_min = 10000000000
    for df_atr in df_atrs:
        if df_atr.min() < x_min:
            x_min = df_atr.min()
        if df_atr.max() > x_max:
            x_max = df_atr.max()
            
    if(not np.isnan(x_min) or not np.isnan(x_max) or x_min >= x_max):
        return
    
    for i, df_atr in enumerate(df_atrs):
        ax = fig.add_subplot(2, 2, i+1)
        ax.hist(df_atr, bins=bins, range=(x_min, x_max), color=colors[i])
        ax.set_title(exps[i])
        ax.set_ylim(0, len(df_atr))
        ax.set_xlabel(xlabel=xlabel)
        ax.set_ylabel(ylabel=ylabel)
        
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{subscript}.png", dpi=300)
    plt.close() 
    

def plot_histgrams(df, layout, exps, save_root):
        
    df_agent0 = df.filter(like="agent0", axis=1)
    df_agent1 = df.filter(like="agent1", axis=1)
    
    for j, column in enumerate(df_agent0.columns):
        df_all[column.replace("agent0", "all")] = df_agent0.iloc[:,j] + df_agent1.iloc[:,j] 
        
    dfs = {"all":df_all,"agent0":df_agent0, "agent1":df_agent1}
    
    for key, value in dfs.items():
        cmap = plt.get_cmap("tab20")
        colors = []
        for i in range(10):
            # no orange
            if i == 1:
                continue
            if i % 2 == 1:
                colors.append(cmap(2*i))
            else:
                colors.append(cmap(2*i))
        
        df_dish_remain = value[f"put_dish_on_X_by_{key}"] - value[f"pickup_dish_from_X_by_{key}"]
        df_soup_remain = value[f"put_soup_on_X_by_{key}"] - value[f"pickup_soup_from_X_by_{key}"]
        df_plate_remain = value[f"put_dish_on_X_by_{key}"] - value[f"pickup_dish_from_X_by_{key}"] + value[f"put_soup_on_X_by_{key}"] - value[f"pickup_soup_from_X_by_{key}"]
        df_stay = value[f"STAY_by_{key}"]
        df_movement = value[f"MOVEMENT_by_{key}"]
        df_onion = value[f"potting_onion_by_{key}"]
        df_tomato = value[f"potting_tomato_by_{key}"]
        df_size_2 = value[f"deliver_size_two_order_by_{key}"]
        df_size_3 = value[f"deliver_size_three_order_by_{key}"]
        df_reward = value[f"sparse_r_by_{key}"]
        
        df_put_onion_on_X = value[f"put_onion_on_X_by_{key}"]
        df_put_tomato_on_X = value[f"put_tomato_on_X_by_{key}"]
        df_put_dish_on_X = value[f"put_dish_on_X_by_{key}"]
        df_put_soup_on_X = value[f"put_soup_on_X_by_{key}"]
        
        df_pickup_onion_from_X = value[f"pickup_onion_from_X_by_{key}"]
        df_pickup_tomato_from_X = value[f"pickup_tomato_from_X_by_{key}"]
        df_pickup_dish_from_X = value[f"pickup_dish_from_X_by_{key}"]
        df_pickup_soup_from_X = value[f"pickup_soup_from_X_by_{key}"]
        
        df_pickup_onion_from_O = value[f"pickup_onion_from_O_by_{key}"]
        df_pickup_tomato_from_T = value[f"pickup_tomato_from_T_by_{key}"]
        df_pickup_dish_from_D = value[f"pickup_dish_from_D_by_{key}"]
        df_SOUP_PICKUP = value[f"SOUP_PICKUP_by_{key}"]
        
        
        pd.set_option('display.max_rows', None)
        logger.debug(df_reward)
        
        df_atrs = {"dish_remain" : df_dish_remain, "soup_remain" : df_soup_remain,
                   "plate_remain" : df_plate_remain, "movement" : df_movement, "onion" : df_onion,
                   "tomato" : df_tomato,  "size_2" : df_size_2,
                   "size_3" : df_size_3, "reward" : df_reward, "stay" : df_stay,
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
                   "onion" : "Counts for cooking onions",
                   "tomato" : "Counts for cooking tomatos",
                   "size_2" : "Counts for delivering size 2 recipe",
                   "size_3" : "Counts for delivering size 3 recipe",
                   "reward" : "Final scores",
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
                   "plate_remain" : 20, "movement" : 20, "onion" : 20,
                   "tomato" : 20, "size_2" : 20,
                   "size_3" : 20, "reward" : 20, "stay" : 20,
                   "put_onion_on_X" : 20, "put_tomato_on_X" :20,  "put_dish_on_X" :20, "put_soup_on_X" :20,
                   "pickup_onion_from_X" : 20, "pickup_tomato_from_X" : 20, "pickup_dish_from_X" : 20,
                   "pickup_soup_from_X" : 20, "pickup_onion_from_O" : 20, "pickup_tomato_from_T" : 20,
                   "pickup_dish_from_D" : 20, "pickup_soup_from_P" : 20}
        ylabel = "Number of AIs"
        
        
        if key == "all":
            
            plates_remain_agent0 = df_agent0[f"put_dish_on_X_by_agent0"] - df_agent0[f"pickup_dish_from_X_by_agent0"]
            plates_remain_agent1 = df_agent1[f"put_dish_on_X_by_agent1"] - df_agent1[f"pickup_dish_from_X_by_agent1"]
            
            plates_pickup_agent0 = df_agent0[f"pickup_dish_from_X_by_agent0"] - df_agent0[f"put_dish_on_X_by_agent0"]
            plates_pickup_agent1 = df_agent1[f"pickup_dish_from_X_by_agent1"] - df_agent1[f"put_dish_on_X_by_agent1"]
            
            df_atrs["plates_coord_0_to_1"] = plates_pickup_agent1
            df_atrs["plates_coord_1_to_0"] = plates_pickup_agent0
            df_atrs["plates_coord_total"] = df_atrs["plates_coord_0_to_1"] + df_atrs["plates_coord_1_to_0"]
            
            atrs_disc["plates_coord_0_to_1"] = "Counts for plate pickup - plate placement"
            atrs_disc["plates_coord_1_to_0"] = "Counts for plate pickup - plate placement"
            atrs_disc["plates_coord_total"] = "Counts for plate pickup - plate placement"
            
            atrs_bins["plates_coord_0_to_1"] = 20
            atrs_bins["plates_coord_1_to_0"] = 20
            atrs_bins["plates_coord_total"] = 20
            
        
        df_all_sorted= pd.DataFrame()
        
        for atr_label, df_atr in df_atrs.items():
            
            hist_dir = f"{save_root}histgrams"
            if not os.path.exists(hist_dir):
                os.makedirs(hist_dir)
            
            df_sorted = df_atr.sort_values(ascending=False)
            df_all_sorted[f"{atr_label}_i"] = df_sorted.index
            df_all_sorted[f"{atr_label}_v"] = df_sorted.values
            
            df_atrs_filtered = []
            for i, exp in enumerate(exps):
                df_atrs_filtered.append(df_atr.filter(regex=f"^{exp}(\d)+$", axis=0))
            
            dump_hist_comp(df_atrs_filtered, exps, bins=atrs_bins[atr_label],
                          xlabel=atrs_disc[atr_label], ylabel=ylabel,
                          save_dir=hist_dir, subscript=f"hist_{atr_label}_{key}",
                          colors=colors)
            
        df_all_sorted.to_csv(f"{save_root}/ranking_all_{key}.csv")
        
        for i, exp in enumerate(exps):
            
            hist_dir = f"{save_root}{exp}/histgrams"
            ranking_dir = f"{save_root}{exp}/rankings"

            if not os.path.exists(hist_dir):
                os.makedirs(hist_dir)
                
            if not os.path.exists(ranking_dir):
                os.makedirs(ranking_dir)

            df_sorted = pd.DataFrame()
            
            for atr_label, df_atr in df_atrs.items():
                # need to filter
                df_atr_filtered = df_atr.filter(regex=f"^{exp}(\d)+$", axis=0)
                
                #logger.debug(type(df_exp))
                
                dump_hist(value, df_atr_filtered, bins=atrs_bins[atr_label],
                          xlabel=atrs_disc[atr_label], ylabel=ylabel,
                          save_dir=hist_dir, subscript=f"hist_{atr_label}_{exp}_{key}",
                          color=colors[i])
            
                df_sorted[f"{atr_label}_i"] = df_atr_filtered.sort_values(ascending=False).index
                df_sorted[f"{atr_label}_v"] = df_atr_filtered.sort_values(ascending=False).values

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
    
    _collect_gif = False
    
    _gif_from_traj = False
    
    _plot_radar = False
    
    _plot_hist = False
    
    _plot_pca = False
    
    save_root = f"./eval/results/{layout}/analysis/"
    
    #exps = ["hsp","adaptive_hsp_plate","adaptive_hsp_plate_vs_hsp_plate"]
    #exps = ["hsp_plate_shared", "hsp_plate_shared-pop_cross_play", "adaptive_hsp_plate_shared-pop_cross_play"]
    exps = ["hsp_plate_shared-pop_cross_play", "hsp_plate-S2-s36-adp_cp-s5" , "adaptive_hsp_plate-S2-s36-adp_cp-s5"]
    #algs = ["bias"]
    #exps = ["hsp"]
    # seed_max = [72, 72, 72]
    seed_max = [20, 10, 10]
    #exps = [e for e in range(1,4)]
    #ranks = [r for r in range(1)]
    #run = 1
    #traj_num = 1
    
    is_self = [True , False, False]
    
    partner_agents = [1, 1, 1]
    self_agents = ["hsp","hsp_cp","hsp_cp"]
    
    init_dir(save_root, exps)
    
    dfs = [] # index : alg
    index = 0
    
    all_match_ups = []
     
    for seeds, self_play, self_agent, partner_num in zip(seed_max, is_self, self_agents, partner_agents):
        match_ups = []
        if self_play:
            for i in range(seeds):
                match_ups.append([
                    f"{self_agent}{i}_final_w0-{self_agent}{i}_final_w1",
                    f"{self_agent}{i}_final_w1-{self_agent}{i}_final_w0",
                ])
        else:
            for i in range(seeds):
                per_seed = []
                for j in range(partner_num):
                    per_seed.append(f"{self_agent}-{i}-bias{j}_final")
                for j in range(partner_num):
                    per_seed.append(f"bias_final{j}-{self_agent}-{i}")
                match_ups.append(per_seed)
        all_match_ups.append(match_ups)        
        
    logger.debug(all_match_ups)
    
    for seeds, exp, match_ups in zip(seed_max, exps, all_match_ups):
        
        df = load_behavior(layout, exp, match_ups, use_new, save_root, _collect_gif, _gif_from_traj)
        #df = load_trajectory(layout, alg, exps, ranks, run, traj_num, True)
        #logger.debug(df)
        dfs.append(df)
        
        index += 1
    
    df_all = pd.concat(dfs, axis=0)
    df_all.to_csv(f"{save_root}{exp}/preprocess.csv")
    
    #logger.debug(df_all)

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
        plot_histgrams(df_all, layout, exps, save_root)
    
    if _plot_pca:
        plot_pca(df_all, exps, layout, False, save_root)
    
    
    
    
    
    
