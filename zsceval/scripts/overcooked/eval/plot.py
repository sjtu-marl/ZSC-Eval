import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from adjustText import adjust_text

import sys
import json
import pickle
import os
import glob
import shutil
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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


def load_behavior(layout, alg, exp, seed_max, use_new):
    
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
                #print(name)
                if not "either" in name:
                   values.append(json_dict[name])
            json_values.append(values)
            
        summon_gif(layout, exp, i)
    
    LABELS = sorted(LABELS_NEW) if use_new else sorted(LABELS_OLD)
    
    print(len(LABELS))
    print(f"{len(json_values)}, {len(json_values[0])}")
    
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
                #print(name)
                if not "either" in name:
                   values.append(json_dict[name])
            json_values.append(values)
            
        summon_gif(layout, exp, i)
    
    LABELS = sorted(LABELS_NEW) if use_new else sorted(LABELS_OLD)
    
    print(len(LABELS))
    print(f"{len(json_values)}, {len(json_values[0])}")
    
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
    
    for exp in exps:
        run_dir = Path(os.path.expanduser("~") + "/ZSC/results/Overcooked/" + layout_name + "/population/eval-" + alg + str(exp) + "/run" + str(run)) 
        for rank in ranks:
            traj = None
            with open(f"{run_dir}/trajs_store/{layout_name}/traj_{rank}_{traj_num}.pkl",'rb') as f:
                traj = pickle.load(f)
            
            print(len(traj))
    
def plot_pca(df_all, exps, layout):
    
    std_scaler = MinMaxScaler()
    std_scaler.fit(df_all.to_numpy())
    
    df_std = std_scaler.transform(df_all)
    
    df_std = pd.DataFrame(df_std, index=df_all.index, columns=df_all.columns)
    
    print(df_std)
    
    pca = PCA()
    pca.fit(df_std)
    
    score = pd.DataFrame(pca.transform(df_std))

    #print(score)
    
    num = len(score)  # 可視化するデータ数を指定
    
    cmap = plt.get_cmap("tab10")
    
    plt.rcParams["font.size"] = 5
    
    texts = []
    
    # プロットしたデータにサンプル名をラベリング
    for i in range(num):
        alg_index = None
        for j, alg in enumerate(exps):
            alg_index = j if alg in df_all.index.values[i] else alg_index
        
        plt.scatter(score.iloc[i][0], score.iloc[i][1], c=cmap(alg_index)) 
        texts.append(plt.text(score.iloc[i,0], score.iloc[i,1], df_all.index[i], horizontalalignment="center", verticalalignment="bottom"))
    
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
    
    for i, exp in enumerate(exps):
        plt.scatter([], [], c=cmap(i), alpha=0.5, label=exp)
            
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.xlabel("1st principal component")
    plt.ylabel("2nd principal component")
    plt.grid()
    plt.savefig(f"eval/results/{layout}/pca.png", dpi=300)
    
    plt.close()
    
    
def compute_cos_similarity(df_all):
    
    std_scaler = StandardScaler()
    std_scaler.fit(df_all.to_numpy())

    df_1 = pd.DataFrame(std_scaler.transform(df_all), index=df_all.index, columns=df_all.columns)
    df_2 = pd.DataFrame(std_scaler.transform(df_all), index=df_all.index, columns=df_all.columns)
    
    #print(df_1)
    
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
    
    
def plot_radar(df, layout, subscript):
    
    #df_no_zero = df.loc[:, (df != 0).any(axis=0)]
    df_no_zero = df
    
    label_list = df_no_zero.columns.values
    index_list = df_no_zero.index
    
    scaler = MinMaxScaler()
    #print(scaler.fit_transform(df_no_zero))
    df_scaler = pd.DataFrame(scaler.fit_transform(df_no_zero), columns=label_list)
    
    df_scaler.to_csv(f"eval/results/{layout}/min_max_scale.csv")
    
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
        
        print(index_list[i])
         
        plt.savefig(f"eval/results/{layout}/radar_{index_list[i]}_{subscript}.png")
        plt.close()
        
def summon_gif(layout, alg, index):
    
    run_dir = Path(os.path.expanduser("~") + "/ZSC/results/Overcooked/" + layout + "/population/eval-" + alg + str(index) + "/**/*.gif" )
    
    print(str(run_dir))
    
    gifs =  glob.glob(str(run_dir), recursive=True)
    
    if(len(gifs)==0):
        print(f"No gif for {alg}{str(index)}")
        return
    
    shutil.copy(gifs[0], f"eval/results/{layout}/{alg}{str(index)}.gif")
    print(f"gif saved at eval/results/{layout}/{alg}{str(index)}.gif")

if __name__ == "__main__":
    
    layout = sys.argv[1]
    algs = ["sp","mep", "bias"]
    exps = ["sp","mep", "hsp"]
    #algs = ["bias"]
    #exps = ["hsp"]
    seed_max = [5, 5, 72]
    
    #exps = [e for e in range(1,4)]
    #ranks = [r for r in range(1)]
    #run = 1
    #traj_num = 1
    
    dfs = [] # index : alg
    index = 0
    for seeds, alg, exp in zip(seed_max, algs, exps):
    
        df = load_behavior(layout, alg, exp, seeds, True)
        #df = load_trajectory(layout, alg, exps, ranks, run, traj_num, True)
        
        df.to_csv(f"eval/results/{layout}/{alg}/preprocess.csv")
        
        dfs.append(df)
        
        index += 1
        
    df_all = pd.concat(dfs, axis=0)
    
    #print(df_all)

    df_cs = compute_cos_similarity(df_all)
    
    # print(df_cs)
    
    sns.heatmap(df_cs)
    plt.savefig(f"eval/results/{layout}/cos_similarity.png", dpi=300)
    
    plt.close()
    
    df_agent0 = df_all.filter(like="agent0", axis=1)
    df_agent1 = df_all.filter(like="agent1", axis=1)
    
    print(df_agent1)
    
    df_agent0.to_csv(f"eval/results/{layout}/behavior_metrics.csv")
    df_agent1.to_csv(f"eval/results/{layout}/behavior_metrics.csv")
    
    plot_radar(df_agent0, layout, "agent0")
    plot_radar(df_agent1, layout, "agent1")
    
    plot_pca(df_all, exps, layout)
    
    
    
    
    
    
