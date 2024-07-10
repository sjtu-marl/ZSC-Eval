import os
import pickle
import json
import pprint
import re
import shutil
from pathlib import Path

import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np






def convert_ai_ai(layout_name, alg, exps, ranks, run, traj_num, use_new):
    
    if not use_new:
        from zsceval.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import  StateVisualizer
    else:
        from zsceval.envs.overcooked_new.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
        

    for exp in exps:
        run_dir = Path(os.path.expanduser("~") + "/ZSC/results/Overcooked/" + layout_name + "/population/eval-" + alg + str(exp) + "/run" + str(run)) 
        for rank in ranks:
            save_dir = f"{run_dir}/gifs/traj_{rank}_{traj_num}"
            save_dir = os.path.expanduser(save_dir)
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.mkdir(save_dir)
            traj = {}
            with open(f"{run_dir}/trajs/{layout_name}/traj_{rank}_{traj_num}.pkl",'rb') as f:
                traj = pickle.load(f)
            #print(traj)
            
            StateVisualizer().display_rendered_trajectory(traj, img_directory_path=save_dir, ipython_display=False)
            for img_path in os.listdir(save_dir):
                img_path = save_dir + "/" + img_path
            imgs = []
            imgs_dir = os.listdir(save_dir)

            imgs_dir = sorted(imgs_dir, key=lambda x: int(x.split(".")[0]))

            image_shape = None
            for img_path in imgs_dir:
                img_path = save_dir + "/" + img_path
                img = imageio.imread(img_path)
                if (image_shape == None) or (img.shape == image_shape):
                    imgs.append(img)
                    image_shape = img.shape
                
            imageio.mimsave(save_dir + f'/reward_{traj["ep_returns"][0]}.gif', imgs, duration=0.1)
            imgs_dir = os.listdir(save_dir)
            for img_path in imgs_dir:
                img_path = save_dir + "/" + img_path
                if "png" in img_path:
                    os.remove(img_path)
                    
def convert_human_ai():
    
    from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    
    run_dir = Path(os.path.expanduser("~") + "/ZSC-Eval/zsceval/human_exp/data/")
    traj_path_list = os.listdir(f"{run_dir}/traj/")
    for traj_path in traj_path_list:
        save_dir = f"{run_dir}/gifs/"
        save_dir = os.path.expanduser(save_dir)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        traj = {}
        with open(f"{run_dir}/traj/{traj_path}",'rb') as f:
            traj = json.load(f)
            
        grid = OvercookedGridworld.from_layout_name(traj["mdp_params"][0]).terrain_mtx
        traj["mdp_params"][0]["terrain"] = grid
        
        StateVisualizer().display_rendered_trajectory(traj, img_directory_path=save_dir, ipython_display=False)
        for img_path in os.listdir(save_dir):
            img_path = save_dir + "/" + img_path
        imgs = []
        imgs_dir = os.listdir(save_dir)

        imgs_dir = sorted(imgs_dir, key=lambda x: int(x.split(".")[0]))

        image_shape = None
        for img_path in imgs_dir:
            img_path = save_dir + "/" + img_path
            img = imageio.imread(img_path)
            if (image_shape == None) or (img.shape == image_shape):
                imgs.append(img)
                image_shape = img.shape
            
        imageio.mimsave(save_dir + f'/reward_{traj["ep_returns"][0]}.gif', imgs, duration=0.1)
        imgs_dir = os.listdir(save_dir)
        for img_path in imgs_dir:
            img_path = save_dir + "/" + img_path
            if "png" in img_path:
                os.remove(img_path)
                

layout_name = "random3_m"
alg = "hsp"
exps = [e for e in range(1,4)]
ranks = [r for r in range(1)]
run = 3
traj_num = 1

convert_ai_ai(layout_name, alg, exps, ranks, run, traj_num, True)               
#convert_human_ai()