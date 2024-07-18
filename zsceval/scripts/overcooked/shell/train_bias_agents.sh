#!/bin/bash
env="Overcooked"

layout=$1

entropy_coefs="0.2 0.05 0.001"
entropy_coef_horizons="0 6e6 1e7"
if [[ "${layout}" == "small_corridor" ]]; then
    entropy_coefs="0.2 0.05 0.001"
    entropy_coef_horizons="0 8e6 1e7"
fi

reward_shaping_horizon="1e7"
num_env_steps="1e7"

num_agents=2
algo="mappo"
stage="S1"
exp="hsp-${stage}"


if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
    # old layouts
    #! positive reward shaping for "[op]_X" may crash the training, be careful
    #! negative reward shaping for "put_X" may be meaningless
    # "put_onion_on_X",
    # "put_dish_on_X",
    # "put_soup_on_X",
    # "pickup_onion_from_X", random0_medium random0_hard
    # "pickup_onion_from_O", all_old
    # "pickup_dish_from_X",
    # "pickup_dish_from_D", all_old
    # "pickup_soup_from_X", random0 random0_medium random0_hard
    # "USEFUL_DISH_PICKUP", default
    # "SOUP_PICKUP", all_old default
    # "PLACEMENT_IN_POT", all_old default
    # "delivery", all_old
    # "STAY", all_old
    # "MOVEMENT",
    # "IDLE_MOVEMENT",
    # "IDLE_INTERACT_X",
    # "IDLE_INTERACT_EMPTY",
    # sparse_reward all_old

    w1="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
    if [[ "${layout}" == "random0" ]]; then
        w0="0,0,0,0,[0:10],0,[0:10],[-20:0],3,5,3,0,[-0.1:0:0.1],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=30
    elif [[ "${layout}" == "random0_medium" ]]; then
        w0="0,0,0,[-20:0],[-20:0:10],0,[0:10],[-20:0],3,5,3,0,[-0.1:0:0.1],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=54
    elif [[ "${layout}" == "small_corridor" ]]; then
        w0="0,0,0,0,[-20:0:5],0,[-20:0:5],0,3,5,3,[-20:0],[-0.1:0],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=124
    else
        w0="0,0,0,0,[-20:0:10],0,[-20:0:10],0,3,5,3,[-20:0],[-0.1:0:0.1],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=176
    fi
else
    version="new"
    # 0 "put_onion_on_X",
    # 1 "put_tomato_on_X",
    # 2 "put_dish_on_X",
    # 3 "put_soup_on_X",
    # 4 "pickup_onion_from_X",
    # 5 "pickup_onion_from_O",
    # 6 "pickup_tomato_from_X",
    # 7 "pickup_tomato_from_T",
    # 8 "pickup_dish_from_X",
    # 9 "pickup_dish_from_D",
    # 10 "pickup_soup_from_X",
    # 11 "USEFUL_DISH_PICKUP",  # counted when #taken_dishes < #cooking_pots + #partially_full_pots and no dishes on the counter
    # 12 "SOUP_PICKUP",  # counted when soup in the pot is picked up (not a soup placed on the table)
    # 13 "PLACEMENT_IN_POT",  # counted when some ingredient is put into pot
    # 14 "viable_placement",
    # 15 "optimal_placement",
    # 16 "catastrophic_placement",
    # 17 "useless_placement",  # pot an ingredient to a useless recipe
    # 18 "potting_onion",
    # 19 "potting_tomato",
    # 20 "cook",
    # 21 "delivery",
    # 22 "deliver_size_two_order",
    # 23 "deliver_size_three_order",
    # 24 "deliver_useless_order",
    # 25 "STAY",
    # 26 "MOVEMENT",
    # 27 "IDLE_MOVEMENT",
    # 28 "IDLE_INTERACT",
    # 29 sparse_reward
    w1="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
    w0="0,0,0,0,0,0,0,0,0,0,0,3,5,3,0,0,0,0,[-20:0],[-20:0],0,0,[-5:0:20],[-15:0:10],0,[-0.1:0:0.1],0,0,0,1"
    seed_begin=1
    seed_max=72
fi

for seed in $(seq ${seed_begin} ${seed_max});
do
    echo "seed is ${seed}:"
    python train/train_bias_agent.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" --layout_name ${layout} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 100 --dummy_batch_size 2 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    --overcooked_version ${version} \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --use_hsp --w0 ${w0} --w1 ${w1} --share_policy --random_index \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
    --use_proper_time_limits \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads 20 \
    --wandb_name "your wandb name"
done
