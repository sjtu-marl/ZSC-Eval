#!/bin/bash
env="Overcooked"

layout=$1

if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi
entropy_coefs="0.2 0.05 0.001"
entropy_coef_horizons="0 6e6 1e7"

reward_shaping_horizon="1e8"

num_env_steps="1e7"

num_agents=2
seed_begin=1
seed_max=1

algo="mappo"
exp="store_config_mlp"
echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, seed from ${seed_begin} to ${seed_max}"
for seed in $(seq ${seed_begin} ${seed_max});
do
    echo "seed is ${seed}:"
    python train/train_sp.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 50 --dummy_batch_size 1 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    --overcooked_version ${version} \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads 10 \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
    --use_proper_time_limits \
    --use_wandb
done


algo="rmappo"
exp="store_config_rnn"
echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, seed from ${seed_begin} to ${seed_max}"
for seed in $(seq ${seed_begin} ${seed_max});
do
    echo "seed is ${seed}:"
    python train/train_sp.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 50 --dummy_batch_size 1 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    --overcooked_version ${version} \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads 10 \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" \
    --use_proper_time_limits \
    --use_wandb
done
