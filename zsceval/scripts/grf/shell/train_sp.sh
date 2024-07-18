#!/bin/bash
env="GRF"

# academy_3_vs_1_with_keeper
scenario=$1
num_agents=$2

entropy_coefs="0.02 0.01 0.01"
entropy_coef_horizons="0 2.5e6 5e6"

num_env_steps="5e6"

algo="mappo"
exp="sp"
seed_begin=1
seed_max=5
ulimit -n 65536

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed from ${seed_begin} to ${seed_max}"
for seed in $(seq ${seed_begin} ${seed_max});
do
    echo "seed is ${seed}"
    python train/train_sp.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario}_superhard --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 125 --eval_stochastic --dummy_batch_size 1 --num_mini_batch 1 --episode_length 200 --num_env_steps ${num_env_steps} \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --representation "simple115v2_custom" --rewards "scoring,checkpoints" --reward_config '{"score": 5.0, "checkpoints": 1.0}' \
    --use_proper_time_limits \
    --layer_N 2 --layer_after_N 1 --use_recurrent_policy \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads 20 \
    --use_wandb
done
