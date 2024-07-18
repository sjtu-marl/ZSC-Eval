#!/bin/bash
env="GRF"

# academy_3_vs_1_with_keeper
scenario=$1
num_agents=$2

entropy_coefs="0.02 0.01 0.01"
entropy_coef_horizons="0 2.5e6 5e6"

num_env_steps="5e6"

algo="traj"
stage="S1"
population_size=5
exp="traj-${stage}-s${population_size}"
seed=1
path=../../policy_pool

train_batch=125
ulimit -n 65536 || ulimit -n 4096

export POLICY_POOL=${path}
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed ${seed}"
python train/train_traj.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" --scenario_name ${scenario} --num_agents ${num_agents} \
--seed 1 --n_training_threads 1 --num_mini_batch 1 --episode_length 200 --num_env_steps ${num_env_steps} --train_env_batch ${train_batch} --n_rollout_threads ${train_batch} --eval_stochastic --dummy_batch_size 1 \
--ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
--representation "simple115v2_custom" --rewards "scoring,checkpoints" --reward_config '{"score": 5.0, "checkpoints": 1.0}' \
--use_proper_time_limits \
--stage 1 \
--traj_entropy_alpha 0.1 --traj_gamma 0.5  \
--population_yaml_path ${path}/${scenario}/traj/s1/train-s${population_size}.yml \
--population_size ${population_size} --adaptive_agent_name traj_adaptive \
--save_interval 25 --log_interval 2 --use_eval --eval_interval 20 --n_eval_rollout_threads $((population_size * 10)) --eval_episodes 50 \
--use_wandb
