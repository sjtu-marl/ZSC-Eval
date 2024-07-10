#!/bin/bash
env="Overcooked"

layout=$1
population_size=$2

if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi
entropy_coefs="0.2 0.05 0.01"
entropy_coef_horizons="0 6e6 1e7"
if [[ "${layout}" == "small_corridor" ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 8e6 1e7"
fi

reward_shaping_horizon="1e8"
num_env_steps="1e7"

num_agents=2
algo="traj"
stage="S1"
exp="traj-${stage}-s${population_size}"
seed=1



train_batch=250

ulimit -n 65536
path=../../policy_pool
export POLICY_POOL=${path}

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, seed is ${seed}, stage is ${stage}"
python train/train_traj.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" --layout_name ${layout} --num_agents ${num_agents} \
--seed ${seed} --n_training_threads 1 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
--train_env_batch ${train_batch} --n_rollout_threads ${train_batch} --dummy_batch_size 1 \
--overcooked_version ${version} \
--ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
--save_interval 25 --log_interval 1 --use_eval --eval_interval 20 --n_eval_rollout_threads $((population_size * 2)) --eval_episodes 10 \
--stage 1 \
--traj_entropy_alpha 0.1 --traj_gamma 0.5 --traj_entropy_alpha 0.1 \
--population_yaml_path ${path}/${layout}/traj/s1/train-s${population_size}.yml \
--population_size ${population_size} --adaptive_agent_name traj_adaptive \
--use_proper_time_limits \
--wandb_name "hogebein"