#!/bin/bash
env="GRF"

# academy_3_vs_1_with_keeper
scenario=$1
num_agents=$2

entropy_coefs="0.02 0.01 0.01"
entropy_coef_horizons="0 1.8e7 3.6e7"

num_env_steps="3.6e7"

algo="cole"
exp="cole"
population_size=15
pop="s${population_size}"
seed_begin=1
seed_max=3
path=../../policy_pool

train_batch=200
ulimit -n 65536 || ulimit -n 4096

export POLICY_POOL=${path}
export EVOLVE_ACTOR_POOL="${HOME}/ZSC/tmp"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7


echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed from ${seed_begin} to ${seed_max}"
for seed in $(seq ${seed_begin} ${seed_max});
do
    python train/train_cole.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}-${pop}" --scenario_name ${scenario} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --num_mini_batch 1 --episode_length 200 --num_env_steps ${num_env_steps} --train_env_batch ${train_batch} --n_rollout_threads ${train_batch} --eval_stochastic --dummy_batch_size 1 \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --representation "simple115v2_custom" --rewards "scoring,checkpoints" --reward_config '{"score": 5.0, "checkpoints": 1.0}' \
    --use_proper_time_limits \
    --stage 2 \
    --num_generation $((population_size * 2)) --generation_interval 30 --prioritized_alpha 1.25 --cole_ucb_factor 1 \
    --algorithm_type evolution \
    --population_yaml_path ${path}/${scenario}/cole/s1/train-${pop}-${seed}.yml \
    --population_size ${population_size} --adaptive_agent_name cole_adaptive --use_agent_policy_id \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads 100 --eval_episodes 5 \
    --use_wandb
done
