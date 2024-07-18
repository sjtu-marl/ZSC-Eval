
#!/bin/bash
env="GRF"

# academy_3_vs_1_with_keeper
scenario=$1
num_agents=$2

entropy_coefs="0.02 0.01 0.01"
entropy_coef_horizons="0 1.5e7 3e7"

num_env_steps="3e7"

algo="adaptive"
stage="S2"
population_size=9
exp="fcp-${stage}-s${population_size}"
pop="s${population_size}"
seed_begin=1
seed_max=3
path=../../policy_pool

export POLICY_POOL=${path}
train_batch=200

ulimit -n 65536 || ulimit -n 4096
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, pop is ${pop}, exp is ${exp}, seed from ${seed_begin} to ${seed_max}, stage is ${stage}"
for seed in $(seq ${seed_begin} ${seed_max});
do
    python train/train_adaptive.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" --scenario_name ${scenario} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads ${train_batch} --eval_stochastic --dummy_batch_size 1 --num_mini_batch 1 --episode_length 200 --num_env_steps ${num_env_steps} \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --representation "simple115v2_custom" --rewards "scoring,checkpoints" --reward_config '{"score": 5.0, "checkpoints": 1.0}' \
    --use_proper_time_limits \
    --stage 2 \
    --population_yaml_path ${path}/${scenario}/fcp/s2/train-s${population_size}-${pop}-${seed}.yml \
    --population_size ${population_size} --adaptive_agent_name fcp_adaptive --use_agent_policy_id \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads 100 --eval_episodes 5 \
    --use_wandb
done
