#!/bin/bash
env="Overcooked"

layout=$1
if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi

entropy_coefs="0.2 0.05 0.01"
entropy_coef_horizons="0 5e6 1e7"
if [[ "${layout}" == "small_corridor" ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 8e6 1e7"
fi

reward_shaping_horizon="1e8"
num_env_steps="1e7"

num_agents=2
algo="adaptive"

path="../../policy_pool"
export POLICY_POOL=${path}

policy_version="hsp"

yml_dir=${path}/${layout}/hsp/s1/${policy_version}
n=$(find ${yml_dir} -name "train_br_*.yml" | wc -l)

echo "Train $n BR agents"

n_training_threads=125
population_size=1

for (( i=1; i<=$n; i++ ))
# for (( i=1; i<=$n/2; i++ ))
# for (( i=$n/2+1; i<=$n; i++ ))
do
    exp="br"
    yml=${yml_dir}/train_${exp}_${i}.yml
    
    python train/train_adaptive.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" --layout_name ${layout} --num_agents ${num_agents} \
    --seed ${i} --n_training_threads 1 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    --overcooked_version ${version} \
    --n_rollout_threads ${n_training_threads} --dummy_batch_size 1 \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --stage 2 \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 10 --n_eval_rollout_threads 50 --eval_episodes 50 --eval_stochastic \
    --population_yaml_path ${yml} \
    --population_size ${population_size} --adaptive_agent_name "br_agent" \
    --use_proper_time_limits \
    --wandb_name "hogebein" \
    --eval_result_path eval/results/${layout}/bias/eval-${exp}_${i}.json
done