
#!/bin/bash
env="GRF"

# academy_3_vs_1_with_keeper
scenario=$1
num_agents=$2

entropy_coefs="0.02 0.01 0.01"
entropy_coef_horizons="0 5e6 1e7"

num_env_steps="1e7"

algo="adaptive"
path=../../policy_pool
export POLICY_POOL=${path}
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
ulimit -n 65536 || ulimit -n 4096

train_batch=250
policy_version="hsp"

yml_dir=${path}/${scenario}/hsp/s1/${policy_version}
n=$(find ${yml_dir} -name "train_br_*.yml" | wc -l)

echo "Train $n BR agents"

# for (( i=1; i<=n/4; i++ ))
# for (( i=n/4+1; i<=n/2; i++ ))
# for (( i=n/2+1; i<=(n/4)*3 + 2; i++ ))
for (( i=(n/4)*3 + 3; i<=n; i++ ))
do
    exp="br"
    yml=${yml_dir}/train_${exp}_${i}.yml
    number=$(grep -c '^bias' ${yml})
    echo br to $number bias agents
    population_size=$((number))
    python train/train_adaptive.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --seed ${i} --n_training_threads 1 --n_rollout_threads ${train_batch} --eval_stochastic --dummy_batch_size 2 --num_mini_batch 1 --episode_length 200 --num_env_steps ${num_env_steps} \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --representation "simple115v2_custom" --rewards "scoring,checkpoints" --reward_config '{"score": 5.0, "checkpoints": 1.0}' \
    --use_proper_time_limits \
    --stage 2 \
    --population_yaml_path ${yml} \
    --population_size ${population_size} --adaptive_agent_name "br_agent" --use_agent_policy_id \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads 20 --eval_episodes 50 --eval_stochastic \
    --eval_result_path eval/results/${scenario}/bias/eval-${exp}_${i}.json \
    --wandb_name "your wandb name"
done
