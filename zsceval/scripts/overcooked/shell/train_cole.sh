
#!/bin/bash
env="Overcooked"

layout=$1
population_size=$2

if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi

if [[ ${population_size} == 25 ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 2.5e7 5e7"
    if [[ "${layout}" == "small_corridor" ]]; then
        entropy_coefs="0.2 0.05 0.01"
        entropy_coef_horizons="0 4e7 5e7"
    fi
    reward_shaping_horizon="5e7"
    num_env_steps="5e7"
    prioritized_alpha=1.5
    generation_interval=20
    pop="s${population_size}"
elif [[ ${population_size} == 50 ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 5e7 1e8"
    if [[ "${layout}" == "small_corridor" ]]; then
        entropy_coefs="0.2 0.05 0.01"
        entropy_coef_horizons="0 8e7 1e8"
    fi
    reward_shaping_horizon="1e8"
    num_env_steps="1e8"
    prioritized_alpha=1.0
    generation_interval=20
    pop="s${population_size}"
elif [[ ${population_size} == 75 ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 7.5e7 1.5e8"
    if [[ "${layout}" == "small_corridor" ]]; then
        entropy_coefs="0.2 0.05 0.01"
        entropy_coef_horizons="0 1.2e8 1.5e8"
    fi
    reward_shaping_horizon="1.5e8"
    num_env_steps="1.5e8"
    prioritized_alpha=0.5
    generation_interval=20
    pop="s${population_size}"
fi


num_agents=2
algo="cole"
exp="cole-S2-s${population_size}"

seed_begin=1
seed_max=5
path="../../policy_pool"


n_training_threads=125

ulimit -n 65536 || ulimit -n 4096
export POLICY_POOL=${path}
export EVOLVE_ACTOR_POOL="${HOME}/ZSC/tmp"

echo "env is ${env}, layout is ${layout}, algo is ${algo}, pop is ${pop}, exp is ${exp}, seed from ${seed_begin} to ${seed_max}"
for seed in $(seq ${seed_begin} ${seed_max});
do
    python train/train_cole.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" --layout_name ${layout} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    --overcooked_version ${version} \
    --n_rollout_threads ${n_training_threads} --dummy_batch_size 1 \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --stage 2 \
    --save_interval 25 --log_interval 1 --use_eval --eval_interval 20 --n_eval_rollout_threads $((population_size * 2 + 1)) --eval_episodes 5 \
    --population_yaml_path ${path}/${layout}/cole/s1/train-${pop}-${seed}.yml \
    --population_size ${population_size} --adaptive_agent_name cole_adaptive --use_agent_policy_id \
    --use_proper_time_limits  \
    --num_generation $((population_size * 2)) --generation_interval ${generation_interval} --prioritized_alpha ${prioritized_alpha} \
    --algorithm_type evolution \
    --wandb_name "hogebein"

done

