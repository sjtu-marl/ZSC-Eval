#!/bin/bash
env="Overcooked"

layout=$1

if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi


num_agents=2
algo="population"

algorithm="e3t"
exp="e3t"

bias_agent_version="hsp"

declare -A LAYOUTS_KS
LAYOUTS_KS["random0"]=10
LAYOUTS_KS["random0_medium"]=10
LAYOUTS_KS["random1"]=10
LAYOUTS_KS["random3"]=10
LAYOUTS_KS["small_corridor"]=10
LAYOUTS_KS["unident_s"]=10
LAYOUTS_KS["random0_m"]=15
LAYOUTS_KS["random1_m"]=15
LAYOUTS_KS["random3_m"]=15

path=../../policy_pool
export POLICY_POOL=${path}
ulimit -n 65536

K=$((2 * LAYOUTS_KS[${layout}]))
bias_yml="${path}/${layout}/hsp/s1/${bias_agent_version}/benchmarks-s${K}.yml"
yml_dir=eval/eval_policy_pool/${layout}/results
mkdir -p ${yml_dir}

n=$(grep -o -E 'bias.*_(final|mid):' ${bias_yml} | wc -l)
echo "Evaluate ${layout} with ${n} agents"
population_size=$((n + 1))

for seed in $(seq 1 5); do
    exp_name="${exp}"
    agent_name="${exp_name}-${seed}"
    
    eval_exp="eval-${agent_name}"
    yml=${yml_dir}/${eval_exp}.yml
    
    pt_name="${seed}"
    
    sed -e "s/agent_name/${agent_name}/g" -e "s/rnn_policy_config/mlp_policy_config/g" -e "s/algorithm/${algorithm}/g" -e "s/\/s2/\/s1/g" -e "s/population/${exp_name}/g" -e "s/seed/${pt_name}/g" "${bias_yml}" > "${yml}"
    
    python eval/eval_with_population.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${eval_exp}" --layout_name "${layout}" \
    --num_agents ${num_agents} --seed 1 --episode_length 400 --n_eval_rollout_threads $((n * 20)) --eval_episodes $((n * 40)) --eval_stochastic --dummy_batch_size 2 \
    --use_proper_time_limits \
    --use_wandb \
    --population_yaml_path "${yml}" --population_size ${population_size} \
    --overcooked_version ${version} --eval_result_path "eval/results/${layout}/${algorithm}/${eval_exp}.json" \
    --agent_name "${agent_name}"
done
