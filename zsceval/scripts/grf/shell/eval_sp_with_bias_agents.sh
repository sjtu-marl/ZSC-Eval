#!/bin/bash
env="GRF"

# academy_3_vs_1_with_keeper
scenario=$1
num_agents=$2

algo="population"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

algorithm="fcp"
exp="sp"

factorial() {
    local n=$1
    local result=1
    for ((i=1; i<=n; i++)); do
        result=$((result * i))
    done
    echo $result
}

permutation() {
    local n=$1
    local k=$2
    local n_fact=$(factorial $n)
    local nk_fact=$(factorial $((n-k)))
    echo $((n_fact / nk_fact))
}

num_combs() {
    local num_pops=$1
    local num_agents=$2
    local perms=$(permutation $num_pops 2)
    echo $((num_pops * num_agents * 2 + perms * num_agents))
}

bias_agent_version="hsp"

declare -A LAYOUTS_KS
LAYOUTS_KS["academy_3_vs_1_with_keeper"]=3
K=$((2 * LAYOUTS_KS[${scenario}]))


n_combs=$(num_combs $K $num_agents)
echo "Number of combinations: $n_combs"

path=../../policy_pool
export POLICY_POOL=${path}

K=$((2 * LAYOUTS_KS[${scenario}]))
bias_yml="${path}/${scenario}/hsp/s1/${bias_agent_version}/benchmarks-s${K}.yml"
yml_dir=eval/eval_policy_pool/${scenario}/results/
mkdir -p ${yml_dir}

n=$(grep -o -E 'bias.*_(final|mid):' ${bias_yml} | wc -l)
echo "Evaluate ${scenario} with ${n} agents"
population_size=$((n + 1))

ulimit -n 65536

for seed in $(seq 1 3); do
    exp_name="${exp}"
    agent_name="${exp_name}-${seed}"

    eval_exp="eval-${agent_name}"
    yml=${yml_dir}/${eval_exp}.yml

    pt_name="${exp}${seed}_final_actor"

    sed -e "s/agent_name/${agent_name}/g" -e "s/rnn_policy_config/mlp_policy_config/g" -e "s/algorithm/${algorithm}/g" -e "s/\/s2/\/s1/g" -e "s/population/${exp_name}/g" -e "s/seed/${pt_name}/g" "${bias_yml}" > "${yml}"

    python eval/eval_with_population.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${eval_exp}" --scenario_name "${scenario}" \
    --num_agents ${num_agents} --seed 1 --episode_length 200 --n_eval_rollout_threads $((n_combs * 10)) --eval_episodes $((n_combs * 40)) --eval_stochastic --dummy_batch_size 2 \
    --use_proper_time_limits \
    --use_wandb \
    --population_yaml_path "${yml}" --population_size ${population_size} \
    --eval_result_path "eval/results/${scenario}/sp/${eval_exp}.json" \
    --agent_name "${agent_name}"
done
