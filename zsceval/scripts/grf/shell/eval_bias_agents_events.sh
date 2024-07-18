#!/bin/bash
env="GRF"

# academy_3_vs_1_with_keeper
scenario=$1
num_agents=$2


algo="population"
path="../../policy_pool"


export POLICY_POOL=${path}
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

policy_version="hsp"

echo "env is ${env}, scenario is ${scenario}, eval bias agents"
n=$(find ${path}/${scenario}/hsp/s1/${policy_version} -name "*_final_w0_actor.pt" | wc -l)
echo "Evaluate $n agents in ${path}/${scenario}/hsp/s1/${policy_version}"
yml_dir=eval/eval_policy_pool/${scenario}/bias
mkdir -p ${yml_dir}

eval_template="eval_template"

factorial() {
    local n=$1
    local result=1
    for ((i=1; i<=n; i++)); do
        result=$((result * i))
    done
    echo $result
}

n_combs=$(factorial $num_agents)

for i in $(seq 1 ${n});
do
    hsp_name="hsp${i}"
    if [[ $hsp_name =~ [0-9]+ ]]; then
        echo "evaluate" ${hsp_name}
    fi
    pair_name="${hsp_name}_final_w0"

    for (( i=1; i<num_agents; i++ ))
    do
        pair_name="${pair_name} ${hsp_name}_final_w${i}"
    done
    echo "agents ${pair_name}"
    exp="eval-${hsp_name}"
    yml=${yml_dir}/${exp}.yml

    sed_command="sed"
    for (( i=0; i<num_agents; i++ ))
    do
        sed_command+=" -e \"s/agent$i/${hsp_name}_final_w${i}/g\""
    done

    sed_command+=" -e \"s/pop/${policy_version}/g\" ${path}/${scenario}/hsp/s1/${eval_template}.yml > ${yml}"

    eval $sed_command

    echo "########################################"
    python eval/eval.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --seed 1 --episode_length 200 --n_eval_rollout_threads $((n_combs * 20)) --eval_episodes $((n_combs * 40)) --eval_stochastic --dummy_batch_size 1 \
    --use_proper_time_limits \
    --use_wandb \
    --agent_policy_names ${pair_name} \
    --population_yaml_path ${yml} --population_size ${num_agents} \
    --eval_result_path "eval/results/${scenario}/bias/${exp}.json"
    echo "########################################"
done
