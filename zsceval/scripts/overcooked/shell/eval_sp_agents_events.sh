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

path=../../policy_pool

export POLICY_POOL=${path}

policy_version="sp"

echo "env is ${env}, layout is ${layout}, eval"
n=$(find ${path}/${layout}/fcp/s1/${policy_version} -name "*_final_actor.pt" | wc -l)
echo "Evaluate $n agents in ${path}/${layout}/sp/s1/${policy_version}"
yml_dir=eval/eval_policy_pool/${layout}/sp_pop
mkdir -p ${yml_dir}

eval_template="eval_template"

for i in $(seq 1 ${n});
do
    agent0_policy_name="sp${i}_final"
    agent1_policy_name="sp${i}_final"
    exp="eval-sp${i}"
    yml=${yml_dir}/${exp}.yml
    
    sed -e "s/agent0/${agent0_policy_name}/g" -e "s/agent1/${agent1_policy_name}/g" -e "s/pop/${policy_version}/g" ${path}/${layout}/fcp/s1/sp/${eval_template}.yml > ${yml}
    
    echo "########################################"
    echo "evaluate ${agent0_policy_name}-${agent1_policy_name}"
    python eval/eval.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} \
    --num_agents ${num_agents} --seed 1 --episode_length 400 --n_eval_rollout_threads 40 --eval_episodes 80 --eval_stochastic --dummy_batch_size 2 \
    --use_proper_time_limits \
    --use_wandb \
    --use_render \
    --population_yaml_path ${yml} --population_size 2 \
    --agent0_policy_name ${agent0_policy_name} \
    --agent1_policy_name ${agent1_policy_name} --overcooked_version ${version} --eval_result_path eval/results/${layout}/sp/${exp}.json
    echo "########################################"
done
