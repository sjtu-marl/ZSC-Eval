scenario=$1
num_agents=$2

bash shell/eval_with_bias_agents.sh ${scenario} fcp ${num_agents}
bash shell/eval_with_bias_agents.sh ${scenario} mep ${num_agents}
bash shell/eval_with_bias_agents.sh ${scenario} traj ${num_agents}
bash shell/eval_with_bias_agents.sh ${scenario} hsp ${num_agents}
bash shell/eval_with_bias_agents.sh ${scenario} cole ${num_agents}

bash shell/eval_e3t_with_bias_agents.sh ${scenario} ${num_agents}
bash shell/eval_sp_with_bias_agents.sh ${scenario} ${num_agents}
