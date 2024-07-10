layout=$1

bash shell/eval_with_bias_agents.sh ${layout} fcp
bash shell/eval_with_bias_agents.sh ${layout} mep
bash shell/eval_with_bias_agents.sh ${layout} traj
bash shell/eval_with_bias_agents.sh ${layout} hsp
bash shell/eval_with_bias_agents.sh ${layout} cole

# bash shell/eval_e3t_with_bias_agents.sh ${layout}
# bash shell/eval_sp_with_bias_agents.sh ${layout}
