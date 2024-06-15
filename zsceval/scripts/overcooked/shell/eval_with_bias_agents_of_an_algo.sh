algo=$1

layouts=("random0" "random0_medium" "random1" "random3" "unident_s" "small_corridor" "random0_m" "random1_m")

if [[ $algo != "sp" && $algo != "e3t" ]];
then
    for layout in "${layouts[@]}"; do
        bash shell/eval_with_bias_agents.sh ${layout} ${algo}
    done
else
    for layout in "${layouts[@]}"; do
        bash shell/eval_${algo}_with_bias_agents.sh ${layout}
    done
fi