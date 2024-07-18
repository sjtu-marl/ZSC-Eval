layouts="academy_3_vs_1_with_keeper"
path=../policy_pool/
for layout in ${layouts};
do
    echo ${layout}
    mkdir -p ${path}/${layout}/policy_config
    cp ~/ZSC/results/GRF/${layout}/mappo/store_config_mlp/run1/policy_config.pkl ${path}/${layout}/policy_config/mlp_policy_config.pkl
    cp ~/ZSC/results/GRF/${layout}/rmappo/store_config_rnn/run1/policy_config.pkl ${path}/${layout}/policy_config/rnn_policy_config.pkl
done
