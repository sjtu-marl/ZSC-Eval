layout=$1
path=../../policy_pool/

echo ${layout}
mkdir -p ${path}/${layout}/policy_config
cp ~/ZSC/results/Overcooked/${layout}/mappo/store_config_mlp/run1/policy_config.pkl ${path}/${layout}/policy_config/mlp_policy_config.pkl
cp ~/ZSC/results/Overcooked/${layout}/rmappo/store_config_rnn/run1/policy_config.pkl ${path}/${layout}/policy_config/rnn_policy_config.pkl
