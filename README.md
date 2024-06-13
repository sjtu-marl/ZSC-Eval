# ZSC-Eval: An Evaluation Toolkit and Benchmark for Multi-agent Zero-shot Coordination

This repository is the official implementation of [ZSC-Eval: An Evaluation Toolkit and Benchmark for Multi-agent Zero-shot Coordination](). 

ZSC-Eval is a comprehensive and convenient evaluation toolkit and benchmark for zero-shot coordination (ZSC) algorithms, including partner candidates generation via behavior-preferring rewards, partners selection via Best-Response Diversity (BR-Div), and ZSC capability measurement via Best-Response Proximity (BR-Prox).

We also provide the human evaluation benchmark results and comprehensive benchmark results with ZSC-Eval's generated evaluation partners. Further details please refer to our paper.


**Complete by June 12th**



## 📖 Requirements

To install requirements:

```shell
# install benchmark and overcooked env
conda create -f environment.yml

# install grf
./install_grf.sh

# install bc for runing shell scripts
sudo apt update
sudo apt install bc -y
```

## 🏋️ Training

0. gen policy_config for each layout
```shell
#! modify the policy version
bash store_overcooked_config.sh {layout}
#! modify the layout list
bash mv_policy_config.sh
```

**An Example of policy_config**

[policy_config example](./policy_config.example)

### Prepare the Evaluation Partners

1. train hsp models
```shell
bash train_overcooked_hsp.sh {layout}
```
2. extract hsp models
```shell
python extrace_hsp_S1_models.py {layout}
```
3. evaluate hsp models
```shell
bash eval_events.sh {layout}
```
4. select bias agents and generate evaluation ymls
```shell
python bias/select_bias_agent_br.py --layout {layout} --select dpp --k 10 --eval_result_dir eval/results --N 1000000
# or
python bias/select_bias_agent_br.py --layout [unident_s|unident_s_hard] --select dpp --k 15 --eval_result_dir eval/results --N 1000000
```
5. train BRs for mid-level biased agents
```shell
python gen_hsp_template.py {layout}
bash train_single_br.bash {layout}
```

### Train the ZSC Methods

#### FCP

##### Stage 1

1. train S1
```shell
bash train_overcooked_sp.sh {layout}
```
2. extract S1 models
```shell
#! modify the exp names first
python extract_sp_S1_models.py {layout}
```
##### Stage 2
1. generate S2 ymls
```shell
#! modify the exp names first
python gen_S2_yml.py {layout} fcp
```
2. train S2
```shell
#! modify the pop names first
bash train_overcooked_fcp_stage_2.sh {layout}
```
3. extract S2 models
```shell
#! modify the exp names first
python extract_S2_models.py {layout} fcp
```


#### MEP | TrageDi

##### Stage 1

1. generate population yml
```shell
python gen_pop_ymls.py {layout} [mep|traj]
```
2. train S1
```shell
bash train_overcooked_[mep|traj].sh {layout}
```
3. extract S1 models
```shell
#! modify the exp names first
python extract_[mep|traj]_S1_models.py {layout}
```

##### Stage 2

1. generate S2 ymls
```shell
#! modify the exp names first
python gen_S2_yml.py {layout} [mep|traj]
```
2. train S2
```shell
#! modify the pop names first
bash train_overcooked_[mep|traj]_stage_2.sh {layout}
```
3. extract S2 models
```shell
#! modify the exp names first
python extract_S2_models.py {layout} [mep|traj]
```

#### HSP
1. generate S2 ymls
```shell
python gen_hsp_S2_ymls.py -l ${layout} -k 6 -s 5 -S 12
python gen_hsp_S2_ymls.py -l ${layout} -k 12 -s 10 -S 24
python gen_hsp_S2_ymls.py -l ${layout} -k 18 -s 15 -S 36
```
2. train S2
```shell
bash train_overcooked_hsp_stage_2.sh {layout}
```
3. extract S2 models
```shell
#! modify the exp names first
python extract_S2_models.py {layout} [hsp]
```

#### COLE

1. generate COLE ymls

```shell
python gen_cole_ymls.py {layout}
```

2. train COLE
```shell
bash train_overcooked_cole.sh {layout}
```

3. extract S2 models
```shell
#! modify the exp names first
python extract_S2_models.py {layout} cole
```

#### E3T

```shell
bash train_overcooked_e3t.sh {layout}
```


## 📝 Evaluation

1. evaluate S2 models
```shell
#! modify the pop names
bash eval_with_bias_agents.sh {layout}
bash eval_sp_with_bias_agents.sh {layout}
bash eval_e3t_with_bias_agents.sh {layout}
```
2. compute final results
```shell
#! modify the exp names
python extract_results.py -a {algo} -l {layout}
```

## 🤖 Pre-trained Models

You can download pretrained models here:

```shell
cd zsc_eval
git lfs install
git clone https://huggingface.co/Leoxxxxh/ZSC-Eval-policy_pool policy_pool
```

## 👩🏻‍💻 Human Experiment

### Debug
```shell
export POLICY_POOL="zsc_eval/policy_pool"; python zsc_eval/human_exp/overcooked-flask/app.py
```

### Run

```shell
bash zsc_eval/human_exp/human_exp_up.sh
```

## 🛠️ Code Structure

