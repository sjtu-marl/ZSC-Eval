# ZSC-Eval: An Evaluation Toolkit and Benchmark for Multi-agent Zero-shot Coordination

## Overview
<div align=center>
<img src="assets/ZSC-Eval.png" width="800px">
</div>

This repository is the official implementation of [ZSC-Eval: An Evaluation Toolkit and Benchmark for Multi-agent Zero-shot Coordination](https://arxiv.org/abs/2310.05208v2). 

ZSC-Eval is a comprehensive and convenient evaluation toolkit and benchmark for zero-shot coordination (ZSC) algorithms, including partner candidates generation via behavior-preferring rewards, partners selection via Best-Response Diversity (BR-Div), and ZSC capability measurement via Best-Response Proximity (BR-Prox).

<div align=center>
<img src="assets/table_comparison.png" width="600px">
</div>


This repo includes:
- Evaluation Framework
    - Generation and Selection of Behavior-preferring Evaluation Partners
    - Measurement of ZSC capability via Best-Response Proximity and other metrics
- Environments Support
    - Overcooked-ai 🧑‍🍳
    - Overcooked-ai with Multiple Recipes 🧑‍🍳 (New Coordination Challenge!)
    - Google Research Football ⚽️
- ZSC Algorithms Implementation
    - [FCP: Fictitious Co-Play](https://arxiv.org/abs/2110.08176)
    - [MEP: Maximum Entropy Population-based training](https://arxiv.org/abs/2112.11701)
    - [TrajeDi: Trajectory Diversity PBT](https://proceedings.mlr.press/v139/lupu21a.html)
    - [HSP: Hidden-utility Self-Play](https://arxiv.org/abs/2302.01605)
    - [COLE: Cooperative Open-ended Learning](https://arxiv.org/abs/2302.04831)
    - [E3T: Efficient End-to-End Training](https://papers.nips.cc/paper_files/paper/2023/hash/07a363fd2263091c2063998e0034999c-Abstract-Conference.html)
    - [SP: Self-play](https://github.com/marlbenchmark/on-policy)
- A Human Study Platform
    - Real-time Overcooked game play
    - Subjective Ranking
    - Trajectories Collection
- Benchmarks
    - Benchmark of ZSC Algorithms under ZSC-Eval
    - Benchmark of ZSC Algorithms under Human Evaluation

## 🗺️ Supported Environments

### 🧑‍🍳 Overcooked 

[Overcooked](https://github.com/HumanCompatibleAI/overcooked_ai) is a simulation environment for reinforcement learning derived from the Overcooked! video game and popular for coordination problems.

The Overcooked environment features a two-player collaborative game structure with shared rewards, where each player assumes the role of a chef in a kitchen, working together to prepare and serve soup for a team reward. 

<div align=center>
<img src="assets/overcooked.png" width="500px">
</div>

We further include Overcooked games with multiple recipes, in which agents should decide the schedule of cooking different recipe for higher rewards.

<div align=center>
<img src="assets/overcooked_new.png" width="500px">
</div>


### ⚽️ Google Research Football

[Google Research Football (GRF)](https://github.com/google-research/football) is a simulation environment for reinforcement learning based on the popular football video game. 
We choose the Football *Academy 3 vs. 1 with Keeper* scenario and implement it as a ZSC challenge.

<div align=center>
<img src="assets/grf.png" width="500px">
</div>


## 📖 Installation

To install requirements:

**ZSC-Eval and Overcooked**
```shell
conda env create -f environment.yml
```

**Google Research Football**
```shell
./install_grf.sh
```

## 📝 How to use ZSC-Eval for Evaluating ZSC Algorithms

After installation, here is the steps to use ZSC-Eval for evaluating the ZSC algorithms. We use the Overcooked Environment as an example.

```shell
cd zsceval/scripts/overcooked
```

### Setup the Policy Config

gen policy_config for each layout
```shell
bash shell/store_config.sh {layout}
#! modify the layout names
bash shell/mv_policy_config.sh
```

**An Example of policy_config**

[Policy Config Example](assets/policy_config.example)

### Prepare the Evaluation Partners

1. train behavior-preferring agents
```shell
bash shell/train_bias_agents.sh {layout}
```
2. extract agent models
```shell
cd ..
python extract_models/extract_bias_agents_models.py {layout}
python prep/gen_eval_bias_agent_yml.py {layout}
cd overcooked
```
3. evaluate the agents and get policy behaviors
```shell
bash shell/eval_bias_agents_events.sh {layout}
```
4. select evaluation partners and generate evaluation ymls
```shell
cd ..
python prep/select_bias_agent_br.py --env overcooked --layout {layout} --k 10 --N 1000000
```

Copy the results in `zsceval/scripts/prep/results/{layout}` to `zsceval/utils/bias_agent_vars.py`.

Generate benchmark yamls:

```shell
python prep/gen_bias_agent_yml.py -l {layout}
```

5. train BRs for mid-level biased agents
```shell
cd overcooked
bash shell/train_bias_agents_br.bash {layout}
```

### Evaluate the ZSC Agents

We using the most common baseline, FCP, as an example.

1. evaluate S2 models
```shell
#! modify the exp names
bash shell/eval_with_bias_agents.sh {layout} fcp
```
2. compute final results
```shell
#! modify the exp names
cd ..
python eval/extract_results.py -a {algo} -l {layout}
```


## 🏋️ Train ZSC Algorithms

We re-implement FCP, MEP, TrajeDi, HSP, COLE and E3T as the baselines in ZSC-Eval.
To train these ZSC methods, please follow the guide below:

First, replace `"your_wandb_name"` with your wandb username for convenience experiments management.

### Train FCP

**Stage 1**

1. train self-play agents
```shell
cd overcooked
bash shell/train_sp.sh {layout}
```
2. extract models
```shell
cd ..
#! modify the exp names
python extract_models/extract_sp_models.py {layout} overcooked
```
**Stage 2**
1. generate S2 ymls
```shell
#! modify the exp names
python prep/gen_S2_yml.py {layout} fcp
```
2. train S2
```shell
cd overcooked
#! modify the exp names
bash shell/train_fcp_stage_2.sh {layout} {population_size}
```
3. extract S2 models
```shell
cd ..
#! modify the exp names
python extract_models/extract_S2_models.py {layout} overcooked
```

### Train MEP | TrajeDi

**Stage 1**

1. generate Stage 1 population yml
```shell
python prep/gen_pop_ymls.py {layout} [mep|traj] -s {population_size}
```
2. train S1
```shell
cd overcooked
bash train_[mep|traj]_stage_1.sh {layout} {population_size}
```
3. extract S1 models
```shell
cd ..
#! modify the exp names
python extrace_models/extract_pop_S1_models.py {layout} overcooked
```

**Stage 2**

1. generate S2 yamls
```shell
#! modify the exp names
python prep/gen_S2_yml.py {layout} [mep|traj]
```
2. train S2
```shell
cd overcooked
#! modify the pop names
bash shell/train_[mep|traj]_stage_2.sh {layout} {population_size}
```
3. extract S2 models
```shell
cd ..
#! modify the exp names
python extract_models/extract_S2_models.py {layout} overcooked
```

### Train HSP
1. generate S2 ymls
```shell
python prep/gen_hsp_S2_ymls.py -l ${layout} -k {num_bias_agents} -s {mep_stage_1_population_size} -S {population_size}
```
2. train S2
```shell
cd overcooked
bash shell/train_hsp_stage_2.sh {layout} {population_size}
```
3. extract S2 models
```shell
#! modify the exp names
python extract_models/extract_S2_models.py {layout} overcooked
```

### Train COLE

1. generate COLE ymls

```shell
python prep/gen_cole_ymls.py {layout} -s {population_size}
```

2. train COLE
```shell
cd overcooked
bash shell/train_cole.sh {layout} {population_size}
```

3. extract S2 models
```shell
cd ..
#! modify the exp names
python extract_models/extract_S2_models.py {layout} overcooked
```

### Train E3T

```shell
cd overcooked
bash shell/train_e3t.sh {layout}
```


## 🤖 Pre-trained Models

We also provide the pre-train models for these baselines, you can download pretrained models from [huggingface](https://huggingface.co/Leoxxxxh/ZSC-Eval-policy_pool):

```shell
cd zsceval
git clone https://huggingface.co/Leoxxxxh/ZSC-Eval-policy_pool policy_pool
```

## 👩🏻‍💻 Human Study

We implement a human study platform, including game-playing, subjective ranking, and data collection. Details can be found in [zsceval/human_exp/README.md](zsceval/human_exp/README.md).

### Web UIs
#### Game-playing
<div align=center>
<img src="assets/game_play.png" width="600px">
</div>

#### Ranking
<div align=center>
<img src="assets/ranking.png" width="600px">
</div>


### Deployment

#### Debug Mode
```shell
export POLICY_POOL="zsc_eval/policy_pool"; python zsc_eval/human_exp/overcooked-flask/app.py
```
#### Production Mode
```shell
bash zsc_eval/human_exp/human_exp_up.sh
```

## 🛠️ Code Structure Overview

`zsceval` contains:

`algorithms/`:
- `population/`: trainers for population-based ZSC algorithms
- `r_mappo/`: trainers for self-play based algorithms, including SP and E3T

`envs/`:
- `overcooked/`: overcooked game with single recipe
- `overcooked_new/`: overcooked game with mutiple recipe
- `grf/`: google research football game

`runner/`: experiment runers for each environment

`utils/`:
- `config.py`: basic configuration
- `overcooked_config.py`: configuration for overcooked experimenets 
- `grf_config.py`: configuration for grf experimenets 

`policy_pool/`: training, evaluation yamls and agent models

`human_exp/`: human study platform

`scripts/`
- `prep/`: generate yamls for training
    - `select_bias_agent_br.py`: select evaluation partners
- `extract_models/`: code for extracting trained agent models 
- `render/`: environment rendering
- `overcooked/`: scripts for training and evaluating overcooked agents
    - `eval/`: python scripts for evaluation and extraction evaluation results
        - `results`: benchmark results
    - `shell/`: shell scripts for training and evaluating agents
    - `train/`: python training scripts for each algorithm
- `grf/`: scripts for training and evaluating grf agents
    - `eval/`: python scripts for evaluation and extraction evaluation results
        - `results`: benchmark results
    - `shell/`: shell scripts for training and evaluating agents
    - `train/`: python training scripts for each algorithm


## Benchmark Results

### Overcooked
Overall ZSC-Eval benchmark results in Overcooked.

<div align=center>
<img src="assets/overcooked_results.png" width="600px">
</div>

Human benchmark results in Overcooked.

<div align=center>
<img src="assets/human_overcooked_results.png" width="600px">
</div>


### GRF
Overall ZSC-Eval benchmark results in GRF.

<div align=center>
<img src="assets/grf_results.png" width="300px">
</div>


## Acknowledgements

We implement algorithms heavily based on https://github.com/samjia2000/HSP , and human study platform based on https://github.com/liyang619/COLE-Platform. 