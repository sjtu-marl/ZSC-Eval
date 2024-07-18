`human_exp` contains:

`configs/`:
- `benchmark_configs`: agents for human study
- `before_game.yaml`: basic information collection
- `statement.md`: experiment statements

`data/`:
- `questionnaires/`: human subjective rankings
- `trajs/`: human game-play trajectories collections
- `progress.json` (Generated during experiments): human study progress record for experiments resuming

`overcooked-flask/`:
- `static/`: overcooked game
- `app.py`: code for serving website backend

`agent_pool.py`: interface or trained agents

`human_exp_up.sh`: service launch script
