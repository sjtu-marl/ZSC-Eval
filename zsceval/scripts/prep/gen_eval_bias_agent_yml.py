import os
import sys

from loguru import logger

if __name__ == "__main__":
    layout = sys.argv[1]
    if layout == "all":
        layouts = [
            "random0",
            "random0_medium",
            "random1",
            "random3",
            "small_corridor",
            "unident_s",
            "random0_m",
            "random1_m",
            "random3_m",
            "academy_3_vs_1_with_keeper",
        ]
    else:
        layouts = [layout]

    for layout in layouts:
        logger.info(layout)
        num_agents = 2
        if layout in ["academy_3_vs_1_with_keeper"]:
            num_agents = 3

        yml_path = f"../policy_pool/{layout}/hsp/s1/eval_template.yml"
        os.makedirs(os.path.dirname(yml_path), exist_ok=True)
        yml = open(
            yml_path,
            "w",
            encoding="utf-8",
        )
        for a_i in range(num_agents):
            yml.write(
                f"""
agent{a_i}:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/hsp/s1/pop/agent{a_i}_actor.pt
    """
            )
        yml.close()
