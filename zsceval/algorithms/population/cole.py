import copy
import os
from collections import OrderedDict

import torch

from zsceval.algorithms.population.policy_pool import PolicyPool
from zsceval.algorithms.population.trainer_pool import TrainerPool

POLICY_POOL_PATH = os.environ.get("POLICY_POOL")
ACTOR_POOL_PATH = os.environ.get("EVOLVE_ACTOR_POOL")


class COLE_Trainer(TrainerPool):
    def __init__(self, args, policy_pool: PolicyPool, device=torch.device("cpu")):
        super().__init__(args, policy_pool, device)

        self.seed = args.seed
        self.id = args.id
        self.stage = args.stage
        self.args = args
        self.population_play_ratio = self.args.population_play_ratio

    def init_population(self):
        super().init_population()

        self.agent_name = self.all_args.adaptive_agent_name
        self.population = {
            trainer_name: self.trainer_pool[trainer_name]
            for trainer_name in self.trainer_pool.keys()
            if not trainer_name.startswith(self.agent_name)
        }
        self.population_size = self.all_args.population_size

        self.generated_population_names = []
        # init
        if hasattr(self.args, "layout_name"):
            self.model_path_dir = os.path.join("cole", self.args.layout_name, self.args.experiment_name, f"{self.id}")
        else:
            self.model_path_dir = os.path.join("cole", self.args.scenario_name, self.args.experiment_name, f"{self.id}")
        os.makedirs(os.path.join(ACTOR_POOL_PATH, self.model_path_dir), exist_ok=True)
        # os.makedirs(os.path.join(POLICY_POOL_PATH, self.model_path_dir), exist_ok=True)
        for trainer_i, trainer_name in enumerate(
            list(self.population.keys())[
                : max(
                    self.population_play_ratio,
                    self.population_size // self.args.init_agent_ratio,
                )
            ]
        ):
            model_path = self.save_actor(trainer_name, trainer_i + 1)
            self.policy_pool.update_policy(trainer_name, False, model_path={"actor": model_path})
            self.generated_population_names.append(trainer_name)

        assert len(self.population) % self.population_size == 0

    def reward_shaping_steps(self):
        reward_shaping_steps = super().reward_shaping_steps()
        return [x // (self.population_play_ratio + 2) * (self.population_play_ratio + 1) for x in reward_shaping_steps]

    def save_steps(self):
        steps = super().save_steps()
        # logger.info(f"steps {steps}")
        steps = {
            trainer_name: v // (self.population_play_ratio + 2) * (self.population_play_ratio + 1)
            for trainer_name, v in steps.items()
        }
        return steps

    def reset(
        self,
        map_ea2t,
        n_rollout_threads,
        num_agents,
        # n_repeats,
        load_unused_to_cpu=False,
    ):
        super().reset(map_ea2t, n_rollout_threads, num_agents, load_unused_to_cpu)

    def train(self, sp_size):
        # PPO training
        super().train()
        return copy.deepcopy(self.train_infos)

    def save_actor(self, trainer_name: str, trainer_i: int) -> str:
        trainer = self.trainer_pool[trainer_name]
        policy_actor = trainer.policy.actor
        model_path = os.path.join(self.model_path_dir, f"{trainer_i}.pt")
        if getattr(self.args, "data_parallel", False):

            def get_new_state_dict(state_dict: OrderedDict):
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace("module.", "") if "module." in k else k
                    new_state_dict[name] = v
                return new_state_dict

            torch.save(
                get_new_state_dict(policy_actor.state_dict()),
                os.path.join(ACTOR_POOL_PATH, model_path),
            )
        else:
            torch.save(
                policy_actor.state_dict(),
                os.path.join(ACTOR_POOL_PATH, model_path),
            )
        return model_path
