import math
import time
from collections import defaultdict
from pprint import pformat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from zsceval.algorithms.r_mappo.r_mappo import R_MAPPO
from zsceval.algorithms.utils.util import check
from zsceval.utils.util import get_gard_norm, huber_loss, mse_loss
from zsceval.utils.valuenorm import ValueNorm


class R_MAPPO_Target(R_MAPPO):
    def __init__(
        self,
        args,
        policy: R_MAPPOPolicy,
        source: R_MAPPOPolicy,
        device=torch.device("cpu"),
    ):
        super().__init__(args, policy, device)
        self.source = source
        self.weights_copy_factor = args.weights_copy_factor

    def _copy_params(self, source, target, alpha: float):
        for name, children in source.named_children():
            source_model = children.module if isinstance(children, nn.DataParallel) else children
            target_model = (
                getattr(target, name).module if isinstance(children, nn.DataParallel) else getattr(target, name)
            )
            for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
                target_param.data.copy_((1 - alpha) * target_param.data + alpha * source_param.data)

    def update_from_source(self):
        self._copy_params(self.source.actor, self.policy.actor, self.weights_copy_factor)
        self._copy_params(self.source.critic, self.policy.critic, self.weights_copy_factor)
