import argparse
import unittest

from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_test import (
    TestDirection,
    TestGridworld,
    TestGymEnvironment,
    TestOvercookedEnvironment,
)
from zsceval.envs.overcooked.overcooked_ai_py.planning.planners_test import (
    TestHighLevelPlanner,
    TestJointMotionPlanner,
    TestMediumLevelPlanner,
    TestMotionPlanner,
)

if __name__ == "__main__":
    unittest.main()
