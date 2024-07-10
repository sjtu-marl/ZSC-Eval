import os
import pickle

PLANNERS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_saved_action_manager(filename):
    with open(os.path.join(PLANNERS_DIR, filename), "rb") as f:
        mlp_action_manager = pickle.load(f)
        return mlp_action_manager
