from collections import defaultdict
from typing import List
import os

from loguru import logger

LAYOUT_LIST = [
    "simple",
    "random1",
    "random3",
    "unident_s",
]

NAME_TRANSLATION = {
    "cramped_room": "simple",
    "asymmetric_advantages": "unident_s",
    "coordination_ring": "random1",
    "counter_circuit": "random3",
    "forced_coordination" : "random0",
}

LAYOUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overcooked-flask/static/layouts")


def generate_balanced_permutation(existing_permutations: List[List], sequence: List):
    n = len(sequence)

    # Initialize position counts
    position_counts = [defaultdict(int) for _ in range(n)]

    # Count occurrences of each element in each position
    for perm in existing_permutations:
        for idx, elem in enumerate(perm):
            position_counts[idx][elem] += 1

    logger.debug(f"Position counts\n: {position_counts}")
    # Generate a new permutation
    new_permutation = [None] * n
    used_elements = set()

    for idx in range(n):
        # Find the element with the minimum count in the current position
        min_count = float("inf")
        chosen_element = None

        for elem in sequence:
            if elem not in used_elements and position_counts[idx][elem] < min_count:
                min_count = position_counts[idx][elem]
                chosen_element = elem

        new_permutation[idx] = chosen_element
        used_elements.add(chosen_element)
        position_counts[idx][chosen_element] += 1  # Update the count for the chosen element in this position

    return new_permutation

def load_dict_from_file(filepath):
    with open(filepath, "r") as f:
        return eval(f.read())

def read_layout_dict(layout_name):
    return load_dict_from_file(os.path.join(LAYOUTS_DIR, layout_name + ".layout"))