import numpy as np


def lbf_segmentation(state, max_food, self_player, other_player):
    env_state = state[:, :max_food * 4]
    self_player_state = state[:, max_food * 4: max_food * 4 + 3]
    if other_player:
        self_idx = int(''.join(filter(str.isdigit, self_player)))
        other_idx = int(''.join(filter(str.isdigit, other_player)))
        if other_idx < self_idx:
            other_player_state = state[:, max_food * 4 + (other_idx + 1) * 3: max_food * 4 + (other_idx + 1) * 3 + 3]
        else:
            other_player_state = state[:, max_food * 4 + other_idx * 3: max_food * 4 + other_idx * 3 + 3]
    else:
        other_player_state = None
    return env_state, self_player_state, other_player_state
