from lbforaging.foraging.pettingzoo_environment import parallel_env as lbf_env
from scalable_ad_hoc_teamwork.experiment_util.environment_segmentation_functions import *


def get_env_instance(identifier: str, **config):
    standard_lbf_config = {"players": 1, "max_player_level": 2, "field_size": (8, 8), "max_food": 2, "sight": 8,
                           "max_episode_steps": 150, "force_coop": False, "normalize_reward": True}

    if identifier == "lbf_env":
        standard_lbf_config.update(config)
        return lbf_env(**standard_lbf_config)
    else:
        raise ValueError("Unknown env identifier: {}".format(identifier))


def get_env_info(identifier, **kwargs):
    if identifier == "lbf_env":
        def seg_func(state, self_player, other_players):
            return lbf_segmentation(state, kwargs["max_food"], self_player, other_players)
        return {"obs_dim_agent": 3, "state_segmentation_func": seg_func}
    else:
        raise ValueError("Unknown env identifier: {}".format(identifier))
