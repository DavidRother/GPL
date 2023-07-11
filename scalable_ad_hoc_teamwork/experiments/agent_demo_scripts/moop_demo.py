from scalable_ad_hoc_teamwork.rl_core.util import make_deterministic, clean_dict_for_evalpy
from scalable_ad_hoc_teamwork.rl_core.training import train
from scalable_ad_hoc_teamwork.experiment_util import env_creator, agent_creator
from scalable_ad_hoc_teamwork.rl_core.evaluation import evaluate
from lbforaging.foraging.pettingzoo_environment import ObservationSpace
from lbforaging.foraging.manual_policy import ManualPolicy
import numpy as np
import torch
import tqdm
import pickle
import evalpy
import time


# seed = 54556456
seed = 14
make_deterministic(seed)

project_path = "../../../experiment_data/"
folder_path = "evalpy"
experiment_identifier = "lbf_test_split_full_state_H5"
experiment_name = f"moop_experiment_{experiment_identifier}_{seed}"
evalpy.set_project(project_path, folder_path)

checkpoint = f"{project_path}models/moop/final_versions/moop_best_save_{experiment_identifier}.agent"

device = torch.device("cpu")

identifier = "lbf_env"
testing_identifier = "lbf_env"

n_agents = 2

tasks = ["collect_apples", "collect_apples"]
obs_spaces = [ObservationSpace.VECTOR_OBSERVATION, ObservationSpace.SYMBOLIC_OBSERVATION]
force_coop = True

env = env_creator.get_env_instance(identifier, players=n_agents, tasks=tasks, obs_spaces=obs_spaces,
                                   force_coop=force_coop)
eval_env = env_creator.get_env_instance(testing_identifier, players=n_agents, tasks=tasks, obs_spaces=obs_spaces,
                                        force_coop=force_coop)

with open(checkpoint, "rb") as input_file:
    moop_agent = pickle.load(input_file)
moop_agent.switch_active_interactions(["collect_apples"])
# moop_agent.switch_active_tasks(["collect_apples"])
moop_agent.switch_mode(moop_agent.INFERENCE)

random_agent = agent_creator.get_agent_instance("random", action_space=env.action_spaces["player_0"])
no_op_agent = agent_creator.get_agent_instance("no_op")
h5_agent = agent_creator.get_agent_instance("lbf_heuristic", heuristic_id="H5")

step_list = []

for i in range(10):

    obs, info = env.reset()
    env.render()

    terminations = {"player_0": False}

    manual_policy = ManualPolicy(env, agent_id="player_0")
    steps = 0

    while not all(terminations.values()):
        # manual_policy("player_0")
        # moop_agent.select_action(obs["player_0"])
        action = {"player_0": moop_agent.select_action(obs["player_0"]),
                  "player_1": h5_agent.select_action(obs["player_1"])}
        # print(action)
        obs, rewards, terminations, truncations, infos = env.step(action)
        # print(observations)
        # print(rewards)
        env.render()
        time.sleep(0.2)
        steps += 1
    step_list.append(steps)
    print(f"Episode finished after {steps} steps")
print(f"Average steps: {np.mean(step_list)}")
print(step_list)
