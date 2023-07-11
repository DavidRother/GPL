from scalable_ad_hoc_teamwork.rl_core.util import make_deterministic, clean_dict_for_evalpy
from scalable_ad_hoc_teamwork.rl_core.training import train
from scalable_ad_hoc_teamwork.experiment_util import env_creator, agent_creator, meta_agent
from scalable_ad_hoc_teamwork.rl_core.evaluation import evaluate
from lbforaging.foraging.pettingzoo_environment import ObservationSpace
from scalable_ad_hoc_teamwork.algorithm.util.observation_preprocessor import ObservationPreprocessor
from scalable_ad_hoc_teamwork.algorithm.ppo_lstm.agent import Architecture
import numpy as np
import torch
import tqdm
import pickle
import evalpy

# seed = 54556456
seed = 15
make_deterministic(seed)

project_path = "../../../experiment_data/"
folder_path = "evalpy"
experiment_identifier = "lbf_test_split_full_state_H5"
experiment_name = f"moop_experiment_{experiment_identifier}_{seed}"
evalpy.set_project(project_path, folder_path)

checkpoint = f""

device = torch.device("cpu")

identifier = "lbf_env"
testing_identifier = "lbf_env"

n_agents = 4

with open(checkpoint, "rb") as input_file:
    moop_agent = pickle.load(input_file)
moop_agent.switch_active_interactions(["collect_apples", "collect_apples", "collect_apples"])
# moop_agent.switch_active_tasks(["collect_apples"])
moop_agent.switch_mode(moop_agent.INFERENCE)

tasks = ["collect_apples", "collect_apples", "collect_apples", "collect_apples"]
obs_spaces = [ObservationSpace.VECTOR_OBSERVATION, ObservationSpace.SYMBOLIC_OBSERVATION,
              ObservationSpace.SYMBOLIC_OBSERVATION, ObservationSpace.SYMBOLIC_OBSERVATION]
force_coop = False
env = env_creator.get_env_instance(testing_identifier, players=n_agents, tasks=tasks, obs_spaces=obs_spaces,
                                   force_coop=force_coop)

random_agent = agent_creator.get_agent_instance("random", action_space=env.action_spaces["player_0"])
no_op_agent = agent_creator.get_agent_instance("no_op")
h1_agent = agent_creator.get_agent_instance("lbf_heuristic", heuristic_id="H1")
h2_agent = agent_creator.get_agent_instance("lbf_heuristic", heuristic_id="H2")
h3_agent = agent_creator.get_agent_instance("lbf_heuristic", heuristic_id="H3")
h4_agent = agent_creator.get_agent_instance("lbf_heuristic", heuristic_id="H4")
h5_agent = agent_creator.get_agent_instance("lbf_heuristic", heuristic_id="H5")

super_agent = meta_agent.MetaAgent([h1_agent, h2_agent, h3_agent, h4_agent, h5_agent])

ep_rews = {}
eval_scores = {}
ep_length_stats = {}

with moop_agent.eval_mode():
    eval_stats = evaluate(env, [moop_agent, super_agent, super_agent, super_agent], 100)
last_eval_mean = np.mean(eval_stats["Last episode reward player_0"]).item()
average_episode_length = np.mean(eval_stats["Last Episode Length"]).item()
best_full_eval_mean = last_eval_mean

print(f"Last eval mean: {last_eval_mean}")
print(f"Average episode length: {average_episode_length}")
print(eval_stats)
