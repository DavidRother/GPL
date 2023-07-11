from scalable_ad_hoc_teamwork.rl_core.util import make_deterministic, clean_dict_for_evalpy
from scalable_ad_hoc_teamwork.rl_core.training import train
from scalable_ad_hoc_teamwork.experiment_util import env_creator, agent_creator
from scalable_ad_hoc_teamwork.rl_core.evaluation import evaluate
import numpy as np
import torch
import tqdm
import pickle
import evalpy

seed = 54556456
make_deterministic(seed)

project_path = "../../../experiment_data/"
folder_path = "evalpy"
experiment_identifier = "lbf_test"
experiment_name = f"moop_experiment_{experiment_identifier}_{seed}"
evalpy.set_project(project_path, folder_path)

checkpoint = f""

device = torch.device("cpu")

identifier = "lbf_env"
testing_identifier = "lbf_env"

n_agents = 1

env = env_creator.get_env_instance(identifier, players=n_agents)
eval_env = env_creator.get_env_instance(testing_identifier, players=n_agents)

env_info = env_creator.get_env_info(identifier, max_food=2)

obs_dim_agent = env_info["obs_dim_agent"]
obs_dim_state = env.observation_spaces["player_0"].shape[0] - obs_dim_agent * n_agents
state_segmentation_func = env_info["state_segmentation_func"]

obs_dim = env.observation_spaces["player_0"].shape[0]
n_actions = env.action_spaces["player_0"].n

if checkpoint:
    with open(checkpoint, "rb") as input_file:
        moop_agent = pickle.load(input_file)
else:
    moop_agent = agent_creator.get_agent_instance("moop", obs_dim_state=obs_dim_state, obs_dim_agent=obs_dim_agent,
                                                  n_actions=n_actions, state_segmentation_func=state_segmentation_func,
                                                  device=device)
    moop_agent.agent_id = "player_0"
    moop_agent.add_task("collect_apples")
moop_agent.switch_active_tasks(["collect_apples"])
moop_agent.switch_mode(moop_agent.LEARN_TASK)

random_agent = agent_creator.get_agent_instance("random", action_space=env.action_spaces["player_0"])
no_op_agent = agent_creator.get_agent_instance("no_op")

num_episodes = 50000

ep_rews = {}
eval_scores = {}
ep_length_stats = {}

eval_stats = evaluate(eval_env, [moop_agent], 100)
last_eval_mean = np.mean(eval_stats["Last episode reward player_0"]).item()
average_episode_length = np.mean(eval_stats["Last Episode Length"]).item()
best_full_eval_mean = last_eval_mean

print(f"Starting with mean reward of {last_eval_mean}")
print(f"Starting with mean episode length of {average_episode_length}")
with evalpy.start_run(experiment_name):
    pbar = tqdm.tqdm(train(env, [moop_agent], num_episodes))
    old_episode_number = 0
    for stats in pbar:
        moop_agent.learn_step()
        stats.update({"Last Eval Mean": last_eval_mean, "Average Episode Length": average_episode_length})
        pbar.set_postfix(stats)
        if stats["Episode Number"] > old_episode_number:
            old_episode_number = stats["Episode Number"]
            if old_episode_number % 100 == 0:
                file_path = f"{project_path}models/moop/temp/moop_{experiment_identifier}_ep{old_episode_number}_seed{seed}.agent"
                moop_agent.save_agent(file_path)
            if old_episode_number % 100 == 0:
                eval_stats = evaluate(eval_env, [moop_agent], 100)
                last_eval_mean = np.mean(eval_stats["Last episode reward player_0"]).item()
                average_episode_length = np.mean(eval_stats["Last Episode Length"]).item()
                if last_eval_mean > best_full_eval_mean:
                    moop_agent.save_agent(
                        f"{project_path}models/moop/final_versions/moop_best_save_{experiment_identifier}.agent")
                    print(last_eval_mean)
                    best_full_eval_mean = last_eval_mean
            stats = clean_dict_for_evalpy(stats)
            evalpy.log_run_step(stats, step_forward=True)
        # if last_eval_mean > 19.3:
        #     break

    eval_stats = evaluate(eval_env, [moop_agent], 100)
    evalpy.log_run_entries({"seed": seed,
                            "Last Eval Mean": np.mean(eval_stats["Last episode reward player_0"]).item(),
                            "Average Episode Length": np.mean(eval_stats["Last Episode Length"]).item()})

moop_agent.save_agent(f"{project_path}models/moop/final_versions/moop_{experiment_identifier}.agent")
