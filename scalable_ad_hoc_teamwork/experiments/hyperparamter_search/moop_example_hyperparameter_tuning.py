from scalable_ad_hoc_teamwork.rl_core.util import make_deterministic, clean_dict_for_evalpy
from scalable_ad_hoc_teamwork.rl_core.training import train
from scalable_ad_hoc_teamwork.experiment_util import env_creator, agent_creator
from scalable_ad_hoc_teamwork.rl_core.evaluation import evaluate
from lbforaging.foraging.pettingzoo_environment import ObservationSpace
from scalable_ad_hoc_teamwork.algorithm.util.observation_preprocessor import ObservationPreprocessor
from scalable_ad_hoc_teamwork.algorithm.ppo_lstm.agent import Architecture
import numpy as np
import itertools
import torch
import tqdm
import pickle
import evalpy

# seed = 54556456
seed = 15
make_deterministic(seed)

project_path = "../../../experiment_data/"
folder_path = "evalpy"
experiment_identifier = "lbf_moop_hyperparameter_tuning"
experiment_name = f"moop_experiment_{experiment_identifier}_{seed}"
evalpy.set_project(project_path, folder_path)

checkpoint = f""

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

env_info = env_creator.get_env_info(identifier, max_food=2)

obs_dim_agent = env_info["obs_dim_agent"]
obs_dim_state = env.observation_spaces["player_0"].shape[0] - obs_dim_agent * n_agents
state_segmentation_func = env_info["state_segmentation_func"]
preprocessor = ObservationPreprocessor(env.observation_spaces["player_0"])
interaction_architecture = Architecture.SPLIT_FULL_STATE

learning_rate_list = [5e-5, 1e-4]
gamma_list = [0.95, 0.9]
vf_loss_coeff_list = [0.25, 0.5, 0.1]
entropy_coeff_list = [0.001, 0.01, 0.0]

param_combination = list(itertools.product(learning_rate_list, gamma_list, vf_loss_coeff_list, entropy_coeff_list))

print(f"Number of combinations: {len(param_combination)}")

for comb_idx, comb in enumerate(param_combination):

    learning_rate = comb[0]
    gamma = comb[1]
    num_batches = 10
    vf_clip_param = 0.3
    clip_param = 0.3
    kl_coeff = 0.2
    vf_loss_coeff = comb[2]
    entropy_coeff = comb[3]

    print(f"learning_rate: {learning_rate}, gamma: {gamma}, "
          f"vf_loss_coeff: {vf_loss_coeff}, entropy_coeff: {entropy_coeff}")
    obs_dim = env.observation_spaces["player_0"].shape[0]
    n_actions = env.action_spaces["player_0"].n
    moop_agent, config = agent_creator.get_agent_instance_with_config("moop", obs_dim_state=obs_dim_state,
                                                                      obs_dim_agent=obs_dim_agent, n_actions=n_actions,
                                                                      learning_rate=learning_rate, gamma=gamma,
                                                                      state_segmentation_func=state_segmentation_func,
                                                                      preprocessor=preprocessor,
                                                                      interaction_architecture=interaction_architecture,
                                                                      num_batches=num_batches,
                                                                      vf_clip_param=vf_clip_param,
                                                                      clip_param=clip_param,
                                                                      kl_coeff=kl_coeff,
                                                                      vf_loss_coeff=vf_loss_coeff,
                                                                      entropy_coeff=entropy_coeff,
                                                                      device=device)
    moop_agent.agent_id = "player_0"
    moop_agent.add_interaction("collect_apples")
    moop_agent.switch_active_interactions(["collect_apples"])
    moop_agent.switch_mode(moop_agent.LEARN_INTERACTION)

    random_agent = agent_creator.get_agent_instance("random", action_space=env.action_spaces["player_0"])
    no_op_agent = agent_creator.get_agent_instance("no_op")
    h1_agent = agent_creator.get_agent_instance("lbf_heuristic", heuristic_id="H1")
    h5_agent = agent_creator.get_agent_instance("lbf_heuristic", heuristic_id="H5")

    num_episodes = 3000

    ep_rews = {}
    eval_scores = {}
    ep_length_stats = {}

    with moop_agent.eval_mode():
        eval_stats = evaluate(eval_env, [moop_agent, h5_agent], 100)
    last_eval_mean = np.mean(eval_stats["Last episode reward player_0"]).item()
    average_episode_length = np.mean(eval_stats["Last Episode Length"]).item()
    best_full_eval_mean = last_eval_mean

    with evalpy.start_run(experiment_name):
        pbar = tqdm.tqdm(train(env, [moop_agent, h5_agent], num_episodes))
        old_episode_number = 0
        for stats in pbar:
            moop_agent.learn_step()
            stats.update({"Last Eval Mean": last_eval_mean, "Average Episode Length": average_episode_length,
                          "Best Eval Mean": best_full_eval_mean, "RunIdx": comb_idx})
            pbar.set_postfix(stats)
            if stats["Episode Number"] > old_episode_number:
                old_episode_number = stats["Episode Number"]
                if old_episode_number % 100 == 0:
                    with moop_agent.eval_mode():
                        eval_stats = evaluate(eval_env, [moop_agent, h5_agent], 100)
                    last_eval_mean = np.mean(eval_stats["Last episode reward player_0"]).item()
                    average_episode_length = np.mean(eval_stats["Last Episode Length"]).item()
                    if last_eval_mean > best_full_eval_mean:
                        moop_agent.save_agent(
                            f"{project_path}models/moop/final_versions/moop_best_save_{experiment_identifier}_{comb_idx}.agent")
                        # print(last_eval_mean)
                        best_full_eval_mean = last_eval_mean
                stats = clean_dict_for_evalpy(stats)
                evalpy.log_run_step(stats, step_forward=True)
            # if last_eval_mean > 19.3:
            #     break
        with moop_agent.eval_mode():
            eval_stats = evaluate(eval_env, [moop_agent, h5_agent], 100)
        evalpy.log_run_entries({"seed": seed,
                                "Last Eval Mean": np.mean(eval_stats["Last episode reward player_0"]).item(),
                                "Average Episode Length": np.mean(eval_stats["Last Episode Length"]).item(),
                                "idx": comb_idx})

    moop_agent.save_agent(f"{project_path}models/moop/final_versions/moop_{experiment_identifier}_{comb_idx}.agent")
