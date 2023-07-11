from scalable_ad_hoc_teamwork.rl_core.util import make_deterministic, clean_dict_for_evalpy
from scalable_ad_hoc_teamwork.rl_core.training import train
from scalable_ad_hoc_teamwork.experiment_util import env_creator, agent_creator
from scalable_ad_hoc_teamwork.rl_core.evaluation import evaluate
from lbforaging.foraging.pettingzoo_environment import ObservationSpace
import numpy as np
import torch
import tqdm
import pickle
import evalpy


def training_start_up(seed, training_agent, partner_agent, env, eval_env, experiment_identifier, num_episodes):
    make_deterministic(seed)
    
    project_path = "../../../experiment_data/"
    folder_path = "evalpy"
    experiment_name = f"moop_experiment_{experiment_identifier}_{seed}"
    evalpy.set_project(project_path, folder_path)
    
    with training_agent.eval_mode():
        eval_stats = evaluate(eval_env, [training_agent, partner_agent], 100)
    last_eval_mean = np.mean(eval_stats["Last episode reward player_0"]).item()
    best_full_eval_mean = last_eval_mean
    
    print(f"Starting with mean reward of {last_eval_mean}")
    with evalpy.start_run(experiment_name):
        pbar = tqdm.tqdm(train(env, [training_agent, partner_agent], num_episodes))
        old_episode_number = 0
        for stats in pbar:
            training_agent.learn_step()
            stats.update({"Last Eval Mean": last_eval_mean})
            pbar.set_postfix(stats)
            if stats["Episode Number"] > old_episode_number:
                old_episode_number = stats["Episode Number"]
                if old_episode_number % 5000 == 0:
                    file_path = f"{project_path}models/moop/temp/moop_{experiment_identifier}_ep{old_episode_number}_seed{seed}.agent"
                    training_agent.save_agent(file_path)
                if old_episode_number % 100 == 0:
                    with training_agent.eval_mode():
                        eval_stats = evaluate(eval_env, [training_agent, partner_agent], 100)
                    last_eval_mean = np.mean(eval_stats["Last episode reward player_0"]).item()
                    if last_eval_mean > best_full_eval_mean:
                        training_agent.save_agent(
                            f"{project_path}models/moop/final_versions/moop_best_save_{experiment_identifier}.agent")
                        print(last_eval_mean)
                        best_full_eval_mean = last_eval_mean
                stats = clean_dict_for_evalpy(stats)
                evalpy.log_run_step(stats, step_forward=True)
        with training_agent.eval_mode():
            eval_stats = evaluate(eval_env, [training_agent, partner_agent], 100)
        evalpy.log_run_entries({"seed": seed,
                                "Last Eval Mean": np.mean(eval_stats["Last episode reward player_0"]).item()})
    
    training_agent.save_agent(f"{project_path}models/moop/final_versions/moop_{experiment_identifier}.agent")
