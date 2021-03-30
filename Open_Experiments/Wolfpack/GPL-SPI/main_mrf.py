import argparse
import gym
import random
import Wolfpack_gym
from Agent import MRFAgent
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from gym.vector import AsyncVectorEnv
from datetime import date
import random
import string
import os
import json
import copy
import math

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount_rate.')
parser.add_argument('--num_episodes', type=int, default=2000, help="Number of episodes for training.")
parser.add_argument('--update_frequency', type=int, default=4, help="Timesteps between updates.")
parser.add_argument('--saving_frequency', type=int,default=50,help="Number of episodes between checkpoints.")
parser.add_argument('--num_envs', type=int,default=16, help="Number of parallel environments for training.")
parser.add_argument('--tau', type=float,default=0.001, help="Tau for soft target update.")
parser.add_argument('--eval_eps', type=int, default=5, help="Number of episodes for evaluation.")
parser.add_argument('--weight_predict', type=float, default=1.0, help="Weight associated to action prediction loss.")
parser.add_argument('--save_dir', type=str, default='parameters', help="Directory name for saving parameters.")
parser.add_argument('--num_players', type=int, default=3, help="Initial number of players at the start of episode.")
parser.add_argument('--pair_comp', type=str, default='bmm', help="Pairwise factor computation method. Use bmm for low rank factorization.")
parser.add_argument('--info', type=str, default="", help="Additional info.")
parser.add_argument('--seed', type=int, default=0, help="Training seed.")
parser.add_argument('--close_penalty', type=float, default=0.1, help="Penalty for attacking alone.")
parser.add_argument('--constant_temp', type=bool, default=False, help="Constant temperature is used if set to True.")
parser.add_argument('--init_temp', type=float, default=1.0, help="Initial temperature for Boltzmann distribution.")
parser.add_argument('--final_temp', type=float, default=0.01, help="Final temperature for Boltzmann distribution.")
parser.add_argument('--temp_annealing', type=str, default="linear", help="Temperature annealing method.")

args = parser.parse_args()

if __name__ == '__main__':

    args = vars(args)
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")

    def randomString(stringLength=10):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(stringLength))

    random_experiment_name = randomString(10)
    writer = SummaryWriter(log_dir="runs/"+random_experiment_name)
    directory = os.path.join(args['save_dir'], random_experiment_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory,'params.json'), 'w') as json_file:
        json.dump(args, json_file)

    with open(os.path.join('runs',random_experiment_name, 'params.json'), 'w') as json_file:
        json.dump(args, json_file)

    # Initialize GPL-SPI Agent
    agent = MRFAgent(args=args, writer=writer, added_u_dim = 12, temp=args["init_temp"])

    # Define the training environment
    def make_env(env_id, rank, num_players, seed=1285, close_penalty=0.5, implicit_max_player_num=3, with_shuffling=False):
        def _init():
            env = gym.make(env_id, seed=seed + rank, num_players=num_players, close_penalty=close_penalty, implicit_max_player_num=implicit_max_player_num,  with_shuffling=with_shuffling)
            return env

        return _init


    num_players = args['num_players']
    env = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                  args['seed'], args['close_penalty']) for i in range(args['num_envs'])])

    # Save initial model parameters.
    save_dirs = os.path.join(directory, 'params_0')
    agent.save_parameters(save_dirs)

    # Evaluate initial model performance in training environment
    avgs = []
    for ep_val_num in range(args['eval_eps']):
        num_players = args['num_players']
        agent.reset()
        steps = 0
        avg_total_rewards = 0.0
        env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                  2000, args['close_penalty']) for i in range(args['num_envs'])])

        f_done = False
        obs = env_eval.reset()

        while not f_done:
            acts = agent.step(obs, eval=True)
            n_obs, rewards, dones, info = env_eval.step(acts)
            avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
            f_done = any(dones)
            obs = n_obs
        avgs.append(avg_total_rewards)
        print("Finished eval with rewards " + str(avg_total_rewards))
    env_eval.close()
    writer.add_scalar('Rewards/train_set', sum(avgs) / len(avgs),0)

    # Evaluate initial model performance in training environment when temperature approaches 0
    avgs = []
    for ep_val_num in range(args['eval_eps']):
        num_players = args['num_players']
        agent.reset()
        steps = 0
        avg_total_rewards = 0.0
        env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                            2000, args['close_penalty']) for i in range(args['num_envs'])])

        f_done = False
        obs = env_eval.reset()

        while not f_done:
            acts = agent.step(obs, eval=True, hard_eval=True)
            n_obs, rewards, dones, info = env_eval.step(acts)
            avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
            f_done = any(dones)
            obs = n_obs
        avgs.append(avg_total_rewards)
        print("Finished hard eval with rewards " + str(avg_total_rewards))
    env_eval.close()
    writer.add_scalar('Rewards/train_set_hard', sum(avgs) / len(avgs), 0)

    # Evaluate initial model performance in test environment
    avgs = []
    for ep_val_num in range(args['eval_eps']):
        num_players = args['num_players']
        agent.reset()
        steps = 0
        avg_total_rewards = 0.0
        env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                  2000, args['close_penalty'], implicit_max_player_num=5) for i in range(args['num_envs'])])

        f_done = False
        obs = env_eval.reset()

        while not f_done:
            acts = agent.step(obs, eval=True)
            n_obs, rewards, dones, info = env_eval.step(acts)
            avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
            f_done = any(dones)
            obs = n_obs
        avgs.append(avg_total_rewards)
        print("Finished eval with rewards " + str(avg_total_rewards))
    env_eval.close()
    writer.add_scalar('Rewards/eval', sum(avgs) / len(avgs),0)

    # Evaluate initial model performance in test environment when temperature approaches 0.
    avgs = []
    for ep_val_num in range(args['eval_eps']):
        num_players = args['num_players']
        agent.reset()
        steps = 0
        avg_total_rewards = 0.0
        env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                            2000, args['close_penalty'], implicit_max_player_num=5) for i in
                                   range(args['num_envs'])])

        f_done = False
        obs = env_eval.reset()

        while not f_done:
            acts = agent.step(obs, eval=True, hard_eval=True)
            n_obs, rewards, dones, info = env_eval.step(acts)
            avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
            f_done = any(dones)
            obs = n_obs
        avgs.append(avg_total_rewards)
        print("Finished hard eval with rewards " + str(avg_total_rewards))
    env_eval.close()
    writer.add_scalar('Rewards/eval_hard', sum(avgs) / len(avgs), 0)

    # Agent training loop
    for ep_num in range(args['num_episodes']):
        print(ep_num)

        train_avgs = 0
        steps = 0
        f_done = False

        # Reset agent hidden vectors at the beginning of each episode.
        agent.reset()
        if not args['constant_temp']:
            # Anneal temperature parameters at each episode.
            if not args['temp_annealing'] == "linear":
                start_log = math.log(args['init_temp'])
                end_log = math.log(args['final_temp'])
                agent.set_temp(math.exp(max(start_log - ((ep_num + 0.0) / 1500) * (start_log-end_log), end_log)))
            else:
                agent.set_temp(max(args['init_temp'] - ((ep_num + 0.0) / 1500) * (args['init_temp'] - args['final_temp']), args['final_temp']))
        obs = env.reset()
        agent.compute_target(None, None, None, None, obs, add_storage=False)

        while not f_done:
            acts = agent.step(obs)
            n_obs, rewards, dones, info = env.step(acts)
            f_done = any(dones)

            n_obs_replaced = n_obs
            if f_done:
                n_obs_replaced = copy.deepcopy(n_obs)
                for key in n_obs_replaced.keys():
                    for idx in range(len(n_obs_replaced[key])):
                        if dones[idx]:
                            n_obs_replaced[key][idx] = info[idx]['terminal_observation'][key]
            steps += 1

            train_avgs += (sum(rewards) + 0.0) / len(rewards)
            agent.compute_target(obs, acts, rewards, dones, n_obs_replaced, add_storage=True)

            if steps % args['update_frequency'] == 0 or f_done:
                agent.update()

            obs = n_obs

        writer.add_scalar('Rewards/train', train_avgs, ep_num)

        # Checkpoint and evaluate agents every few episodes.
        if (ep_num + 1) % args['saving_frequency'] == 0:
            save_dirs = os.path.join(directory, 'params_'+str((ep_num +
                                                               1) // args['saving_frequency']))
            agent.save_parameters(save_dirs)

            # Run evaluation in training environment.
            avgs = []
            for ep_val_num in range(args['eval_eps']):
                num_players = args['num_players']
                agent.reset()
                steps = 0
                avg_total_rewards = 0.0
                env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                  2000, args['close_penalty']) for i in range(args['num_envs'])])

                f_done = False
                obs = env_eval.reset()

                while not f_done:
                    acts = agent.step(obs, eval=True)
                    n_obs, rewards, dones, info = env_eval.step(acts)
                    avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
                    f_done = any(dones)
                    obs = n_obs
                avgs.append(avg_total_rewards)
                print("Finished eval with rewards " + str(avg_total_rewards))
            env_eval.close()
            writer.add_scalar('Rewards/train_set', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])

            # Run evaluation in training environment when temperature approaches 0.
            avgs = []
            for ep_val_num in range(args['eval_eps']):
                num_players = args['num_players']
                agent.reset()
                steps = 0
                avg_total_rewards = 0.0
                env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                                    2000, args['close_penalty']) for i in range(args['num_envs'])])

                f_done = False
                obs = env_eval.reset()

                while not f_done:
                    acts = agent.step(obs, eval=True, hard_eval=True)
                    n_obs, rewards, dones, info = env_eval.step(acts)
                    avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
                    f_done = any(dones)
                    obs = n_obs
                avgs.append(avg_total_rewards)
                print("Finished hard eval with rewards " + str(avg_total_rewards))
            env_eval.close()
            writer.add_scalar('Rewards/train_set_hard', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])

            # Run evaluation in testing environment.
            avgs = []
            for ep_val_num in range(args['eval_eps']):
                num_players = args['num_players']
                agent.reset()
                steps = 0
                avg_total_rewards = 0.0
                env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                  2000, args['close_penalty'], implicit_max_player_num=5) for i in range(args['num_envs'])])

                f_done = False
                obs = env_eval.reset()

                while not f_done:
                    acts = agent.step(obs, eval=True)
                    n_obs, rewards, dones, info = env_eval.step(acts)
                    avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
                    f_done = any(dones)
                    obs = n_obs
                avgs.append(avg_total_rewards)
                print("Finished eval with rewards " + str(avg_total_rewards))
            env_eval.close()
            writer.add_scalar('Rewards/eval', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])

            # Run evaluation in testing environment when temperature approaches 0.
            avgs = []
            for ep_val_num in range(args['eval_eps']):
                num_players = args['num_players']
                agent.reset()
                steps = 0
                avg_total_rewards = 0.0
                env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                                    2000, args['close_penalty'], implicit_max_player_num=5) for i in
                                           range(args['num_envs'])])

                f_done = False
                obs = env_eval.reset()

                while not f_done:
                    acts = agent.step(obs, eval=True, hard_eval=True)
                    n_obs, rewards, dones, info = env_eval.step(acts)
                    avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
                    f_done = any(dones)
                    obs = n_obs
                avgs.append(avg_total_rewards)
                print("Finished hard eval with rewards " + str(avg_total_rewards))
            env_eval.close()
            writer.add_scalar('Rewards/eval_hard', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])


