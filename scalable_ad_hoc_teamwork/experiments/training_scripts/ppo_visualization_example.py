from scalable_ad_hoc_teamwork.experiment_util import env_creator
import pickle
import time

checkpoint = f"../../../experiment_data/models/ppo/final_versions/ppo_lbf_test.agent"

if checkpoint:
    with open(checkpoint, "rb") as input_file:
        ppo_agent = pickle.load(input_file)


identifier = "lbf_env"

env = env_creator.get_env_instance(identifier)

obs, info = env.reset()
env.render()

terminations = {"player_0": False}

while not all(terminations.values()):
    action = {"player_0": ppo_agent.select_action(obs["player_0"])}
    obs, rewards, terminations, truncations, infos = env.step(action)
    print(obs)
    print(rewards)
    env.render()
    time.sleep(0.2)

