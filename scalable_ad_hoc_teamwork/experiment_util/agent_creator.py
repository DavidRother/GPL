from scalable_ad_hoc_teamwork.algorithm.ppo.ppo_agent import PPOAgent
from scalable_ad_hoc_teamwork.algorithm.ppo_lstm.agent import Agent
from scalable_ad_hoc_teamwork.algorithm.ppo_tom_architecture.agent import Agent as TomAgent
from scalable_ad_hoc_teamwork.experiment_util.expert_controllers.random_agent import RandomAgent
from scalable_ad_hoc_teamwork.experiment_util.expert_controllers.no_action_agent import NoActionAgent
from scalable_ad_hoc_teamwork.experiment_util.expert_controllers.lbf_heuristic_agent import LBFHeuristicAgent
from lbforaging.foraging.manual_policy import ManualPolicy as LBFManualPolicy


def get_agent_instance(identifier: str, **config):
    agent, config = get_agent_instance_with_config(identifier, **config)
    return agent


def get_agent_instance_with_config(identifier: str, **config):
    if identifier == "ppo":
        standard_config = {"memory_size": 400, "num_epochs": 10, "learning_rate": 1e-4, "num_batches": 5, "gamma": 0.99}
        standard_config.update(config)
        obs_dim = standard_config["obs_dim"]
        n_actions = standard_config["n_actions"]
        device = standard_config["device"]
        n_agents = standard_config["n_agents"]
        memory_size = standard_config["memory_size"]
        num_epochs = standard_config["num_epochs"]
        learning_rate = standard_config["learning_rate"]
        num_batches = standard_config["num_batches"]
        ppo_agent = PPOAgent(obs_dim, n_actions, n_agents, memory_size, num_epochs, learning_rate, num_batches, device)
        return ppo_agent, standard_config
    elif identifier == "moop":
        standard_config = {"memory_size": 400, "num_epochs": 10, "learning_rate": 1e-4, "num_batches": 5, "gamma": 0.99,
                           "vf_clip_param": 0.3, "clip_param": 0.3, "kl_coeff": 0.2,
                           "vf_loss_coeff": 0.25, "entropy_coeff": 0.01}
        standard_config.update(config)
        obs_dim_state = standard_config["obs_dim_state"]
        obs_dim_agent = standard_config["obs_dim_agent"]
        n_actions = standard_config["n_actions"]
        memory_size = standard_config["memory_size"]
        num_epochs = standard_config["num_epochs"]
        learning_rate = standard_config["learning_rate"]
        num_batches = standard_config["num_batches"]
        state_segmentation_func = standard_config["state_segmentation_func"]
        preprocessor = standard_config["preprocessor"]
        gamma = standard_config["gamma"]
        interaction_architecture = standard_config["interaction_architecture"]
        vf_clip_param = standard_config["vf_clip_param"]
        clip_param = standard_config["clip_param"]
        kl_coeff = standard_config["kl_coeff"]
        vf_loss_coeff = standard_config["vf_loss_coeff"]
        entropy_coeff = standard_config["entropy_coeff"]
        device = standard_config["device"]
        agent = Agent(obs_dim_state, obs_dim_agent, n_actions, memory_size, num_epochs, learning_rate,
                      num_batches, gamma, state_segmentation_func, preprocessor, interaction_architecture,
                      vf_clip_param, clip_param, kl_coeff, vf_loss_coeff, entropy_coeff, device)
        return agent, standard_config
    elif identifier == "tom_moop":
        standard_config = {"memory_size": 400, "num_epochs": 10, "learning_rate": 1e-4, "num_batches": 10,
                           "gamma": 0.99, "memory_size_action_prediction": 10000}
        standard_config.update(config)
        obs_dim_state = standard_config["obs_dim_state"]
        obs_dim_agent = standard_config["obs_dim_agent"]
        n_actions = standard_config["n_actions"]
        memory_size = standard_config["memory_size"]
        num_epochs = standard_config["num_epochs"]
        learning_rate = standard_config["learning_rate"]
        num_batches = standard_config["num_batches"]
        state_segmentation_func = standard_config["state_segmentation_func"]
        preprocessor = standard_config["preprocessor"]
        gamma = standard_config["gamma"]
        interaction_architecture = standard_config["interaction_architecture"]
        device = standard_config["device"]
        memory_size_action_prediction = standard_config["memory_size_action_prediction"]
        agent = TomAgent(obs_dim_state, obs_dim_agent, n_actions, memory_size, memory_size_action_prediction,
                         num_epochs, learning_rate, num_batches, gamma, state_segmentation_func, preprocessor,
                         interaction_architecture, device)
        return agent, standard_config
    elif identifier == "lbf_manual":
        env = config["env"]
        return LBFManualPolicy(env, agent_id="player_0"), None
    elif identifier == "random":
        action_space = config["action_space"]
        return RandomAgent(action_space), None
    elif identifier == "no_op":
        return NoActionAgent(), None
    elif identifier == "lbf_heuristic":
        heuristic_id = config["heuristic_id"]
        return LBFHeuristicAgent(heuristic_id), None
    else:
        raise ValueError("Unknown agent identifier: {}".format(identifier))
