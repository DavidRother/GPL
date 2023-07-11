from scalable_ad_hoc_teamwork.algorithm.ppo.buffer import MultiAgentBuffer, REWARDS
from scalable_ad_hoc_teamwork.algorithm.ppo.network import PPONetwork
from scalable_ad_hoc_teamwork.algorithm.ppo.postprocessing import postprocess
from scalable_ad_hoc_teamwork.algorithm.ppo.postporcess_interaction_reward import postprocess_interaction_reward
from scalable_ad_hoc_teamwork.algorithm.ppo.ppo_loss import ppo_surrogate_loss
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from functools import reduce

import pickle


class PPOAgent:

    LEARN_INTERACTION = "learn_interaction"
    LEARN_TASK = "learn_task"
    INFERENCE = "inference"

    def __init__(self, obs_dim, n_actions, n_agents, memory_size, num_epochs, learning_rate, num_batches, device):
        self.task_models = {}
        self.interaction_models = {}
        self.n_agents = n_agents
        self.memory_size = memory_size
        self.memory = MultiAgentBuffer(memory_size, self.n_agents)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.current_active_tasks = []
        self.learning_rate = learning_rate
        self.last_logits = None
        self.step = 0
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.mode = self.LEARN_TASK
        self.current_active_interaction_tasks = []

    def load_task_model(self, task, path):
        pass

    def save_agent(self, base_path, clear_memory=True):
        if clear_memory:
            self.memory.reset()
        Path(base_path).parent.mkdir(parents=True, exist_ok=True)
        with open(base_path, "wb") as output_file:
            pickle.dump(self, output_file)

    def get_active_task_model(self):
        return self.task_models[self.get_active_task()]

    def get_active_interaction_model(self):
        return self.interaction_models[self.get_active_interaction()]

    def get_active_interaction(self):
        return self.current_active_interaction_tasks[0]

    def get_active_task(self):
        return self.current_active_tasks[0]

    def add_task(self, task):
        new_model = PPONetwork(self.obs_dim, self.n_actions, self.learning_rate, self.device)
        self.task_models[task] = new_model

    def add_interaction(self, interaction):
        new_model = PPONetwork(self.obs_dim, self.n_actions, self.learning_rate, self.device)
        self.interaction_models[interaction] = new_model

    def switch_active_tasks(self, new_active_tasks):
        self.current_active_tasks = new_active_tasks
        self.memory.reset()

    def select_action(self, state, other_states=None, *args, **kwargs):
        if self.current_active_tasks and not self.current_active_interaction_tasks:
            active_model = self.get_active_task_model()
            action, logits = active_model.select_action(state)
            self.last_logits = logits
            return action
        elif self.current_active_interaction_tasks and not self.current_active_tasks:
            assert other_states is not None
            active_model = self.get_active_interaction_model()
            action, logits = active_model.select_action(other_states[0])
            self.last_logits = logits
            return action
        else:
            assert other_states is not None
            active_model = self.get_active_task_model()
            active_interaction_models = [self.interaction_models[inter] for inter in
                                         self.current_active_interaction_tasks]
            combined_dist = self.combine_policies(active_model, active_interaction_models, state, other_states)
            return combined_dist.sample().item()

    def switch_mode(self, new_mode):
        assert new_mode in [self.LEARN_TASK, self.LEARN_INTERACTION, self.INFERENCE]
        self.mode = new_mode
        self.memory.reset()

    def switch_active_interactions(self, new_active_interactions):
        self.current_active_interaction_tasks = new_active_interactions
        self.memory.reset()

    def store_transition(self, transition, **kwargs):
        if self.mode is self.INFERENCE:
            return
        if len(transition) == 5:
            state = {"player_0": transition[0]}
            next_state = {"player_0": transition[3]}
        elif len(transition) == 7:
            state = {"player_0": transition[5]}
            next_state = {"player_0": transition[6]}
        else:
            state = {"player_0": transition[0]}
            next_state = {"player_0": transition[3]}
        action = {"player_0": transition[1]}
        reward = {"player_0": transition[2]}
        done = {"player_0": transition[4]}
        info = {"player_0": None}
        logits = {"player_0": self.last_logits}
        self.memory.commit(state, action, next_state, reward, done, info, logits, self.step)
        self.last_logits = None
        if done:
            self.step = 0
        else:
            self.step += 1

    def learn_step(self, other_agent_model=None):
        if self.memory.full():
            if self.mode is self.LEARN_TASK:
                active_model = self.get_active_task_model()
            elif self.mode is self.LEARN_INTERACTION:
                active_model = self.get_active_interaction_model()
            else:
                return
            agents = {f"player_0": active_model}
            agent_1_config = {"prosocial_level": 0.0, "update_prosocial_level": False, "use_prosocial_head": False}
            agent_configs = {"player_0": agent_1_config}
            if self.mode is self.LEARN_INTERACTION:
                assert other_agent_model is not None, "When computing the interaction reward we need the model of the "\
                                                      "other agent"
                postprocess_interaction_reward("player_0", self.memory, other_agent_model, self.device)
            postprocess(agents, self.memory, agent_configs)

            for epoch in range(self.num_epochs):
                for player in agents:
                    for batch in self.memory.buffer[player].build_batches(self.num_batches):
                        loss, stats = ppo_surrogate_loss(agents[player], batch, agent_configs[player])

                        active_model.optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), config.max_grad_norm)
                        active_model.optimizer.step()

                # print(player, stats)
            self.memory.reset()

    def info_callback(self, agent_id, info):
        pass

    @staticmethod
    def calc_entropy(dist, base=2):
        return -np.sum(dist * np.log(dist, out=np.zeros_like(dist), where=(dist != 0)) / np.log(base), axis=1)

    def calc_extropy(self, dist):
        dist = 1 - dist
        return self.calc_entropy(dist)

    @staticmethod
    def calc_relative_entropy(dist1, dist2, base=2):
        frac = np.divide(dist1, dist2, where=(dist2 != 0))
        return np.sum(dist1 * np.log(frac, out=np.zeros_like(frac), where=(frac != 0)) / np.log(base), axis=1)

    def calc_jensen_shannon_divergence(self, dist1, dist2, base=2):
        m = 0.5 * (dist1 + dist2)
        return 0.5 * (self.calc_relative_entropy(dist1, m, base) + self.calc_relative_entropy(dist2, m, base))

    def combine_policies(self, task_policy, interaction_policies, state, other_states):
        dist_task = task_policy.get_dist(state)
        dist_interactions = [interaction_policy.get_dist(other_state) for other_state, interaction_policy
                             in zip(other_states, interaction_policies)]
        dists = np.stack([dist_task.probs.cpu().numpy().flatten()] +
                         [d.probs.cpu().numpy().flatten() for d in dist_interactions])
        support = dists.shape[1]
        complement_entropies = 1 - self.calc_entropy(dists, support)
        jsd = self.calc_jensen_shannon_divergence(dists, dists[0], support)
        complement_relative_entropies = 1 - jsd
        ent_weights = complement_entropies * complement_relative_entropies
        weights = ent_weights / sum(ent_weights)
        print(f"Entropy + distance weights: {complement_entropies * complement_relative_entropies}")
        print(f"Interaction weights: {weights}")
        dists = [dist ** weight for dist, weight in zip(dists, weights)]
        combined_dist = reduce(np.multiply, dists)
        print(f"Weighted Dists: {[list(dist) for dist in dists]}")
        print("Combined dist: ", combined_dist)
        final_dist = Categorical(torch.Tensor(combined_dist))
        return final_dist

    def reset(self):
        pass

