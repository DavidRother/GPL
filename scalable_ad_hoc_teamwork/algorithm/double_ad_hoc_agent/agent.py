import copy

from scalable_ad_hoc_teamwork.algorithm.ppo_lstm.learning.buffer import MultiAgentBuffer
from scalable_ad_hoc_teamwork.algorithm.ppo_lstm.networks.network_task import TaskPPONetwork
from scalable_ad_hoc_teamwork.algorithm.ppo_lstm.networks.network_interaction import InteractionPPONetwork
from scalable_ad_hoc_teamwork.algorithm.ppo_lstm.learning.postprocessing_task import postprocess as postprocess_task
from scalable_ad_hoc_teamwork.algorithm.ppo_lstm.learning.ppo_loss_task import ppo_surrogate_loss as ppo_surrogate_loss_task
from scalable_ad_hoc_teamwork.algorithm.ppo_lstm.learning.postprocessing_interaction \
    import postprocess as postprocess_interaction
from scalable_ad_hoc_teamwork.algorithm.ppo_lstm.learning.ppo_loss_interaction \
    import ppo_surrogate_loss as ppo_surrogate_loss_interaction
from pathlib import Path
import numpy as np
import torch
import dill as pickle


class Agent:

    LEARN_INTERACTION = "learn_interaction"
    LEARN_TASK = "learn_task"
    INFERENCE = "inference"

    def __init__(self, obs_dim_state, obs_dim_agent, n_actions, memory_size, num_epochs, learning_rate,
                 num_batches, state_segmentation_func, device):
        self.task_models = {}
        self.interaction_models = {}
        self.credit_models = {}
        self.interaction_credit_models = {}
        self.memory_size = memory_size
        self.memory = MultiAgentBuffer(memory_size, 1)
        self.obs_dim_state = obs_dim_state
        self.obs_dim_agent = obs_dim_agent
        self.n_actions = n_actions
        self.device = device
        self.current_active_tasks = []
        self.current_active_interaction_tasks = []
        self.learning_rate = learning_rate
        self.last_logits = None
        self.last_hidden_state = None
        self.step = 0
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.mode = self.LEARN_TASK
        self.policy_momentum = 0.8
        self.weighted_momentum = {}
        self.other_agent_tasks = {}
        self.other_agents_last_actions = {}
        self.next_other_agents_last_actions = {}
        self.agent_id = None
        self.state_segmentation_func = state_segmentation_func

    def task_segmentation(self, state):
        composite_state = self.state_segmentation_func(state, self.agent_id, [])
        new_state = np.concatenate([composite_state[0], composite_state[1]], axis=1)
        return new_state

    def interaction_segmentation(self, state, other_id, last_agent_action):
        composite_state = self.state_segmentation_func(state, self.agent_id, other_id)
        new_state_world = composite_state[0]
        new_state_agents = np.concatenate([composite_state[1], composite_state[2]], axis=1)
        new_state_other = self.one_hot(composite_state[2], last_agent_action[other_id])
        return new_state_world, new_state_agents, new_state_other

    def select_action(self, state, **kwargs):
        task_weights, task_credit_weights = self.get_task_weights(state)
        interaction_weights, interaction_credit_weights = self.get_interaction_weights(state)
        all_weights = torch.Tensor(np.array(task_weights + interaction_weights))
        all_credit_weights = torch.Tensor(np.array(task_credit_weights + interaction_credit_weights))
        categorical = self.compute_distribution(all_weights, all_credit_weights)
        action = categorical.sample()
        return action.item()

    def info_callback(self, agent_id, info):
        self.other_agents_last_actions = self.next_other_agents_last_actions
        self.next_other_agents_last_actions = {player: info[player]["action"] for player in info if player != agent_id}
        self.other_agent_tasks = {player: info[player]["task"] for player in info if player != agent_id}

    def get_task_weights(self, state):
        task_weights = []
        task_credit_weights = []
        for task in self.current_active_tasks:
            torch_state = torch.Tensor(self.task_segmentation(state.reshape(1, -1)))
            action, logits = self.task_models[task].select_action(torch_state)
            self.last_logits = logits
            exps = np.exp(logits - np.max(logits))  # subtract max for numerical stability
            probs = exps / np.sum(exps)
            task_weights.append(probs)
            values = self.task_models[task].value_function()
            credit_weights = (values / torch.sum(values)).detach().cpu().numpy()
            task_credit_weights.append(credit_weights)
        return task_weights, task_credit_weights

    def get_interaction_weights(self, state):
        interaction_weights = []
        interaction_credit_weights = []
        if self.mode == self.LEARN_TASK:
            return interaction_weights, interaction_credit_weights
        for other_id, task in self.other_agent_tasks.items():
            env_state, self_player_state, other_player_state = \
                self.interaction_segmentation(state.reshape(1, -1), other_id, self.next_other_agents_last_actions)
            env_state = torch.Tensor(env_state.reshape(1, -1))
            self_player_state = torch.Tensor(self_player_state.reshape(1, -1))
            other_player_state = torch.Tensor(other_player_state.reshape(1, -1))
            action, logits, hidden_state = self.interaction_models[task].select_action(env_state, self_player_state,
                                                                                       other_player_state)
            self.last_logits = logits
            self.last_hidden_state = hidden_state
            exps = np.exp(logits - np.max(logits))  # subtract max for numerical stability
            probs = exps / np.sum(exps)
            interaction_weights.append(probs)
            values = self.interaction_models[task].value_function()
            credit_weights = (values / torch.sum(values)).detach().cpu().numpy()
            interaction_credit_weights.append(credit_weights)
        return interaction_weights, interaction_credit_weights

    def update_agent_id(self, agent_id):
        self.agent_id = agent_id

    def compute_distribution(self, distribution_weights, credit_weights):
        normalized_credit_weights = credit_weights / credit_weights.sum(dim=1).unsqueeze(1)
        weighted_weights = normalized_credit_weights * distribution_weights
        if self.policy_momentum:
            if self.agent_id not in self.weighted_momentum:
                self.weighted_momentum[self.agent_id] = weighted_weights
            else:
                self.weighted_momentum[self.agent_id] = self.policy_momentum * self.weighted_momentum[self.agent_id] + (1 - self.policy_momentum) * weighted_weights
            average_distribution = torch.sum(self.weighted_momentum[self.agent_id], dim=0)
        else:
            average_distribution = torch.sum(weighted_weights, dim=0)
        categorical = torch.distributions.Categorical(average_distribution)
        return categorical

    def add_task(self, task):
        new_model = TaskPPONetwork(self.obs_dim_state + self.obs_dim_agent, self.n_actions,
                                   self.learning_rate, self.device)
        self.task_models[task] = new_model

    def add_interaction(self, interaction):
        new_model = InteractionPPONetwork(self.obs_dim_state, self.obs_dim_agent * 2,
                                          self.obs_dim_agent + self.n_actions, self.n_actions,
                                          self.learning_rate, self.device)
        self.interaction_models[interaction] = new_model

    def switch_active_tasks(self, new_active_tasks):
        self.current_active_tasks = new_active_tasks
        self.memory.reset()

    def switch_active_interactions(self, new_active_interactions):
        self.current_active_interaction_tasks = new_active_interactions
        self.memory.reset()

    def store_transition(self, transition, other_transitions=None, **kwargs):
        if self.mode is self.INFERENCE:
            return
        state = {"player_0": transition[0]}
        next_state = {"player_0": transition[3]}
        action = {"player_0": transition[1]}
        if self.mode is self.LEARN_TASK:
            reward = {"player_0": transition[2]}
        else:
            # reward = {"player_0": transition[2]}
            reward = {"player_0": sum([tran[2] for tran in other_transitions])}
        done = {"player_0": transition[4]}
        info = {"player_0": None}
        logits = {"player_0": self.last_logits}
        try:
            last_hidden_state = (self.last_hidden_state[0].detach(), self.last_hidden_state[1].detach())
        except:
            last_hidden_state = None
        hidden_state = {"player_0": last_hidden_state}
        other_agents_last_actions = {"player_0": self.other_agents_last_actions}
        next_other_agents_last_actions = {"player_0": self.next_other_agents_last_actions}
        self.memory.commit(state, action, next_state, reward, done, info, logits, hidden_state,
                           other_agents_last_actions, next_other_agents_last_actions, self.step)
        self.last_logits = None
        if done:
            self.step = 0
        else:
            self.step += 1

    def learn_step(self, **kwargs):
        if self.memory.full():
            if self.mode is self.LEARN_TASK:
                self.learn_step_task()
            elif self.mode is self.LEARN_INTERACTION:
                self.learn_step_interaction()
            else:
                return

    def learn_step_task(self):
        active_model = self.get_active_task_model()
        postprocess = postprocess_task
        ppo_surrogate_loss = ppo_surrogate_loss_task
        seg_func = self.task_segmentation
        agents = {f"player_0": active_model}
        agent_1_config = {"task_segmentation_func": seg_func, "agent_id": self.agent_id,
                          'other_agent_id': list(self.other_agent_tasks.keys())[0]}
        agent_configs = {"player_0": agent_1_config}
        postprocess(agents, self.memory, agent_configs)
        self.core_learn_step(agents, agent_configs, ppo_surrogate_loss, active_model)

    def learn_step_interaction(self):
        active_model = self.get_active_interaction_model()
        postprocess = postprocess_interaction
        ppo_surrogate_loss = ppo_surrogate_loss_interaction
        seg_func = self.interaction_segmentation
        old_hidden_state = active_model._hidden_state
        agents = {f"player_0": active_model}
        agent_1_config = {"task_segmentation_func": seg_func, "agent_id": self.agent_id,
                          'other_agent_id': list(self.other_agent_tasks.keys())[0]}
        agent_configs = {"player_0": agent_1_config}
        postprocess(agents, self.memory, agent_configs)

        self.core_learn_step(agents, agent_configs, ppo_surrogate_loss, active_model)
        active_model._hidden_state = old_hidden_state

    def core_learn_step(self, agents, agent_configs, ppo_surrogate_loss, active_model):
        for epoch in range(self.num_epochs):
            for player in agents:
                for batch in self.memory.buffer[player].build_batches(self.num_batches):
                    loss, stats = ppo_surrogate_loss(agents[player], batch, agent_configs[player])

                    active_model.optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), config.max_grad_norm)
                    active_model.optimizer.step()
        self.memory.reset()

    def get_active_task_model(self):
        return self.task_models[self.get_active_task()]

    def get_active_interaction_model(self):
        return self.interaction_models[self.get_active_interaction()]

    def get_active_interaction(self):
        return self.current_active_interaction_tasks[0]

    def get_active_task(self):
        return self.current_active_tasks[0]

    def switch_mode(self, new_mode):
        assert new_mode in [self.LEARN_TASK, self.LEARN_INTERACTION, self.INFERENCE]
        self.mode = new_mode
        self.memory.reset()

    def reset(self):
        for interaction_model in self.interaction_models.values():
            interaction_model.reset()
        self.weighted_momentum = {}
        self.other_agent_tasks = {}
        self.other_agents_last_actions = {}
        self.next_other_agents_last_actions = {}

    def save_agent(self, base_path, clear_memory=True):
        if clear_memory:
            self.memory.reset()
        Path(base_path).parent.mkdir(parents=True, exist_ok=True)
        with open(base_path, "wb") as output_file:
            pickle.dump(self, output_file)

    def one_hot(self, x, concat_vector):
        if isinstance(x, list):
            return [self.one_hot_single_element(x_i, vec_i) for x_i, vec_i in zip(x, concat_vector)]
        else:
            return self.one_hot_single_element(x, concat_vector)

    def one_hot_single_element(self, concat_vector, x):
        one_hot_vector = np.zeros((1, self.n_actions))
        one_hot_vector[:, x] = 1
        return np.concatenate((one_hot_vector, concat_vector), axis=1)

    def eval_mode(self):
        return _AgentEvalContext(self)


class _AgentEvalContext:

    def __init__(self, agent: Agent):
        self._agent = agent
        self.last_logits = None
        self.last_hidden_state = None
        self.step = None
        self.mode = None
        self.policy_momentum = None
        self.weighted_momentum = None
        self.other_agent_tasks = None
        self.other_agents_last_actions = None
        self.next_other_agents_last_actions = None
        self.agent_id = None

    def __enter__(self):
        self.last_logits = self._agent.last_logits
        self.last_hidden_state = copy.copy(self._agent.last_hidden_state)
        self.step = copy.copy(self._agent.step)
        self.mode = self._agent.mode
        self.weighted_momentum = copy.copy(self._agent.weighted_momentum)
        self.other_agent_tasks = copy.copy(self._agent.other_agent_tasks)
        self.other_agents_last_actions = copy.copy(self._agent.other_agents_last_actions)
        self.next_other_agents_last_actions = copy.copy(self._agent.next_other_agents_last_actions)
        self.agent_id = self._agent.agent_id
        return self._agent

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._agent.last_logits = self.last_logits
        self._agent.last_hidden_state = self.last_hidden_state
        self._agent.step = self.step
        self._agent.mode = self.mode
        self._agent.weighted_momentum = self.weighted_momentum
        self._agent.other_agent_tasks = self.other_agent_tasks
        self._agent.other_agents_last_actions = self.other_agents_last_actions
        self._agent.next_other_agents_last_actions = self.next_other_agents_last_actions
        self._agent.agent_id = self.agent_id

