import copy

from scalable_ad_hoc_teamwork.algorithm.ppo_tom_architecture.learning.buffer import MultiAgentBuffer
from scalable_ad_hoc_teamwork.algorithm.ppo_tom_architecture.networks.network_task import TaskPPONetwork
from scalable_ad_hoc_teamwork.algorithm.ppo_tom_architecture.networks.network_interaction import InteractionPPONetwork
from scalable_ad_hoc_teamwork.algorithm.ppo_tom_architecture.networks.network_interaction_monolith \
    import InteractionPPONetwork as InteractionPPONetworkMonolith
from scalable_ad_hoc_teamwork.algorithm.ppo_tom_architecture.networks.network_action_prediction \
    import ActionPredictionNetwork
from scalable_ad_hoc_teamwork.algorithm.ppo_tom_architecture.learning.postprocessing_task import postprocess as postprocess_task
from scalable_ad_hoc_teamwork.algorithm.ppo_tom_architecture.learning.ppo_loss_task import ppo_surrogate_loss as ppo_surrogate_loss_task
from scalable_ad_hoc_teamwork.algorithm.ppo_tom_architecture.learning.postprocessing_interaction \
    import postprocess as postprocess_interaction
from scalable_ad_hoc_teamwork.algorithm.ppo_tom_architecture.learning.ppo_loss_interaction \
    import ppo_surrogate_loss as ppo_surrogate_loss_interaction
from scalable_ad_hoc_teamwork.algorithm.ppo_tom_architecture.learning.action_prediction_loss \
    import train_action_prediction
from pathlib import Path
from enum import Enum
import numpy as np
import torch
import dill as pickle


class Architecture(Enum):
    MONOLITH = "Monolith"
    SEPARATED = "Separated"


class Agent:

    LEARN_INTERACTION = "learn_interaction"
    LEARN_TASK = "learn_task"
    INFERENCE = "inference"

    def __init__(self, obs_dim_state, obs_dim_agent, n_actions, memory_size, memory_size_action_prediction, num_epochs,
                 learning_rate, num_batches, gamma, state_segmentation_func, preprocessor, interaction_architecture,
                 device):
        self.task_models = {}
        self.interaction_models = {}
        self.action_prediction_models = {}
        self.credit_models = {}
        self.interaction_credit_models = {}
        self.memory_size = memory_size
        self.memory = MultiAgentBuffer(memory_size, 1)
        self.memory_tom_model = MultiAgentBuffer(memory_size_action_prediction, 1)
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
        self.preprocessor = preprocessor
        self.architecture = interaction_architecture
        self.gamma = gamma

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
        with torch.no_grad():
            preprocessed_state = self.preprocessor.process(state)
            task_weights, task_credit_weights = self.get_task_weights(preprocessed_state)
            interaction_weights, interaction_credit_weights = self.get_interaction_weights(preprocessed_state)
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
            action_prediction, hidden_state = self.action_prediction_models[task](env_state, other_player_state)
            other_player_state[:, -self.n_actions:] = torch.softmax(action_prediction, dim=1)
            action, logits, _ = self.interaction_models[task].select_action(env_state, self_player_state,
                                                                            other_player_state)
            self.last_logits = logits
            self.last_hidden_state = hidden_state
            probs, credit_weights = self.extract_weights(logits, task)
            interaction_weights.append(probs)
            interaction_credit_weights.append(credit_weights)
        return interaction_weights, interaction_credit_weights

    def extract_weights(self, logits, task):
        exps = np.exp(logits - np.max(logits))  # subtract max for numerical stability
        probs = exps / np.sum(exps)
        values = self.interaction_models[task].value_function()
        credit_weights = (values / torch.sum(values)).detach().cpu().numpy()
        return probs, credit_weights

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
        if self.architecture == Architecture.MONOLITH:
            new_model = InteractionPPONetworkMonolith(self.obs_dim_state, self.obs_dim_agent * 2,
                                                      self.obs_dim_agent + self.n_actions, self.n_actions,
                                                      self.learning_rate, self.device)
        elif self.architecture == Architecture.SEPARATED:
            new_model = InteractionPPONetwork(self.obs_dim_state, self.obs_dim_agent * 2,
                                              self.obs_dim_agent + self.n_actions, self.n_actions,
                                              self.learning_rate, self.device)
        else:
            raise ValueError("Architecture not recognized")
        self.interaction_models[interaction] = new_model
        self.add_action_prediction(interaction)

    def add_action_prediction(self, interaction):
        new_model = ActionPredictionNetwork(self.obs_dim_state, self.obs_dim_agent + self.n_actions, self.n_actions,
                                            self.learning_rate, self.device)
        self.action_prediction_models[interaction] = new_model

    def switch_active_tasks(self, new_active_tasks):
        self.current_active_tasks = new_active_tasks
        self.memory.reset()

    def switch_active_interactions(self, new_active_interactions):
        self.current_active_interaction_tasks = new_active_interactions
        self.memory.reset()

    def store_transition(self, transition, other_transitions=None, **kwargs):
        if self.mode is self.INFERENCE:
            return
        state = {"player_0": self.preprocessor.process(transition[0])}
        next_state = {"player_0": self.preprocessor.process(transition[3])}
        action = {"player_0": transition[1]}
        if self.mode is self.LEARN_TASK:
            reward = {"player_0": transition[2]}
        else:
            # reward = {"player_0": transition[2]}
            reward = {"player_0": sum([tran[2] for tran in other_transitions])}
        done = {"player_0": transition[4]}
        truncation = {"player_0": transition[5]}
        info = {"player_0": None}
        logits = {"player_0": self.last_logits}
        try:
            last_hidden_state = (self.last_hidden_state[0].detach(), self.last_hidden_state[1].detach())
        except:
            last_hidden_state = None
        hidden_state = {"player_0": last_hidden_state}
        other_agents_last_actions = {"player_0": self.other_agents_last_actions}
        next_other_agents_last_actions = {"player_0": self.next_other_agents_last_actions}
        self.memory.commit(state, action, next_state, reward, done, truncation, info, logits, hidden_state,
                           other_agents_last_actions, next_other_agents_last_actions, self.step)
        self.memory_tom_model.commit(state, action, next_state, reward, done, truncation, info, logits, hidden_state,
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
                          'other_agent_id': None, "gamma": self.gamma}
        agent_configs = {"player_0": agent_1_config}
        postprocess(agents, self.memory, agent_configs)
        self.core_learn_step(agents, agent_configs, ppo_surrogate_loss, active_model)

    def learn_step_interaction(self):
        active_model = self.get_active_interaction_model()
        active_action_prediction_model = self.get_active_action_prediction_model()
        postprocess = postprocess_interaction
        ppo_surrogate_loss = ppo_surrogate_loss_interaction
        seg_func = self.interaction_segmentation
        old_hidden_state = active_action_prediction_model._hidden_state
        agents = {f"player_0": active_model}
        agent_1_config = {"task_segmentation_func": seg_func, "agent_id": self.agent_id,
                          'other_agent_id': list(self.other_agent_tasks.keys())[0], "gamma": self.gamma,
                          "action_prediction_model": active_action_prediction_model}
        agent_configs = {"player_0": agent_1_config}
        postprocess(agents, self.memory, agent_configs)
        self.learn_step_tom_model(agents, active_action_prediction_model, list(self.other_agent_tasks.keys())[0])
        self.core_learn_step(agents, agent_configs, ppo_surrogate_loss, active_model)
        active_action_prediction_model._hidden_state = old_hidden_state
        self.memory.reset()

    def core_learn_step(self, agents, agent_configs, ppo_surrogate_loss, active_model):
        for epoch in range(self.num_epochs):
            for player in agents:
                for batch in self.memory.buffer[player].build_batches(self.num_batches):
                    loss, stats = ppo_surrogate_loss(agents[player], batch, agent_configs[player])

                    active_model.optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), config.max_grad_norm)
                    active_model.optimizer.step()

    def learn_step_tom_model(self, agents, active_model, other_agent_id):
        for epoch in range(self.num_epochs):
            for player in agents:
                for batch in self.memory.buffer[player].build_batches(self.num_batches):
                    loss = train_action_prediction(active_model, batch, other_agent_id)

                    active_model.optimizer.zero_grad()
                    loss.backward()
                    active_model.optimizer.step()

    def get_active_task_model(self):
        return self.task_models[self.get_active_task()]

    def get_active_interaction_model(self):
        return self.interaction_models[self.get_active_interaction()]

    def get_active_action_prediction_model(self):
        return self.action_prediction_models[self.get_active_prediction()]

    def get_active_prediction(self):
        return self.current_active_interaction_tasks[0]

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
        for task_model in self.task_models.values():
            task_model.reset()
        for action_model in self.action_prediction_models.values():
            action_model.reset()
        self.weighted_momentum = {}
        self.other_agent_tasks = {}
        self.other_agents_last_actions = {}
        self.next_other_agents_last_actions = {}

    def save_agent(self, base_path, clear_memory=True):
        if clear_memory:
            self.memory.reset()
            self.memory_tom_model.reset()
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

