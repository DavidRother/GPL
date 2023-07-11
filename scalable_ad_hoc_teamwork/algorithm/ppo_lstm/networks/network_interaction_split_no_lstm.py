import torch
from torch import nn
from torch.distributions.categorical import Categorical


class InteractionPPONetwork(nn.Module):

    def __init__(self, in_dim_state, in_dim_agents, in_dim_other_agent, out_dim, lr=0.0001, device="cpu"):
        super(InteractionPPONetwork, self).__init__()
        self.hidden_size = 128
        self._hidden_layers = nn.Sequential(
            nn.Linear(in_dim_state, 256),
            nn.ReLU(),
        )
        self._hidden_agent_layers = nn.Sequential(
            nn.Linear(in_dim_agents, 256),
            nn.ReLU(),
        )
        self._hidden_composite_layer = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.device = device
        self._logits = nn.Sequential(nn.ReLU(), nn.Linear(256, out_dim))
        self._value_branch = nn.Sequential(nn.ReLU(), nn.Linear(256, 1))
        self._q_branch = nn.Sequential(nn.ReLU(), nn.Linear(256, out_dim))
        self._output = None
        self._state_output = None
        self._agent_output = None
        self._agent_type_output = None
        self._hidden_state = self.init_hidden()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs_state, obs_agent, obs_other_agent, hidden_state=None):
        self._state_output = self._hidden_layers(obs_state)
        self._agent_output = self._hidden_agent_layers(obs_agent)
        self._output = self._hidden_composite_layer(torch.cat((self._state_output, self._agent_output), dim=1))
        logits = self._logits(self._output)
        return logits

    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return self._value_branch(self._output).squeeze(1)

    def q_function(self):
        assert self._output is not None, "must call forward first!"
        return self._q_branch(self._output)

    def import_from_h5(self, h5_file: str) -> None:
        self.load_state_dict(torch.load(h5_file).state_dict())

    def select_action(self, obs_state, obs_agent, obs_other_agent):
        if not isinstance(obs_state, torch.Tensor):
            obs_state = torch.FloatTensor(obs_state).unsqueeze(0).to(self.device)
        if not isinstance(obs_agent, torch.Tensor):
            obs_agent = torch.FloatTensor(obs_agent).unsqueeze(0).to(self.device)
        if not isinstance(obs_other_agent, torch.Tensor):
            obs_other_agent = torch.FloatTensor(obs_other_agent).unsqueeze(0).to(self.device)
        logits = self(obs_state, obs_agent, obs_other_agent).squeeze(1)
        logits_numpy = logits.detach().cpu().numpy()
        action = Categorical(logits=logits).sample().item()
        return action, logits_numpy, self._hidden_state

    def get_dist(self, obs_state, obs_agent, obs_other_agent):
        if not isinstance(obs_state, torch.Tensor):
            obs_state = torch.FloatTensor(obs_state).unsqueeze(0).to(self.device)
        if not isinstance(obs_agent, torch.Tensor):
            obs_agent = torch.FloatTensor(obs_agent).unsqueeze(0).to(self.device)
        if not isinstance(obs_other_agent, torch.Tensor):
            obs_other_agent = torch.FloatTensor(obs_other_agent).unsqueeze(0).to(self.device)
        logits = self(obs_state, obs_agent, obs_other_agent).squeeze(1)
        dist = Categorical(logits=logits)
        return dist

    def init_hidden(self):
        # Initialize hidden and cell states
        # Here `num_layers` is 1 and `num_directions` is also 1 because we're using a single layer unidirectional
        # LSTM by default.
        # You can modify these values if you're using a multilayer or bidirectional LSTM.
        num_layers = 1
        num_directions = 1
        batch_size = 1

        # Initialize hidden state and cell state to zeros
        h0 = torch.zeros(num_layers * num_directions, self.hidden_size, device=self.device)
        c0 = torch.zeros(num_layers * num_directions, self.hidden_size, device=self.device)

        return h0, c0

    def reset(self):
        self._hidden_state = self.init_hidden()


