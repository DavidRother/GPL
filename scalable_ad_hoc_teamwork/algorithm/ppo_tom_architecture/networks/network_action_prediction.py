import torch
from torch import nn
from torch.distributions.categorical import Categorical


class ActionPredictionNetwork(nn.Module):

    def __init__(self, in_dim_state, in_dim_other_agent, out_dim, lr=0.0001, device="cpu"):
        super(ActionPredictionNetwork, self).__init__()
        self.hidden_size = 256
        self._hidden_type_layers = nn.LSTM(in_dim_other_agent + in_dim_state, self.hidden_size)
        self._hidden_composite_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.device = device
        self._logits = nn.Sequential(nn.ReLU(), nn.Linear(256, out_dim))
        self._output = None
        self._agent_type_output = None
        self._hidden_state = self.init_hidden()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs_state, obs_other_agent, hidden_state=None):
        if not isinstance(obs_state, torch.Tensor):
            obs_state = torch.FloatTensor(obs_state).unsqueeze(0).to(self.device)
        if not isinstance(obs_other_agent, torch.Tensor):
            obs_other_agent = torch.FloatTensor(obs_other_agent).unsqueeze(0).to(self.device)
        if obs_other_agent.dim() == 3:
            concat_input = torch.cat((obs_state, obs_other_agent), dim=2)
        else:
            concat_input = torch.cat((obs_state, obs_other_agent), dim=1)
        if hidden_state is not None:
            old_hidden_state = hidden_state
            agent_type_output, self._hidden_state = self._hidden_type_layers(concat_input, old_hidden_state)
            if agent_type_output.dim() == 3:
                agent_type_output = agent_type_output.squeeze(0)
            self._agent_type_output = agent_type_output
        else:
            old_hidden_state = self._hidden_state
            self._agent_type_output, self._hidden_state = self._hidden_type_layers(concat_input, old_hidden_state)
        self._output = self._hidden_composite_layer(self._agent_type_output)
        logits = self._logits(self._output)
        return logits, old_hidden_state

    def import_from_h5(self, h5_file: str) -> None:
        self.load_state_dict(torch.load(h5_file).state_dict())

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
