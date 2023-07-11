import torch
from torch import nn
from torch.distributions.categorical import Categorical


class InteractionPPONetwork(nn.Module):

    def __init__(self, in_dim_state, in_dim_agent, out_dim, lr=0.0001, device="cpu"):
        super(InteractionPPONetwork, self).__init__()
        self._hidden_layers = nn.Sequential(
            nn.Linear(in_dim_state, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self._hidden_agent_layers = nn.Sequential(
            nn.Linear(in_dim_agent * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self._hidden_state = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
        self._hidden_type_layers = nn.LSTM(in_dim_agent, 128)
        self._hidden_composite_layer = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
        )
        self.device = device
        self._logits = nn.Sequential(nn.Tanh(), nn.Linear(256, out_dim))
        self._value_branch = nn.Sequential(nn.Tanh(), nn.Linear(256, 1))
        self._output = None
        self._state_output = None
        self._agent_output = None
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs_state, obs_agent, obs_other_agent):
        self._state_output = self._hidden_layers(obs_state)
        self._agent_output = self._hidden_agent_layers(obs_agent)
        self._output = self._hidden_composite_layer(torch.cat((self._state_output, self._agent_output), dim=1))
        logits = self._logits(self._output)
        return logits

    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return self._value_branch(self._output).squeeze(1)

    def import_from_h5(self, h5_file: str) -> None:
        self.load_state_dict(torch.load(h5_file).state_dict())

    def select_action(self, obs_state, obs_agent):
        if not isinstance(obs_state, torch.Tensor):
            obs_state = torch.FloatTensor(obs_state).unsqueeze(0).to(self.device)
        if not isinstance(obs_agent, torch.Tensor):
            obs_agent = torch.FloatTensor(obs_agent).unsqueeze(0).to(self.device)
        logits = self(obs_state, obs_agent).squeeze(1)
        logits_numpy = logits.detach().cpu().numpy()
        action = Categorical(logits=logits).sample().item()
        return action, logits_numpy

    def get_dist(self, obs_state, obs_agent):
        if not isinstance(obs_state, torch.Tensor):
            obs_state = torch.FloatTensor(obs_state).unsqueeze(0).to(self.device)
        if not isinstance(obs_agent, torch.Tensor):
            obs_agent = torch.FloatTensor(obs_agent).unsqueeze(0).to(self.device)
        logits = self(obs_state, obs_agent).squeeze(1)
        dist = Categorical(logits=logits)
        return dist
