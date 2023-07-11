import torch
from torch import nn
from torch.distributions.categorical import Categorical


class InteractionPPONetwork(nn.Module):

    def __init__(self, in_dim_state, in_dim_agents, in_dim_other_agent, out_dim, lr=0.0001, device="cpu"):
        super(InteractionPPONetwork, self).__init__()
        self._hidden_composite_layer = nn.Sequential(
            nn.Linear(in_dim_state + in_dim_agents + in_dim_other_agent, 512),
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs_state, obs_agent, obs_other_agent, hidden_state=None):
        self._output = self._hidden_composite_layer(torch.cat((obs_state, obs_agent, obs_other_agent), dim=1))
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
        return action, logits_numpy, None

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

    def reset(self):
        pass


