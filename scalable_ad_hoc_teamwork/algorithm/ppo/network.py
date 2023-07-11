import torch
from torch import nn
from torch.distributions.categorical import Categorical


class PPONetwork(nn.Module):

    def __init__(self, in_dim, out_dim, lr=0.0001, device="cpu"):
        super(PPONetwork, self).__init__()
        self._hidden_layers = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.device = device
        self._logits = nn.Sequential(nn.Tanh(), nn.Linear(256, out_dim))
        self._value_branch = nn.Sequential(nn.Tanh(), nn.Linear(256, 1))
        self._output = None
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs):
        self._output = self._hidden_layers(obs)
        logits = self._logits(self._output)
        return logits

    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return self._value_branch(self._output).squeeze(1)

    def import_from_h5(self, h5_file: str) -> None:
        self.load_state_dict(torch.load(h5_file).state_dict())

    def select_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self(state).squeeze(1)
        logits_numpy = logits.detach().cpu().numpy()
        action = Categorical(logits=logits).sample().item()
        return action, logits_numpy

    def get_dist(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self(state).squeeze(1)
        dist = Categorical(logits=logits)
        return dist
