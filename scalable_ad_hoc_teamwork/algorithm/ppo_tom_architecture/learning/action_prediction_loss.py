import torch.nn.functional as F
import torch
from scalable_ad_hoc_teamwork.algorithm.ppo_tom_architecture.learning.buffer import WORLD_STATE, GT_OTHER_STATE, \
    LAST_OTHER_AGENTS_ACTION_NEXT, HIDDEN_STATES


def train_action_prediction(network, train_batch, other_agent_id):
    loss_fn = F.cross_entropy

    state_batch = train_batch[WORLD_STATE].unsqueeze(0)
    other_state_batch = train_batch[GT_OTHER_STATE].unsqueeze(0)
    hidden_states = train_batch[HIDDEN_STATES]

    hxs = [t[0] for t in hidden_states]
    cxs = [t[1] for t in hidden_states]

    hx = torch.cat(hxs, dim=0).unsqueeze(0)
    cx = torch.cat(cxs, dim=0).unsqueeze(0)
    new_hidden_states = (hx, cx)

    action_batch = torch.LongTensor([a[other_agent_id] for a in train_batch[LAST_OTHER_AGENTS_ACTION_NEXT]])

    # Forward pass
    action_logits, _ = network(state_batch, other_state_batch, new_hidden_states)

    # Compute loss
    loss = loss_fn(action_logits, action_batch)

    return loss
