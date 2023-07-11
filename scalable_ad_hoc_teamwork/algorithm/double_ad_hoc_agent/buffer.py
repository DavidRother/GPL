from collections import defaultdict
import torch
import numpy as np


OBS = "obs"
CUR_OBS = "obs"
NEXT_OBS = "new_obs"
ACTIONS = "actions"
REWARDS = "rewards"
PREV_ACTIONS = "prev_actions"
PREV_REWARDS = "prev_rewards"
DONES = "dones"
INFOS = "infos"
SEQ_LENS = "seq_lens"
T = "t"

# Extra action fetches keys.
ACTION_DIST_INPUTS = "action_dist_inputs"
ACTION_PROB = "action_prob"
ACTION_LOGP = "action_logp"

# Uniquely identifies an episode.
EPS_ID = "eps_id"
# An env ID (e.g. the index for a vectorized sub-env).
ENV_ID = "env_id"

# Uniquely identifies a sample batch. This is important to distinguish RNN
# sequences from the same episode when multiple sample batches are
# concatenated (fusing sequences across batches can be unsafe).
UNROLL_ID = "unroll_id"

# Uniquely identifies an agent within an episode.
AGENT_INDEX = "agent_index"

# Value function predictions emitted by the behaviour policy.
VF_PREDS = "vf_preds"
QF_PREDS = "qf_preds"

# Hidden states for recurrent policies.
HIDDEN_STATES = "hidden_states"

LAST_OTHER_AGENTS_ACTION = "last_other_agents_action"
LAST_OTHER_AGENTS_ACTION_NEXT = "last_other_agents_action_next"

WORLD_STATE = "world_state"
WORLD_STATE_NEXT = "world_state_next"

AGENT_STATE = "agent_state"
AGENT_STATE_NEXT = "agent_state_next"

OTHER_STATE = "other_state"
OTHER_STATE_NEXT = "other_state_next"


class MultiAgentBuffer:

    def __init__(self, size, num_agents):
        self.num_agents = num_agents
        self.size = size
        self.buffer = {f"player_{idx}": Buffer(size) for idx in range(num_agents)}
        self.idx = 0

    def commit(self, obs, actions, next_obs, rewards, done, info, logits, hidden_state, last_other_agents_action,
               next_last_other_agents_action, step):
        for player in obs:
            self.buffer[player].commit(obs[player], actions[player], next_obs[player], rewards[player], done[player],
                                       info[player], logits[player], hidden_state[player],
                                       last_other_agents_action[player], next_last_other_agents_action[player], step)
        self.idx += 1

    def __len__(self):
        return self.size

    def reset(self):
        self.buffer = {f"player_{idx}": Buffer(self.size) for idx in range(self.num_agents)}
        self.idx = 0

    def full(self):
        return self.idx == self.size


class Buffer:

    def __init__(self, size):
        self.size = size
        self.data_struct = defaultdict(list)

    def commit(self, obs, actions, next_obs, rewards, done, info, logits, hidden_state, last_other_agents_action,
               next_last_other_agents_action, step):
        self.data_struct[OBS].append(obs)
        self.data_struct[ACTIONS].append(actions)
        self.data_struct[NEXT_OBS].append(next_obs)
        self.data_struct[REWARDS].append(rewards)
        self.data_struct[DONES].append(done)
        self.data_struct[INFOS].append(info)
        self.data_struct[ACTION_DIST_INPUTS].append(logits)
        self.data_struct[HIDDEN_STATES].append(hidden_state)
        self.data_struct[LAST_OTHER_AGENTS_ACTION].append(last_other_agents_action)
        self.data_struct[LAST_OTHER_AGENTS_ACTION_NEXT].append(next_last_other_agents_action)
        self.data_struct[T].append(step)

    def __getitem__(self, item):
        try:
            if isinstance(self.data_struct[item][0], np.ndarray):
                return torch.Tensor(np.array(self.data_struct[item]))
            else:
                return torch.Tensor(self.data_struct[item])
        except:
            return self.data_struct[item]

    def __setitem__(self, key, value):
        assert len(value) == self.size
        self.data_struct[key] = value

    def __len__(self):
        return self.size

    def buffer_episodes(self):
        buf_list = []
        done_indices = [0] + [i + 1 for i, x in enumerate(self.data_struct[DONES]) if x]
        if done_indices[-1] != self.size:
            done_indices.append(self.size)
        for start_index, end_index in zip(done_indices[:-1], done_indices[1:]):
            buf = Buffer(end_index - start_index)
            for item in self.data_struct:
                buf[item] = self.data_struct[item][start_index:end_index]
            buf_list.append(buf)
        return buf_list

    def build_batches(self, num_batches):
        batch_size = int(self.size // num_batches)
        buf_list = []
        inds = np.arange(self.size, )
        np.random.shuffle(inds)
        for start in range(0, self.size, batch_size):
            end = start + batch_size
            batch_indices = inds[start:end]
            buf = Buffer(batch_size)
            for item in self.data_struct:
                l = self.data_struct[item]
                buf[item] = [l[idx] for idx in batch_indices]
            buf_list.append(buf)
        return buf_list

    def get_last_obs(self):
        return self.data_struct[NEXT_OBS][-1]

    def get_last_interaction_obs(self):
        world_state = self.data_struct[WORLD_STATE]
        agent_state = self.data_struct[AGENT_STATE]
        other_state = self.data_struct[OTHER_STATE]
        hidden_state = self.data_struct[HIDDEN_STATES]
        return world_state[-1], agent_state[-1], other_state[-1], hidden_state[-1]

    def reset(self):
        self.data_struct = defaultdict(list)

    def full(self):
        return len(self.data_struct[OBS]) == self.size
