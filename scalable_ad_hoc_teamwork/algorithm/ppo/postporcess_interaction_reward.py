from scalable_ad_hoc_teamwork.algorithm.ppo.buffer import *


def postprocess_interaction_reward(ego_agent, buffer, other_agent_model, device):
    ego_buf_list = buffer.buffer[ego_agent].buffer_episodes()
    buffer_list = []
    for ego_buffer in ego_buf_list:
        buffer_list.append(postprocess_episode_interaction_reward(ego_buffer, other_agent_model, device))

    player = ego_agent
    new_buffer = Buffer(buffer.buffer[player].size)
    for buf in buffer_list:
        for key in buf.data_struct:
            new_buffer.data_struct[key].extend(buf.data_struct[key])
    buffer.buffer[player] = new_buffer

    return buffer


def postprocess_episode_interaction_reward(ego_buffer, other_agent_model, device):
    batch_other_agent_state = ego_buffer[OBS]
    batch_other_agent_state = torch.FloatTensor(batch_other_agent_state).to(device)
    batch_other_agent_next_state = ego_buffer[NEXT_OBS]
    batch_other_agent_next_state = torch.FloatTensor(batch_other_agent_next_state).to(device)
    current_q = other_agent_model.get_q(batch_other_agent_state)
    current_v = other_agent_model.get_v(current_q).detach().cpu().numpy()
    next_q = other_agent_model.get_q(batch_other_agent_next_state)
    next_v = other_agent_model.get_v(next_q).detach().cpu().numpy()
    rewards = ego_buffer[REWARDS]
    interaction_rewards = next_v.squeeze() + rewards.detach().cpu().numpy() - current_v.squeeze()
    ego_buffer[REWARDS] = interaction_rewards.tolist()

    return ego_buffer
