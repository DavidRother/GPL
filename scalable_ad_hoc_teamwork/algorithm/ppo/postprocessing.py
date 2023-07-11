import numpy as np
import scipy.signal
from typing import Dict, Optional
from torch.distributions.categorical import Categorical
from scalable_ad_hoc_teamwork.algorithm.ppo.buffer import *

import torch


class Postprocessing:
    """Constant definitions for postprocessing."""

    ADVANTAGES = "advantages"
    VALUE_TARGETS = "value_targets"
    EGO_ADVANTAGES = "ego_advantages"
    EGO_VALUE_TARGETS = "ego_value_targets"


def adjust_nstep(n_step: int, gamma: float, batch: Buffer) -> None:
    """Rewrites `batch` to encode n-step rewards, dones, and next-obs.

    Observations and actions remain unaffected. At the end of the trajectory,
    n is truncated to fit in the traj length.

    Args:
        n_step (int): The number of steps to look ahead and adjust.
        gamma (float): The discount factor.
        batch (Buffer): The Buffer to adjust (in place).

    Examples:
        n-step=3
        Trajectory=o0 r0 d0, o1 r1 d1, o2 r2 d2, o3 r3 d3, o4 r4 d4=True o5
        gamma=0.9
        Returned trajectory:
        0: o0 [r0 + 0.9*r1 + 0.9^2*r2 + 0.9^3*r3] d3 o0'=o3
        1: o1 [r1 + 0.9*r2 + 0.9^2*r3 + 0.9^3*r4] d4 o1'=o4
        2: o2 [r2 + 0.9*r3 + 0.9^2*r4] d4 o1'=o5
        3: o3 [r3 + 0.9*r4] d4 o3'=o5
        4: o4 r4 d4 o4'=o5
    """

    assert not any(batch[DONES][:-1]), \
        "Unexpected done in middle of trajectory!"

    len_ = len(batch)

    # Shift NEXT_OBS and DONES.
    batch[NEXT_OBS] = np.concatenate(
        [
            batch[OBS][n_step:],
            np.stack([batch[NEXT_OBS][-1]] * min(n_step, len_))
        ],
        axis=0)
    batch[DONES] = np.concatenate(
        [
            batch[DONES][n_step - 1:],
            np.tile(batch[DONES][-1], min(n_step - 1, len_))
        ],
        axis=0)

    # Change rewards in place.
    for i in range(len_):
        for j in range(1, n_step):
            if i + j < len_:
                batch[REWARDS][i] += \
                    gamma**j * batch[REWARDS][i + j]


def compute_advantages(rollout: Buffer,
                       prosocial_rewards: np.ndarray,
                       last_r: float,
                       gamma: float = 0.9,
                       lambda_: float = 1.0,
                       use_gae: bool = True,
                       use_critic: bool = True):
    """
    Given a rollout, compute its value targets and the advantages.

    Args:
        rollout (Buffer): Buffer of a single trajectory.
        prosocial_rewards (np.ndarray): prosocial reward for agent.
        last_r (float): Value estimation for last observation.
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE.
        use_gae (bool): Using Generalized Advantage Estimation.
        use_critic (bool): Whether to use critic (value estimates). Setting
            this to False will use 0 as baseline.

    Returns:
        Buffer (Buffer): Object with experience from rollout and
            processed rewards.
    """

    # assert VF_PREDS in rollout or not use_critic, "use_critic=True but values not found"
    # assert use_critic or not use_gae,  "Can't use gae without using a value function"

    if use_gae:
        vpred_t = np.concatenate(
            [rollout[VF_PREDS],
             np.array([last_r])])
        rewards = rollout[REWARDS]
        delta_t = (prosocial_rewards + gamma * vpred_t[1:] - vpred_t[:-1])
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        rollout[Postprocessing.ADVANTAGES] = discount_cumsum(delta_t, gamma * lambda_)
        rollout[Postprocessing.VALUE_TARGETS] = (rollout[Postprocessing.ADVANTAGES] +
                                                 rollout[VF_PREDS].detach().cpu().numpy()).astype(np.float32)

    else:
        rewards_plus_v = np.concatenate(
            [prosocial_rewards,
             np.array([last_r])])
        discounted_returns = discount_cumsum(rewards_plus_v, gamma)[:-1].astype(np.float32)

        if use_critic:
            rollout[Postprocessing.
                    ADVANTAGES] = discounted_returns - rollout[VF_PREDS]
            rollout[Postprocessing.VALUE_TARGETS] = discounted_returns
        else:
            rollout[Postprocessing.ADVANTAGES] = discounted_returns
            rollout[Postprocessing.VALUE_TARGETS] = np.zeros_like(
                rollout[Postprocessing.ADVANTAGES])

    rollout[Postprocessing.ADVANTAGES] = rollout[Postprocessing.ADVANTAGES].tolist()
    rollout[Postprocessing.VALUE_TARGETS] = rollout[Postprocessing.VALUE_TARGETS].tolist()

    return rollout


def compute_prosocial_gae_for_sample_batch(
        agent,
        agent_config,
        sample_batch: Buffer,
        other_agent_batches: Optional[Dict[str, Buffer]] = None) -> Buffer:
    """Adds GAE (generalized advantage estimations) to a trajectory.

    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.

    Args:
        agent(Policy): The Policy used to generate the trajectory
            (`sample_batch`)
        agent_config:
        sample_batch (Buffer): The Buffer to postprocess.
        other_agent_batches (Optional[Dict[PolicyID, Buffer]]): Optional
            dict of AgentIDs mapping to other agents' trajectory data (from the
            same episode). NOTE: The other agents use the same policy.

    Returns:
        Buffer: The postprocessed, modified Buffer (or a new one).
    """

    gamma = 0.99
    lambda_ = 0.9
    use_gae = True
    use_critic = True

    # Trajectory is actually complete -> last r=0.0.
    if sample_batch[DONES][-1]:
        last_r = 0.0
        last_ego_r = 0.0
    # Trajectory has been truncated -> last r=VF estimate of last obs.
    else:
        # Input dict is provided to us automatically via the Model's
        # requirements. It's a single-timestep (last one in trajectory)
        # input_dict.
        # Create an input dict according to the Model's requirements.
        obs = torch.Tensor(sample_batch.get_last_obs()).unsqueeze(0)
        _ = agent(obs)
        last_r = agent.value_function().item()
    prosocial_rewards = sample_batch[REWARDS].detach().cpu().numpy()
    # Adds the policy logits, VF preds, and advantages to the batch,
    # using GAE ("generalized advantage estimation") or not.
    batch = compute_advantages(sample_batch, prosocial_rewards, last_r, gamma,  lambda_, use_gae=use_gae,
                               use_critic=use_critic)

    return batch


def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """Calculates the discounted cumulative sum over a reward sequence `x`.

    y[t] - discount*y[t+1] = x[t]
    reversed(y)[t] - discount*reversed(y)[t-1] = reversed(x)[t]

    Args:
        gamma (float): The discount factor gamma.

    Returns:
        np.ndarray: The sequence containing the discounted cumulative sums
            for each individual reward in `x` till the end of the trajectory.

    Examples:
        >>> x = np.array([0.0, 1.0, 2.0, 3.0])
        >>> gamma = 0.9
        >>> discount_cumsum(x, gamma)
        ... array([0.0 + 0.9*1.0 + 0.9^2*2.0 + 0.9^3*3.0,
        ...        1.0 + 0.9*2.0 + 0.9^2*3.0,
        ...        2.0 + 0.9*3.0,
        ...        3.0])
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


def postprocess(agents, ma_buffer, agent_configs):
    for player in agents:
        _ = agents[player](ma_buffer.buffer[player][OBS])
        ma_buffer.buffer[player][VF_PREDS] = agents[player].value_function().tolist()
        dist = Categorical(logits=ma_buffer.buffer[player][ACTION_DIST_INPUTS])
        ma_buffer.buffer[player][ACTION_PROB] = dist.probs[range(dist.probs.shape[0]), :, ma_buffer.buffer[player][ACTIONS].long()].detach().cpu().numpy().tolist()
        ma_buffer.buffer[player][ACTION_LOGP] = dist.logits[range(dist.probs.shape[0]), :, ma_buffer.buffer[player][ACTIONS].long()].detach().cpu().numpy().tolist()

    buf_dict = {player: ma_buffer.buffer[player].buffer_episodes() for player in agents}
    for player in agents:
        new_buf_list = []
        other_agents = [*agents]
        other_agents.remove(player)
        for idx, buf in enumerate(buf_dict[player]):
            other_agent_buf_dict = {p: buf_dict[p][idx] for p in buf_dict if p != player}
            new_buf_list.append(compute_prosocial_gae_for_sample_batch(agents[player], agent_configs[player],
                                                                       buf, other_agent_buf_dict))
        new_buffer = Buffer(ma_buffer.buffer[player].size)
        for buf in new_buf_list:
            for key in buf.data_struct:
                new_buffer.data_struct[key].extend(buf.data_struct[key])
        ma_buffer.buffer[player] = new_buffer

    return ma_buffer
