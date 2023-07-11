import torch.distributions.kl as kl

from scalable_ad_hoc_teamwork.algorithm.ppo_tom_architecture.learning.postprocessing_interaction import *


def ppo_surrogate_loss(agent, train_batch, agent_config):
    """Constructs the loss for Proximal Policy Objective.
    Args:
        agent: The Policy to calculate the loss for.
        train_batch (SampleBatch): The training data.
        agent_config
    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """

    use_critic = True
    vf_clip_param = 0.3
    clip_param = 0.3
    kl_coeff = 0.2
    vf_loss_coeff = 0.25
    entropy_coeff = 0.02

    logits = agent(train_batch[WORLD_STATE], train_batch[AGENT_STATE], train_batch[OTHER_STATE])

    dist = Categorical(logits=logits)
    curr_action_dist = dist
    reduce_mean_valid = torch.mean

    prev_action_dist = Categorical(logits=train_batch[ACTION_DIST_INPUTS])
    prev_action_dist.logits = prev_action_dist.logits.squeeze()

    logp_1 = curr_action_dist.logits[range(dist.probs.shape[0]), train_batch[ACTIONS].long()]
    logp_2 = train_batch[ACTION_LOGP].squeeze()
    logp_ratio = torch.exp(logp_1 - logp_2)
    action_kl = kl.kl_divergence(prev_action_dist, curr_action_dist).mean()
    mean_kl_loss = reduce_mean_valid(action_kl)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES] * torch.clamp(logp_ratio, 1 - clip_param, 1 + clip_param))
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)

    # Compute a value function loss.
    if use_critic:
        prev_value_fn_out = train_batch[VF_PREDS]
        value_fn_out = agent.value_function()
        vf_loss1 = torch.pow(value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_clipped = prev_value_fn_out + torch.clamp(value_fn_out - prev_value_fn_out, -vf_clip_param, vf_clip_param)
        vf_loss2 = torch.pow(vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_loss = torch.max(vf_loss1, vf_loss2)
        mean_vf_loss = reduce_mean_valid(vf_loss)
    else:
        vf_loss = mean_vf_loss = 0.0
        ego_vf_loss = mean_ego_vf_loss = 0
    total_loss = reduce_mean_valid(-surrogate_loss + kl_coeff * action_kl + vf_loss_coeff * vf_loss -
                                   entropy_coeff * curr_entropy)

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    tower_stats = {}
    tower_stats["total_loss"] = total_loss.item()
    tower_stats["mean_policy_loss"] = mean_policy_loss.item()
    tower_stats["mean_vf_loss"] = mean_vf_loss.item()
    tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], agent.value_function()).item()
    tower_stats["mean_entropy"] = mean_entropy.item()
    tower_stats["mean_kl_loss"] = mean_kl_loss.item()

    return total_loss, tower_stats


def explained_variance(y, pred):
    y_var = torch.var(y, dim=[0])
    diff_var = torch.var(y - pred, dim=[0])
    min_ = torch.tensor([-1.0]).to(pred.device)
    return torch.max(min_, 1 - (diff_var / y_var))[0]
