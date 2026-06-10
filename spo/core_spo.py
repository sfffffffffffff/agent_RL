# Copyright 2025
#
# Segment Policy Optimization (SPO-chain) for multi-turn agent RL.
# Each environment turn is treated as one segment; segment advantages are
# estimated via Monte Carlo returns without a critic model.

from collections import defaultdict

from typing import Optional

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


def _compute_segment_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Backward discounted returns along one trajectory (SPO-chain MC estimation)."""
    traj_returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        traj_returns[t] = running_return
    return traj_returns


def _apply_group_baseline(
    segment_returns: np.ndarray,
    uid: np.ndarray,
    baseline_mode: str,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Normalize segment MC returns within each prompt group (uid)."""
    if baseline_mode == "none":
        return segment_returns

    id2scores = defaultdict(list)
    for i, group_id in enumerate(uid):
        id2scores[str(group_id)].append(segment_returns[i])

    id2mean = {}
    id2std = {}
    for group_id, scores in id2scores.items():
        if len(scores) == 1:
            id2mean[group_id] = 0.0
            id2std[group_id] = 1.0
        else:
            id2mean[group_id] = float(np.mean(scores))
            id2std[group_id] = float(np.std(scores))

    advantages = np.zeros_like(segment_returns, dtype=np.float32)
    for i, group_id in enumerate(uid):
        gid = str(group_id)
        if baseline_mode == "group_mean":
            advantages[i] = segment_returns[i] - id2mean[gid]
        elif baseline_mode == "group_std_norm":
            advantages[i] = (segment_returns[i] - id2mean[gid]) / (id2std[gid] + epsilon)
        else:
            raise ValueError(f"Unsupported baseline_mode: {baseline_mode}")
    return advantages


def compute_segment_mc_advantage_return(
    response_mask: torch.Tensor,
    traj_uid: np.ndarray,
    turn_step: np.ndarray,
    step_rewards: torch.Tensor,
    gamma: float = 0.95,
    uid: Optional[np.ndarray] = None,
    baseline_mode: str = "group_std_norm",
    epsilon: float = 1e-6,
):
    """Turn-level Segment PO (SPO-chain) advantage for multi-turn agent RL.

    Each batch row is one environment step (one segment / turn). MC returns are
    computed backward within each trajectory; optional group baseline uses ``uid``
    (same prompt, multiple trajectories when env.rollout.n > 1).

    Advantages are broadcast to all response tokens of the segment, then whitened.

    Args:
        response_mask: (bs, response_length)
        traj_uid: (bs,) trajectory ids
        turn_step: (bs,) step index within trajectory
        step_rewards: (bs,) scalar reward after each turn
        gamma: discount factor
        uid: (bs,) prompt group ids for baseline normalization
        baseline_mode: "none", "group_mean", or "group_std_norm"
        epsilon: numerical stability for std normalization

    Returns:
        advantages, returns: same shape as response_mask
    """
    with torch.no_grad():
        device = response_mask.device
        bs = response_mask.shape[0]
        rewards_np = step_rewards.detach().cpu().numpy().astype(np.float32)

        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        for i in range(bs):
            uid_to_indices[str(traj_uid[i])].append(i)

        segment_returns = np.zeros(bs, dtype=np.float32)
        for _uid, indices in uid_to_indices.items():
            indices = sorted(indices, key=lambda ii: int(turn_step[ii]))
            traj_rewards = np.array([rewards_np[ii] for ii in indices], dtype=np.float32)
            traj_returns = _compute_segment_returns(traj_rewards, gamma)
            for ti, ii in enumerate(indices):
                segment_returns[ii] = traj_returns[ti]

        if uid is None:
            uid = np.array([str(i) for i in range(bs)], dtype=object)
        segment_adv = _apply_group_baseline(segment_returns, uid, baseline_mode, epsilon)

        advantages = torch.zeros_like(response_mask)
        returns = torch.zeros_like(response_mask)
        for i in range(bs):
            advantages[i] = segment_adv[i] * response_mask[i]
            returns[i] = segment_returns[i] * response_mask[i]

        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns
