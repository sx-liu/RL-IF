"""
RLHF trainers for KL-penalized PPO, RLIF filtering, and clipped PPO.

Three trainer classes:
  - RLHFKLTrainer:      KL divergence as explicit loss term, no clipping.
  - RLHFTracInTrainer:  Same as RLHFKLTrainer + TracIn gradient-alignment filtering.
  - RLHFClippedTrainer: Clipped PPO, no KL penalty anywhere.

All three extend LayerwisePPOTrainer from train/trainer.py.

Key design decision (per user direction): KL is applied as an EXPLICIT LOSS TERM
in compute_ppo_loss, not via reward shaping. The GAE advantages are computed from
pure reward model scores only.
"""

import copy
import logging
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch import Tensor

from train.trainer import LayerwisePPOTrainer, compute_gae, INVALID_LOGPROB
from train.tracin import RLHFGhostTracIn
from train.rlif import RLHFGhostRLIF, compute_validation_gradient

logger = logging.getLogger(__name__)


def _select_by_score(scores: Tensor, filter_frac: float) -> Tensor:
    """
    Drop the bottom filter_frac fraction of all samples by influence score.

    filter_frac=0.0 → keep all
    filter_frac=0.1 → drop the 10% lowest-scoring samples
    filter_frac=1.0 → drop everything

    Returns a 1-D LongTensor of local indices into the mini-batch.
    """
    n = scores.numel()
    if filter_frac == 0.0:
        return torch.arange(n, device=scores.device)

    n_drop = min(n, round(n * filter_frac))
    if n_drop == 0:
        return torch.arange(n, device=scores.device)

    _, order = scores.sort()          # ascending: lowest scores first
    kept = order[n_drop:]             # drop the n_drop lowest
    return kept.sort().values


class RLHFKLTrainer(LayerwisePPOTrainer):
    """
    KL-penalized RLHF trainer (Task 1).

    Differences from LayerwisePPOTrainer:
    1. KL is applied as an explicit loss term: total_loss = pg_loss + vf_coef * vf_loss + kl_coef * kl_loss
    2. No PPO clipping on the policy ratio.
    3. Rewards for GAE are pure reward model scores (no KL reward shaping).

    All other behavior (RLIF strategy selection, early stopping, eval, etc.) is inherited.
    """

    @torch.no_grad()
    def _generate_rollout_data(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
    ) -> Dict[str, Any]:
        """
        Generate rollouts using pure reward model scores (no KL reward shaping).

        Override of LayerwisePPOTrainer._generate_rollout_data: removes the
        KL penalty subtraction from rewards before GAE. KL will be applied as
        an explicit loss term in compute_ppo_loss instead.
        """
        # Generate responses
        response_ids, response_mask, response_texts = self.generate_rollouts(query_ids, query_mask)

        # Compute rewards
        query_texts = self.tokenizer.batch_decode(query_ids, skip_special_tokens=True)
        raw_rewards = self.reward_model.compute_rewards(query_texts, response_texts)

        # Compute logprobs and values
        self.model.eval()
        old_logprobs, _, old_values = self.batched_forward_pass(
            query_ids, response_ids, query_mask, response_mask,
            batch_size=self.mini_batch_size,
        )

        # Compute reference logprobs (for KL loss at training time)
        ref_logprobs = self.batched_ref_forward_pass(
            query_ids, response_ids, query_mask, response_mask,
            batch_size=self.mini_batch_size,
        )

        # Mask out invalid logprobs at padded positions (TRL 0.26.1)
        padding_mask = (response_mask == 0)
        old_logprobs = torch.masked_fill(old_logprobs, padding_mask, INVALID_LOGPROB)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

        # Values: use padding_mask_p1 (one extra valid position for bootstrapping)
        response_len = response_ids.size(1)
        seq_lens = response_mask.sum(dim=1) - 1
        seq_lens_p1 = (seq_lens + 1).clamp(max=response_len - 1)
        response_idxs = torch.arange(response_len, device=response_mask.device).unsqueeze(0)
        padding_mask_p1 = response_idxs > seq_lens_p1.unsqueeze(1)
        old_values = torch.masked_fill(old_values, padding_mask_p1, 0)

        # Per-token reward tensor (reward only at last response token)
        rewards = torch.zeros_like(response_mask, dtype=torch.float, device=self.device)
        for i in range(len(response_texts)):
            nonzero_indices = response_mask[i].nonzero()
            if len(nonzero_indices) > 0:
                last_idx = nonzero_indices[-1].item()
                rewards[i, last_idx] = raw_rewards[i]

        # NOTE: NO KL reward shaping here. Pure rewards → GAE.
        # KL penalty is applied as explicit loss term in compute_ppo_loss.

        # Compute KL penalty for logging purposes only
        kl_penalty = self._compute_kl(old_logprobs, ref_logprobs)

        # GAE on pure rewards
        advantages, returns = compute_gae(
            rewards, old_values, response_mask.float(),
            self.gamma, self.gae_lambda,
        )

        return {
            "response_ids": response_ids,
            "response_mask": response_mask,
            "response_texts": response_texts,
            "raw_rewards": raw_rewards,
            "old_logprobs": old_logprobs,
            "old_values": old_values,
            "ref_logprobs": ref_logprobs,
            "kl_penalty": kl_penalty,
            "advantages": advantages,
            "returns": returns,
        }

    def _run_ppo_epochs(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        rollout_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Override to pass ref_logprobs through mini-batch training steps.

        Extracts ref_logprobs from rollout_data and threads them into
        compute_ppo_loss via training_step_with_kl.
        """
        response_ids = rollout_data["response_ids"]
        response_mask = rollout_data["response_mask"]
        old_logprobs = rollout_data["old_logprobs"]
        old_values = rollout_data["old_values"]
        advantages = rollout_data["advantages"]
        returns = rollout_data["returns"]
        ref_logprobs = rollout_data["ref_logprobs"]

        batch_size = query_ids.shape[0]
        all_mini_batch_stats = []

        self.model.train()

        early_stopped = False
        for ppo_epoch in range(self.ppo_epochs):
            if early_stopped:
                break

            perm = torch.randperm(batch_size, device=query_ids.device)

            for mb_start in range(0, batch_size, self.mini_batch_size):
                mb_end = min(mb_start + self.mini_batch_size, batch_size)
                mb_inds = perm[mb_start:mb_end]

                mb_stats = self.training_step_with_kl(
                    query_ids[mb_inds],
                    response_ids[mb_inds],
                    query_mask[mb_inds],
                    response_mask[mb_inds],
                    old_logprobs[mb_inds].detach(),
                    advantages[mb_inds].detach(),
                    returns[mb_inds].detach(),
                    old_values[mb_inds].detach(),
                    ref_logprobs[mb_inds].detach(),
                )
                all_mini_batch_stats.append(mb_stats)

                # Early stopping check
                policy_kl = mb_stats.get("policy/policykl", 0.0)
                if self._early_stop(policy_kl):
                    early_stopped = True
                    break

        # Aggregate stats across all mini-batches
        stats = all_mini_batch_stats[-1].copy()
        if len(all_mini_batch_stats) > 1:
            all_keys = set()
            for s in all_mini_batch_stats:
                all_keys.update(s.keys())
            for key in all_keys:
                values = [s[key] for s in all_mini_batch_stats if key in s]
                if not values or not isinstance(values[0], (int, float)):
                    continue
                stats[key] = float(np.mean(values))
        # timing keys: sum across mini-batches to get per-rollout totals
        for tkey in ("timing/forward_s", "timing/optimize_s"):
            vals = [s[tkey] for s in all_mini_batch_stats if tkey in s]
            if vals:
                stats[tkey] = float(sum(vals))

        stats["ppo/early_stopped"] = 1.0 if early_stopped else 0.0
        return stats

    def training_step_with_kl(
        self,
        query_ids: Tensor,
        response_ids: Tensor,
        query_mask: Tensor,
        response_mask: Tensor,
        old_logprobs: Tensor,
        advantages: Tensor,
        returns: Tensor,
        old_values: Tensor,
        ref_logprobs: Tensor,
    ) -> Dict[str, float]:
        """
        Training step that passes ref_logprobs into compute_ppo_loss.

        Mirrors LayerwisePPOTrainer.training_step but threads ref_logprobs
        through the compute_loss_fn and filter_batch_fn closures.
        """
        batch_size = query_ids.shape[0]
        lr = self.optimizer.param_groups[0]["lr"]

        strategy = self._get_effective_strategy(batch_size)
        self.optimizer.zero_grad()

        def compute_loss_fn():
            return self.compute_ppo_loss(
                query_ids, response_ids, query_mask, response_mask,
                old_logprobs, advantages, returns, old_values, ref_logprobs,
            )

        def filter_batch_fn(selected_indices):
            def filtered_compute_loss_fn():
                return self.compute_ppo_loss(
                    query_ids[selected_indices],
                    response_ids[selected_indices],
                    query_mask[selected_indices],
                    response_mask[selected_indices],
                    old_logprobs[selected_indices],
                    advantages[selected_indices],
                    returns[selected_indices],
                    old_values[selected_indices],
                    ref_logprobs[selected_indices],
                )
            return filtered_compute_loss_fn

        # Labels for token-based gradient scaling (same as parent)
        response_len = response_mask.size(1)
        seq_lens = response_mask.sum(dim=1) - 1
        seq_lens_p1 = (seq_lens + 1).clamp(max=response_len - 1)
        response_idxs = torch.arange(response_len, device=response_mask.device).unsqueeze(0)
        value_mask = (response_idxs <= seq_lens_p1.unsqueeze(1))
        labels = torch.where(
            value_mask,
            torch.zeros_like(response_mask),
            torch.full_like(response_mask, -100),
        )

        t_fwd = time.time()
        loss, stats = strategy.execute_training_step(
            model=self.model,
            batch_size=batch_size,
            compute_loss_fn=compute_loss_fn,
            lr=lr,
            filter_batch_fn=filter_batch_fn,
            labels=labels,
        )
        stats["timing/forward_s"] = time.time() - t_fwd

        # Record curation statistics
        if self.method != "NA" and self.grad_hook is not None:
            sel_state = getattr(self.grad_hook, 'selection_state', None)
            if sel_state is not None:
                if self.method == "Layerwise":
                    if hasattr(sel_state, '_layer_selections') and sel_state._layer_selections:
                        n_selected_list = [n for _, n in sel_state._layer_selections]
                        stats["selection/avg_selected"] = sum(n_selected_list) / len(n_selected_list)
                elif self.method == "Subset":
                    n_selected = getattr(sel_state, 'num_selected', None)
                    if n_selected is not None:
                        stats["selection/n_selected"] = n_selected

        # Gradient clipping
        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.args.max_grad_norm,
            )

        stats["lr"] = lr
        t_opt = time.time()
        self.optimizer.step()
        stats["timing/optimize_s"] = time.time() - t_opt
        self.optimizer.zero_grad()
        return stats

    def compute_ppo_loss(
        self,
        query_ids: Tensor,
        response_ids: Tensor,
        query_mask: Tensor,
        response_mask: Tensor,
        old_logprobs: Tensor,
        advantages: Tensor,
        returns: Tensor,
        old_values: Tensor,
        ref_logprobs: Tensor = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        PPO loss with explicit KL divergence term and NO policy ratio clipping.

        total_loss = pg_loss + vf_coef * vf_loss + kl_coef * kl_loss

        Where:
          pg_loss  = E[-A(s,a) * π_new(a|s) / π_old(a|s)]  (unclipped)
          vf_loss  = 0.5 * E[(V(s) - R)^2]  (clipped value)
          kl_loss  = E[log π_new(a|s) - log π_ref(a|s)]  (explicit KL term)

        Args:
            query_ids, response_ids: Token IDs [batch, len]
            query_mask, response_mask: Attention masks
            old_logprobs: Log probs from rollout [batch, response_len]
            advantages: GAE advantages from pure rewards [batch, response_len]
            returns: GAE returns [batch, response_len]
            old_values: Value estimates from rollout [batch, response_len]
            ref_logprobs: Reference policy log probs [batch, response_len]

        Returns:
            (total_loss, stats_dict)
        """
        batch_size = query_ids.shape[0]
        query_len = query_ids.shape[1]
        response_len = response_ids.shape[1]

        # Concatenate query and response
        input_ids = torch.cat([query_ids, response_ids], dim=1)
        attention_mask = torch.cat([query_mask, response_mask], dim=1)

        # Position IDs and masked input (TRL 0.26.1 approach)
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        input_ids_masked = torch.masked_fill(input_ids, ~attention_mask.bool(), 0)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids_masked,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )

        logits, values_full = self._extract_model_outputs(outputs)
        if values_full is None:
            values_full = torch.zeros(batch_size, input_ids.shape[1], device=self.device)

        # Extract response-token slices
        start_idx, end_idx = self._get_response_slice_indices(query_len, response_len)
        values = values_full[:, start_idx:end_idx]
        logits_for_probs = logits[:, start_idx:end_idx, :]

        if logits_for_probs.shape[1] != response_len:
            raise RuntimeError(
                f"PPO loss logits shape mismatch: expected {response_len}, "
                f"got {logits_for_probs.shape[1]}"
            )

        # New log probs
        temperature = getattr(self.args, 'temperature', 1.0)
        new_logprobs = self._compute_token_logprobs(
            logits_for_probs, response_ids, temperature=temperature
        )

        # Mask padded positions (TRL 0.26.1)
        padding_mask = (response_mask == 0)
        new_logprobs = torch.masked_fill(new_logprobs, padding_mask, INVALID_LOGPROB)

        # Policy gradient loss — UNCLIPPED (no clip_range)
        logprob_diff = new_logprobs - old_logprobs
        ratio = torch.exp(logprob_diff)
        pg_loss = (-advantages * ratio * response_mask.float()).sum() / response_mask.sum().clamp(min=1)

        # Value loss with clipping (same as parent)
        seq_lens = response_mask.sum(dim=1) - 1
        seq_lens_p1 = (seq_lens + 1).clamp(max=response_len - 1)
        response_idxs = torch.arange(response_len, device=response_mask.device).unsqueeze(0)
        value_mask = (response_idxs <= seq_lens_p1.unsqueeze(1)).long()

        values = values * value_mask
        values_clipped = old_values + torch.clamp(
            values - old_values,
            -self.cliprange_value,
            self.cliprange_value,
        )
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2)
        vf_loss = (vf_loss * value_mask).sum() / value_mask.sum().clamp(min=1)

        # Explicit KL loss term: KL(π_new || π_ref) = E[log π_new - log π_ref]
        kl_loss = torch.zeros(1, device=self.device)
        if ref_logprobs is not None:
            valid_new_logprobs = torch.where(padding_mask, torch.zeros_like(new_logprobs), new_logprobs)
            valid_ref_logprobs = torch.where(padding_mask, torch.zeros_like(ref_logprobs), ref_logprobs)
            kl_per_token = valid_new_logprobs - valid_ref_logprobs
            kl_loss = (kl_per_token * response_mask.float()).sum() / response_mask.sum().clamp(min=1)

        # Total loss
        total_loss = pg_loss + self.vf_coef * vf_loss + self.kl_ctl.value * kl_loss

        # Compute stats
        with torch.no_grad():
            stats = self._compute_ppo_stats(
                ratio, values, old_values, new_logprobs, old_logprobs, response_mask
            )

        # Skip batch if ratio exceeds threshold
        ratio_threshold = getattr(self.args, 'ratio_threshold', 10.0)
        if stats["policy/ratio"] > ratio_threshold:
            total_loss = total_loss * 0.0
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0

        stats["loss/policy"] = pg_loss.item()
        stats["loss/value"] = vf_loss.item()
        stats["loss/kl"] = kl_loss.item() if torch.is_tensor(kl_loss) else float(kl_loss)
        stats["loss/kl_coef"] = self.kl_ctl.value
        stats["loss/total"] = total_loss.item()

        return total_loss, stats


class RLHFTracInTrainer(RLHFKLTrainer):
    """
    TracIn gradient-alignment filtering trainer for RLHF.

    Extends RLHFKLTrainer with inline ghost-gradient TracIn influence scoring.
    For each PPO mini-batch:
      1. Forward pass → per-sample scoring losses (pg + kl, no value loss)
      2. Backward on sum → hooks capture [N,T,D] activations; param.grad populated
      3. u_flat = full mini-batch gradient (self-reference validation direction)
      4. scores_i = <∇_θ loss_i, u_flat>  (ghost inner product, one pass)
      5. Keep only positive-influence samples; skip update if none survive
      6. Compute full training loss on selected subset and update

    NOTE: This uses TracIn (single-checkpoint dot product), not full RLIF
    (multi-step trajectory influence). Kept as a baseline alongside RLIF.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracin = RLHFGhostTracIn(self.model)
        logger.info("RLHFTracInTrainer: TracIn inline filtering enabled")

    def _compute_per_sample_scoring_losses(
        self,
        query_ids: Tensor,
        response_ids: Tensor,
        query_mask: Tensor,
        response_mask: Tensor,
        old_logprobs: Tensor,
        advantages: Tensor,
        ref_logprobs: Tensor,
    ) -> Tensor:
        """
        Compute per-sample KL-penalized policy gradient losses for RLIF scoring.

        Omits the value loss (consistent with reference skipping value_net for scoring).
        Returns a [N] tensor — .sum().backward() triggers ghost gradient hooks.

        Loss per sample i:
            pg_i  = -mean_t(A_t * ratio_t * mask_t)
            kl_i  = mean_t((log π_new - log π_ref) * mask_t)
            loss_i = pg_i + kl_coef * kl_i
        """
        batch_size = query_ids.shape[0]
        query_len = query_ids.shape[1]
        response_len = response_ids.shape[1]

        input_ids = torch.cat([query_ids, response_ids], dim=1)
        attention_mask = torch.cat([query_mask, response_mask], dim=1)
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        input_ids_masked = torch.masked_fill(input_ids, ~attention_mask.bool(), 0)

        outputs = self.model(
            input_ids=input_ids_masked,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )

        logits, _ = self._extract_model_outputs(outputs)

        start_idx, end_idx = self._get_response_slice_indices(query_len, response_len)
        logits_for_probs = logits[:, start_idx:end_idx, :]

        temperature = getattr(self.args, 'temperature', 1.0)
        new_logprobs = self._compute_token_logprobs(
            logits_for_probs, response_ids, temperature=temperature
        )

        padding_mask = (response_mask == 0)
        new_logprobs = torch.masked_fill(new_logprobs, padding_mask, INVALID_LOGPROB)

        mask = response_mask.float()
        n_valid = mask.sum(dim=1).clamp(min=1)

        ratio = torch.exp(new_logprobs - old_logprobs)
        pg_per_sample = -(advantages * ratio * mask).sum(dim=1) / n_valid

        valid_new = torch.where(padding_mask, torch.zeros_like(new_logprobs), new_logprobs)
        valid_ref = torch.where(padding_mask, torch.zeros_like(ref_logprobs), ref_logprobs)
        kl_per_sample = ((valid_new - valid_ref) * mask).sum(dim=1) / n_valid

        per_sample_losses = pg_per_sample + self.kl_ctl.value * kl_per_sample
        return per_sample_losses

    def _run_ppo_epochs(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        rollout_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        PPO epochs with inline RLIF influence-based sample filtering.

        Overrides RLHFKLTrainer._run_ppo_epochs to add per-mini-batch filtering.
        """
        response_ids = rollout_data["response_ids"]
        response_mask = rollout_data["response_mask"]
        old_logprobs = rollout_data["old_logprobs"]
        old_values = rollout_data["old_values"]
        advantages = rollout_data["advantages"]
        returns = rollout_data["returns"]
        ref_logprobs = rollout_data["ref_logprobs"]

        batch_size = query_ids.shape[0]
        all_mini_batch_stats: List[Dict] = []

        self.model.train()
        early_stopped = False

        for ppo_epoch in range(self.ppo_epochs):
            if early_stopped:
                break

            perm = torch.randperm(batch_size, device=query_ids.device)

            for mb_start in range(0, batch_size, self.mini_batch_size):
                mb_end = min(mb_start + self.mini_batch_size, batch_size)
                mb_inds = perm[mb_start:mb_end]

                # --- Phase 1: Compute per-sample scoring losses ---
                self.optimizer.zero_grad()
                per_sample_losses = self._compute_per_sample_scoring_losses(
                    query_ids[mb_inds],
                    response_ids[mb_inds],
                    query_mask[mb_inds],
                    response_mask[mb_inds],
                    old_logprobs[mb_inds].detach(),
                    advantages[mb_inds].detach(),
                    ref_logprobs[mb_inds].detach(),
                )

                # Backward on sum: hooks capture [N,T,D] activations + param.grad
                per_sample_losses.sum().backward()
                u_flat = self.tracin.get_u_flat()

                # --- Phase 2: Compute influence scores ---
                scores = self.tracin.compute_scores(u_flat)  # [N]

                # --- Phase 3: Filter samples by score rank ---
                n_total = len(mb_inds)
                selected_local = _select_by_score(scores, getattr(self.args, "filter_frac", 0.5))
                n_selected = len(selected_local)

                if n_selected == 0:
                    logger.debug(
                        "TracIn: all %d samples filtered out (epoch %d) — skipping update",
                        n_total, ppo_epoch,
                    )
                    self.optimizer.zero_grad()
                    stats = {"selection/n_selected": 0, "selection/frac_kept": 0.0}
                    all_mini_batch_stats.append(stats)
                    continue

                selected_global = mb_inds[selected_local]

                # --- Phase 4: Compute full training loss on selected subset ---
                self.optimizer.zero_grad()
                total_loss, stats = self.compute_ppo_loss(
                    query_ids[selected_global],
                    response_ids[selected_global],
                    query_mask[selected_global],
                    response_mask[selected_global],
                    old_logprobs[selected_global].detach(),
                    advantages[selected_global].detach(),
                    returns[selected_global].detach(),
                    old_values[selected_global].detach(),
                    ref_logprobs[selected_global].detach(),
                )
                total_loss.backward()

                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

                stats["selection/n_selected"] = float(n_selected)
                stats["selection/frac_kept"] = float(n_selected) / float(n_total)
                stats["lr"] = self.optimizer.param_groups[0]["lr"]
                all_mini_batch_stats.append(stats)

                policy_kl = stats.get("policy/policykl", 0.0)
                if self._early_stop(policy_kl):
                    early_stopped = True
                    break

        if not all_mini_batch_stats:
            return {"ppo/early_stopped": 0.0, "selection/frac_kept": 0.0}

        # Aggregate across mini-batches (mean)
        stats = all_mini_batch_stats[-1].copy()
        if len(all_mini_batch_stats) > 1:
            all_keys: set = set()
            for s in all_mini_batch_stats:
                all_keys.update(s.keys())
            for key in all_keys:
                values = [s[key] for s in all_mini_batch_stats if key in s]
                if not values or not isinstance(values[0], (int, float)):
                    continue
                stats[key] = float(np.mean(values))

        stats["ppo/early_stopped"] = 1.0 if early_stopped else 0.0
        return stats


class RLHFIIFTracInTrainer(RLHFTracInTrainer):
    """
    IIF-TracIn: rollout-level influence filtering before any parameter updates.

    Unlike per-step TracIn (which scores each mini-batch inline during PPO epochs),
    IIF-TracIn scores ALL rollout samples at θ_0 in a single pass, filters globally,
    then runs standard KL-PPO training on the filtered subset.

    Algorithm:
      1. Compute u = ∇_θ Σ_i L_score(x_i) over full rollout (mini-batched, hooks off)
      2. Compute score_i = <∇_θ L_score(x_i), u>  for all i  (ghost trick, hooks on)
      3. Filter: drop bottom filter_frac globally across all N samples
      4. Train: run standard PPO epochs on filtered subset (RLHFKLTrainer logic)
    """

    def _run_ppo_epochs(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        rollout_data: Dict[str, Any],
    ) -> Dict[str, float]:
        batch_size    = query_ids.shape[0]
        response_ids  = rollout_data["response_ids"]
        response_mask = rollout_data["response_mask"]
        old_logprobs  = rollout_data["old_logprobs"]
        advantages    = rollout_data["advantages"]
        ref_logprobs  = rollout_data["ref_logprobs"]

        # ── Phase 1: compute u over full rollout (hooks disabled) ────────────
        t_influence_start = time.time()
        self.tracin._hooks_enabled = False
        self.model.train()
        self.model.zero_grad()
        try:
            for mb_start in range(0, batch_size, self.mini_batch_size):
                mb = slice(mb_start, min(mb_start + self.mini_batch_size, batch_size))
                losses = self._compute_per_sample_scoring_losses(
                    query_ids[mb], response_ids[mb],
                    query_mask[mb], response_mask[mb],
                    old_logprobs[mb].detach(),
                    advantages[mb].detach(),
                    ref_logprobs[mb].detach(),
                )
                losses.sum().backward()          # accumulate into .grad across chunks
            u = self.tracin.get_u_flat()         # u = ∇_θ Σ_i loss_i  (CPU, flat)
        finally:
            self.tracin._hooks_enabled = True
            self.model.zero_grad()

        logger.debug("IIFTracIn: u norm=%.4f", u.norm().item())

        # ── Phase 2: score all samples (hooks enabled, mini-batched) ────────
        all_scores: List[Tensor] = []
        for mb_start in range(0, batch_size, self.mini_batch_size):
            mb = slice(mb_start, min(mb_start + self.mini_batch_size, batch_size))
            self.optimizer.zero_grad()
            losses = self._compute_per_sample_scoring_losses(
                query_ids[mb], response_ids[mb],
                query_mask[mb], response_mask[mb],
                old_logprobs[mb].detach(),
                advantages[mb].detach(),
                ref_logprobs[mb].detach(),
            )
            losses.sum().backward()
            all_scores.append(self.tracin.compute_scores(u).cpu())
            self.optimizer.zero_grad()

        scores = torch.cat(all_scores)           # [N]

        # ── Phase 3: global filter ───────────────────────────────────────────
        filter_frac = getattr(self.args, "filter_frac", 0.1)
        selected    = _select_by_score(scores, filter_frac).to(query_ids.device)
        frac_kept   = selected.numel() / batch_size
        t_influence = time.time() - t_influence_start

        logger.debug(
            "IIFTracIn: kept %d/%d samples (frac=%.3f)",
            selected.numel(), batch_size, frac_kept,
        )

        if selected.numel() == 0:
            return {"ppo/early_stopped": 0.0, "selection/frac_kept": 0.0}

        # ── Phase 4: standard KL-PPO on filtered subset ──────────────────────
        sel_idx = selected.cpu().tolist()
        sel_data = {}
        for k, v in rollout_data.items():
            if isinstance(v, Tensor):
                sel_data[k] = v[selected]
            elif isinstance(v, list):
                sel_data[k] = [v[i] for i in sel_idx]
            else:
                sel_data[k] = v
        stats = RLHFKLTrainer._run_ppo_epochs(
            self, query_ids[selected], query_mask[selected], sel_data
        )
        stats["selection/frac_kept"]  = frac_kept
        stats["selection/n_selected"] = float(selected.numel())
        stats["timing/influence_s"]   = t_influence
        return stats


class RLHFClippedTrainer(LayerwisePPOTrainer):
    """
    Clipped PPO baseline (Task 3, Baseline 2).

    Uses standard clipped PPO objective with NO KL penalty anywhere:
    - No KL reward shaping in _generate_rollout_data
    - No KL loss term in compute_ppo_loss (uses parent's clipped PPO loss)
    - Set init_kl_coef=0.0 in args to ensure kl_ctl.value = 0

    GAE advantages are computed from pure reward model scores.
    """

    @torch.no_grad()
    def _generate_rollout_data(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
    ) -> Dict[str, Any]:
        """
        Generate rollouts with pure reward model scores (no KL reward shaping).

        Override of LayerwisePPOTrainer._generate_rollout_data: removes the
        KL penalty subtraction from rewards. KL is not applied anywhere in
        this trainer.
        """
        # Generate responses
        response_ids, response_mask, response_texts = self.generate_rollouts(query_ids, query_mask)

        # Compute rewards
        query_texts = self.tokenizer.batch_decode(query_ids, skip_special_tokens=True)
        raw_rewards = self.reward_model.compute_rewards(query_texts, response_texts)

        # Compute logprobs and values
        self.model.eval()
        old_logprobs, _, old_values = self.batched_forward_pass(
            query_ids, response_ids, query_mask, response_mask,
            batch_size=self.mini_batch_size,
        )

        # Compute reference logprobs for logging only
        ref_logprobs = self.batched_ref_forward_pass(
            query_ids, response_ids, query_mask, response_mask,
            batch_size=self.mini_batch_size,
        )

        # Mask out invalid logprobs at padded positions
        padding_mask = (response_mask == 0)
        old_logprobs = torch.masked_fill(old_logprobs, padding_mask, INVALID_LOGPROB)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

        # Values: padding_mask_p1 (one extra valid position)
        response_len = response_ids.size(1)
        seq_lens = response_mask.sum(dim=1) - 1
        seq_lens_p1 = (seq_lens + 1).clamp(max=response_len - 1)
        response_idxs = torch.arange(response_len, device=response_mask.device).unsqueeze(0)
        padding_mask_p1 = response_idxs > seq_lens_p1.unsqueeze(1)
        old_values = torch.masked_fill(old_values, padding_mask_p1, 0)

        # Per-token reward tensor (only at last response token)
        rewards = torch.zeros_like(response_mask, dtype=torch.float, device=self.device)
        for i in range(len(response_texts)):
            nonzero_indices = response_mask[i].nonzero()
            if len(nonzero_indices) > 0:
                last_idx = nonzero_indices[-1].item()
                rewards[i, last_idx] = raw_rewards[i]

        # NO KL reward shaping. Pure rewards → GAE.
        kl_penalty = self._compute_kl(old_logprobs, ref_logprobs)  # for logging

        advantages, returns = compute_gae(
            rewards, old_values, response_mask.float(),
            self.gamma, self.gae_lambda,
        )

        return {
            "response_ids": response_ids,
            "response_mask": response_mask,
            "response_texts": response_texts,
            "raw_rewards": raw_rewards,
            "old_logprobs": old_logprobs,
            "old_values": old_values,
            "ref_logprobs": ref_logprobs,
            "kl_penalty": kl_penalty,
            "advantages": advantages,
            "returns": returns,
        }



class RLHFRLIFTrainer(RLHFTracInTrainer):
    """
    RLIF non-local data filtering trainer for RLHF (Task 2).

    Implements the full 4-phase trajectory influence algorithm:

      Phase 1 — Auxiliary training:
        Run ppo_epochs × n_mini_batches standard KL PPO steps.
        Capture (xs, gs) ghost gradient snapshots on CPU after each step.

      Phase 2 — RLIF score computation:
        u_T is recovered from the last Phase 1 snapshot (mean gradient at θ_{T-1})
        — no separate forward+backward pass needed.
        Walk backwards: each snapshot is streamed to GPU one at a time.
        At each step t (reversed):
            score_{t,i} = (lr/N) * <∇_{θ_t} loss_i, u_t>   (snapshot on GPU)
            u_{t-1}     = u_t - lr * F_t * u_t              (FIM from snapshot on GPU)

      Phase 3 — Filter:
        For each step's mini-batch, discard the bottom filter_frac fraction
        by score rank.

      Phase 4 — Replay:
        Restore θ_0 and optimizer state saved before Phase 1.
        Retrain through the trajectory using only the filtered mini-batches.

    The FIM approximation (H ≈ F_emp) avoids second-order differentiation.
    Phase 1 snapshots (xs, gs) are stored on CPU; Phase 2 streams them to GPU
    one step at a time, eliminating both model-checkpoint saves and the
    expensive full-batch _compute_starting_u pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracin.close()
        self.rlif = RLHFGhostRLIF(self.model)
        # Multi-rollout horizon buffer (populated when rlif_horizon > steps_per_rollout)
        self._rlif_buffer: List[tuple] = []        # (query_ids_cpu, query_mask_cpu, data_cpu, grad_snapshots, mb_inds_list)
        self._rlif_buffer_steps: int   = 0
        self._rlif_initial_model: Optional[Dict] = None
        self._rlif_initial_opt:   Optional[Dict] = None
        self._rlif_phase1_time: float  = 0.0      # accumulated aux-training time across buffered rollouts
        logger.info("RLHFRLIFTrainer: non-local RLIF trajectory-influence filtering enabled")

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_policy_params(self) -> Dict:
        """Save trainable policy params (excl. v_head) to CPU — used for RLIF scoring."""
        return {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad and "v_head" not in name
        }

    def _save_all_trainable_params(self) -> Dict:
        """Save ALL trainable params (incl. v_head) to CPU — used for replay restore."""
        return {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def _restore_params(self, state: Dict) -> None:
        """Copy a CPU param snapshot back to GPU in-place."""
        param_dict = dict(self.model.named_parameters())
        for name, saved in state.items():
            if name in param_dict:
                param_dict[name].data.copy_(saved.to(param_dict[name].device))

    # ------------------------------------------------------------------
    # Phase 1: Auxiliary training
    # ------------------------------------------------------------------

    def _auxiliary_training(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        rollout_data: Dict[str, Any],
    ) -> Tuple[List[Dict], List[Tensor]]:
        """
        Standard KL PPO for ppo_epochs × n_mini_batches steps with no filtering.
        Captures (xs, gs) ghost gradient snapshots on CPU after each step for
        use in Phase 2 — no model-parameter checkpoints stored.

        Returns:
            grad_snapshots: list of {"xs": Dict, "gs": Dict} CPU tensors, length T
            mb_inds_list:   list of mini-batch global index tensors (CPU), length T
        """
        response_ids  = rollout_data["response_ids"]
        response_mask = rollout_data["response_mask"]
        old_logprobs  = rollout_data["old_logprobs"]
        old_values    = rollout_data["old_values"]
        advantages    = rollout_data["advantages"]
        returns       = rollout_data["returns"]
        ref_logprobs  = rollout_data["ref_logprobs"]

        batch_size     = query_ids.shape[0]
        grad_snapshots : List[Dict]   = []
        mb_inds_list   : List[Tensor] = []

        self.model.train()
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(batch_size, device=query_ids.device)
            for mb_start in range(0, batch_size, self.mini_batch_size):
                mb_end  = min(mb_start + self.mini_batch_size, batch_size)
                mb_inds = perm[mb_start:mb_end]

                mb_inds_list.append(mb_inds.cpu())

                # Standard KL PPO step — ghost hooks populate (xs, gs) at θ_t
                self.training_step_with_kl(
                    query_ids[mb_inds],
                    response_ids[mb_inds],
                    query_mask[mb_inds],
                    response_mask[mb_inds],
                    old_logprobs[mb_inds].detach(),
                    advantages[mb_inds].detach(),
                    returns[mb_inds].detach(),
                    old_values[mb_inds].detach(),
                    ref_logprobs[mb_inds].detach(),
                )

                # Contract (xs, gs) over tokens on GPU → [N, D_out, D_in] per layer.
                # ~6× smaller than raw (xs, gs); stays on GPU for zero-copy Phase 2.
                grad_snapshots.append(self.rlif._snapshot_compressed_and_clear_hooks())

        logger.debug("RLIF Phase 1 done: %d gradient snapshots captured", len(grad_snapshots))
        return grad_snapshots, mb_inds_list

    # ------------------------------------------------------------------
    # Phase 2: RLIF score computation
    # ------------------------------------------------------------------

    def _compute_starting_u(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        rollout_data: Dict[str, Any],
        u_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """
        u_T = ∇_θ L_scoring(θ_T) on a subset (or all) of the rollout batch.

        Args:
            u_indices: if given, compute u only on these global sample indices.
                       Used for multi-rollout 'last_rollout' scope.
                       None means use the full batch.
        """
        if u_indices is not None:
            idx = u_indices.to(query_ids.device)
            q_ids   = query_ids[idx]
            q_mask  = query_mask[idx]
            r_ids   = rollout_data["response_ids"][idx]
            r_mask  = rollout_data["response_mask"][idx]
            olp     = rollout_data["old_logprobs"][idx]
            adv     = rollout_data["advantages"][idx]
            ref_lp  = rollout_data["ref_logprobs"][idx]
        else:
            q_ids   = query_ids
            q_mask  = query_mask
            r_ids   = rollout_data["response_ids"]
            r_mask  = rollout_data["response_mask"]
            olp     = rollout_data["old_logprobs"]
            adv     = rollout_data["advantages"]
            ref_lp  = rollout_data["ref_logprobs"]

        self.model.eval()
        self.rlif._hooks_enabled = False
        try:
            self.model.zero_grad()
            all_losses = self._compute_per_sample_scoring_losses(
                q_ids, r_ids, q_mask, r_mask,
                olp.detach(), adv.detach(), ref_lp.detach(),
            )
            u = compute_validation_gradient(self.model, all_losses)
        finally:
            self.rlif._hooks_enabled = True
            self.model.zero_grad()
            self.model.train()

        return u

    def _compute_trajectory_scores(
        self,
        query_ids: Tensor,
        grad_snapshots: List[Dict],
        mb_inds_list: List[Tensor],
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Walk backwards through compressed gradient snapshots, computing per-sample RLIF scores.

        Snapshots are C[n, D_out, D_in] = Σ_t G[n,t]^T ⊗ X[n,t] tensors already on GPU
        (contracted in Phase 1 before any CPU transfer). u_T is the mean gradient from
        the last snapshot — no model forward/backward or PCIe transfers needed.

        At step t (reversed):
            score_{t,i} = (lr/N_t) * <∇_{θ_t} loss_i, u_t>   (snapshot on GPU)
            u_{t-1}     = u_t - lr * F_t * u_t                 (FIM from snapshot on GPU)

        Returns:
            step_scores: list of (mb_inds_cpu, scaled_scores_cpu) in FORWARD order.
        """
        T   = len(grad_snapshots)
        lr  = self.optimizer.param_groups[0]["lr"]
        dev = query_ids.device

        # u_T: mean gradient at θ_{T-1} from the last compressed snapshot (already on GPU).
        # No model forward/backward or PCIe transfer needed.
        u = self.rlif.compute_u_from_compressed(grad_snapshots[-1])   # [D], on GPU
        logger.info("RLIF Phase 2: T=%d steps, u_T norm=%.4f (compressed, GPU)", T, u.norm().item())

        step_scores: List[Optional[Tuple[Tensor, Tensor]]] = [None] * T

        for t in reversed(range(T)):
            c       = grad_snapshots[t]                                # already on GPU
            mb_inds = mb_inds_list[t].to(dev)
            N       = mb_inds.size(0)

            scores_t = self.rlif.compute_scores_from_compressed(c, u)   # [N], GPU
            step_scores[t] = (mb_inds.cpu(), ((lr / N) * scores_t).cpu())

            if t > 0:
                u = self.rlif.propagate_u_fim_from_compressed(c, scores_t, u, lr)

            if t % max(1, T // 5) == 0:
                logger.debug("  step %3d | u norm: %.4f", t, u.norm().item())

        return step_scores  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Phase 3: Filter
    # ------------------------------------------------------------------

    def _filter_trajectory(
        self,
        step_scores: List[Tuple[Tensor, Tensor]],
    ) -> Tuple[List[Tensor], float]:
        """
        Apply score-rank filtering to each step's mini-batch.

        Returns:
            filtered_mb_inds: per-step list of selected global indices (CPU)
            frac_kept: overall fraction of samples retained
        """
        filter_frac = getattr(self.args, "filter_frac", 0.5)
        filtered: List[Tensor] = []
        n_total = n_kept = 0

        for mb_inds, scores in step_scores:
            sel_local  = _select_by_score(scores, filter_frac)
            sel_global = mb_inds[sel_local]
            filtered.append(sel_global)
            n_total += mb_inds.size(0)
            n_kept  += sel_global.size(0)

        frac_kept = n_kept / max(n_total, 1)
        logger.info(
            "RLIF Phase 3: kept %d / %d samples (%.1f%%) across %d steps",
            n_kept, n_total, 100.0 * frac_kept, len(step_scores),
        )
        return filtered, frac_kept

    # ------------------------------------------------------------------
    # Phase 4: Replay
    # ------------------------------------------------------------------

    def _replay_filtered(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        filtered_mb_inds: List[Tensor],
        rollout_data: Dict[str, Any],
        frac_kept: float,
    ) -> Dict[str, float]:
        """
        Replay the trajectory using only the filtered mini-batches.
        Model and optimizer have already been restored to θ_0.
        Steps whose mini-batch is fully filtered out are skipped.
        """
        response_ids  = rollout_data["response_ids"]
        response_mask = rollout_data["response_mask"]
        old_logprobs  = rollout_data["old_logprobs"]
        old_values    = rollout_data["old_values"]
        advantages    = rollout_data["advantages"]
        returns       = rollout_data["returns"]
        ref_logprobs  = rollout_data["ref_logprobs"]

        all_stats: List[Dict] = []
        self.model.train()

        for sel_global in filtered_mb_inds:
            if sel_global.numel() == 0:
                continue

            sel = sel_global.to(query_ids.device)
            self.optimizer.zero_grad()
            total_loss, stats = self.compute_ppo_loss(
                query_ids[sel],
                response_ids[sel],
                query_mask[sel],
                response_mask[sel],
                old_logprobs[sel].detach(),
                advantages[sel].detach(),
                returns[sel].detach(),
                old_values[sel].detach(),
                ref_logprobs[sel].detach(),
            )
            total_loss.backward()

            if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

            stats["selection/n_selected"] = float(sel.size(0))
            stats["lr"] = self.optimizer.param_groups[0]["lr"]
            all_stats.append(stats)

        if not all_stats:
            return {"ppo/early_stopped": 0.0, "selection/frac_kept": frac_kept}

        # Aggregate across replayed steps
        agg = all_stats[-1].copy()
        if len(all_stats) > 1:
            keys: set = set()
            for s in all_stats:
                keys.update(s.keys())
            for key in keys:
                vals = [s[key] for s in all_stats if key in s]
                if not vals or not isinstance(vals[0], (int, float)):
                    continue
                agg[key] = float(np.mean(vals))

        agg["selection/frac_kept"] = frac_kept
        agg["ppo/early_stopped"]   = 0.0
        return agg

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run_rlif_core(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        rollout_data: Dict[str, Any],
        initial_model: Dict,
        initial_opt: Dict,
    ) -> Dict[str, float]:
        """
        4-phase RLIF over a single (possibly merged) batch.
        initial_model / initial_opt are the θ_0 snapshots to restore in Phase 4.
        """
        # Phase 1 ─ auxiliary training (captures gradient snapshots on CPU)
        logger.info("RLIF Phase 1: auxiliary training (%d epochs, batch=%d)",
                    self.ppo_epochs, query_ids.shape[0])
        t1 = time.time()
        grad_snapshots, mb_inds_list = self._auxiliary_training(
            query_ids, query_mask, rollout_data
        )
        t2 = time.time()

        # Phase 2 ─ RLIF scoring (u from last snapshot, GPU matmuls)
        logger.info("RLIF Phase 2: computing trajectory scores (%d steps)", len(grad_snapshots))
        step_scores = self._compute_trajectory_scores(
            query_ids, grad_snapshots, mb_inds_list,
        )
        del grad_snapshots
        t3 = time.time()

        # Phase 3 ─ filter
        logger.info("RLIF Phase 3: filtering")
        filtered_mb_inds, frac_kept = self._filter_trajectory(step_scores)
        del step_scores
        t4 = time.time()

        # Phase 4 ─ restore θ_0 and replay
        logger.info("RLIF Phase 4: replaying with filtered trajectory")
        self._restore_params(initial_model)
        self.optimizer.load_state_dict(initial_opt)
        del initial_model, initial_opt

        stats = self._replay_filtered(
            query_ids, query_mask, filtered_mb_inds, rollout_data, frac_kept
        )
        t5 = time.time()

        stats["timing/phase1_aux_s"]    = t2 - t1
        stats["timing/phase2_score_s"]  = t3 - t2
        stats["timing/phase3_filter_s"] = t4 - t3
        stats["timing/phase4_replay_s"] = t5 - t4
        return stats

    def _run_ppo_epochs(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        rollout_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        RLIF entry point. Supports a multi-rollout horizon via `rlif_horizon`:

          rlif_horizon <= 0 or <= steps_per_rollout:
            Single-rollout RLIF (original behaviour).

          rlif_horizon > steps_per_rollout:
            Buffer consecutive rollouts (all generated from the same θ_0) until
            the accumulated gradient steps reach rlif_horizon, then run RLIF over
            the merged batch.  During buffering, no parameter updates are made so
            that all rollout data is on-policy w.r.t. the same θ_0.
        """
        batch_size        = query_ids.shape[0]
        steps_per_rollout = self.ppo_epochs * (batch_size // self.mini_batch_size)
        horizon           = getattr(self.args, "rlif_horizon", -1)

        if horizon <= 0 or horizon <= steps_per_rollout:
            # ── single-rollout path (original behaviour) ──────────────────
            initial_model = self._save_all_trainable_params()
            initial_opt   = copy.deepcopy(self.optimizer.state_dict())
            return self._run_rlif_core(
                query_ids, query_mask, rollout_data, initial_model, initial_opt
            )

        # ── multi-rollout buffering path ───────────────────────────────────
        if not self._rlif_buffer:
            # First rollout in this horizon window — snapshot θ_0
            self._rlif_initial_model = self._save_all_trainable_params()
            self._rlif_initial_opt   = copy.deepcopy(self.optimizer.state_dict())
            self._rlif_phase1_time   = 0.0

        # Phase 1: run auxiliary training immediately for this rollout
        logger.info("RLIF Phase 1 (rollout %d): auxiliary training (%d epochs, batch=%d)",
                    len(self._rlif_buffer) + 1, self.ppo_epochs, batch_size)
        t_aux_start = time.time()
        checkpoints, mb_inds_list = self._auxiliary_training(query_ids, query_mask, rollout_data)
        self._rlif_phase1_time += time.time() - t_aux_start

        data_cpu = {k: (v.cpu() if isinstance(v, Tensor) else v) for k, v in rollout_data.items()}
        self._rlif_buffer.append((query_ids.cpu(), query_mask.cpu(), data_cpu, checkpoints, mb_inds_list))
        self._rlif_buffer_steps += steps_per_rollout

        if self._rlif_buffer_steps < horizon:
            logger.info(
                "RLIF buffer: %d/%d steps accumulated (%d rollouts) — waiting for more rollouts",
                self._rlif_buffer_steps, horizon, len(self._rlif_buffer),
            )
            return {"ppo/buffering": 1.0, "selection/frac_kept": 1.0}

        # Buffer full — merge rollouts and run Phases 2-4
        n_rollouts = len(self._rlif_buffer)
        logger.info(
            "RLIF buffer full: %d steps across %d rollouts — running Phases 2-4",
            self._rlif_buffer_steps, n_rollouts,
        )
        dev = query_ids.device

        # Merge rollout data
        merged_query = torch.cat([b[0] for b in self._rlif_buffer]).to(dev)
        merged_mask  = torch.cat([b[1] for b in self._rlif_buffer]).to(dev)
        merged_data: Dict[str, Any] = {}
        for k in self._rlif_buffer[0][2]:
            v0 = self._rlif_buffer[0][2][k]
            if isinstance(v0, Tensor):
                merged_data[k] = torch.cat([b[2][k] for b in self._rlif_buffer]).to(dev)
            elif isinstance(v0, list):
                merged_data[k] = [item for b in self._rlif_buffer for item in b[2][k]]
            else:
                merged_data[k] = v0

        # Merge grad snapshots and re-index mb_inds into the merged batch
        all_grad_snapshots: List[Dict] = []
        all_mb_inds: List[Tensor] = []
        offset = 0
        for b in self._rlif_buffer:
            _, _, _, snaps, mbs = b
            all_grad_snapshots.extend(snaps)
            for mb in mbs:
                all_mb_inds.append(mb + offset)
            offset += b[0].shape[0]

        initial_model  = self._rlif_initial_model
        initial_opt    = self._rlif_initial_opt
        phase1_time    = self._rlif_phase1_time

        self._rlif_buffer.clear()
        self._rlif_buffer_steps  = 0
        self._rlif_initial_model = None
        self._rlif_initial_opt   = None
        self._rlif_phase1_time   = 0.0

        # Phase 2 — RLIF scoring (u from last snapshot, GPU matmuls)
        logger.info("RLIF Phase 2: computing trajectory scores (%d steps)", len(all_grad_snapshots))
        t2 = time.time()
        step_scores = self._compute_trajectory_scores(
            merged_query, all_grad_snapshots, all_mb_inds,
        )
        del all_grad_snapshots
        t3 = time.time()

        # Phase 3 — filter
        logger.info("RLIF Phase 3: filtering")
        filtered_mb_inds, frac_kept = self._filter_trajectory(step_scores)
        del step_scores
        t4 = time.time()

        # Phase 4 — restore θ_0 and replay
        logger.info("RLIF Phase 4: replaying with filtered trajectory")
        self._restore_params(initial_model)
        self.optimizer.load_state_dict(initial_opt)
        del initial_model, initial_opt

        stats = self._replay_filtered(
            merged_query, merged_mask, filtered_mb_inds, merged_data, frac_kept
        )
        t5 = time.time()

        stats["timing/phase1_aux_s"]    = phase1_time   # total across all buffered rollouts
        stats["timing/phase2_score_s"]  = t3 - t2
        stats["timing/phase3_filter_s"] = t4 - t3
        stats["timing/phase4_replay_s"] = t5 - t4
        return stats
