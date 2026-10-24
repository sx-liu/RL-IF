import os
import os.path as osp
import pickle
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

from rl_attribution.utils.logger import get_logger
from rl_attribution.utils.tensor import (
    flatten_ob_act,
    flatten_ob_act_prob_adv,
)

from collections import defaultdict

logger = get_logger()

def _load_scores(folder: str, method: str, group_by: str) -> Tuple[Dict[Any, float], str]:
    """
    Return (raw_scores_dict, key_used)
    """
    if method == "RLIF-opt":
        file = osp.join(folder, "attribution_scores.pkl")
        results = pickle.load(open(file, "rb"))
        key = "influence_scores"

    elif method in ["TracIn", "TracIn-fixed", "HYDRA"]:
        file = osp.join(folder, "attribution_scores.pkl")
        results = pickle.load(open(file, "rb"))
        key = "influence_scores"

    elif method == "RLIF-opt-advnorm":
        file = osp.join(folder, "sgd_influence_advnorm_scores.pkl")
        results = pickle.load(open(file, "rb"))
        key = "sgd_influence_advnorm_scores"

    elif method == "RLIF-opt-fixed":
        file = osp.join(folder, "sgd_influence_scores.pkl")
        results = pickle.load(open(file, "rb"))
        key = "influence_scores"

    elif method == "RLIF":
        file = osp.join(folder, "attribution_scores.pkl")
        results = pickle.load(open(file, "rb"))
        key = "influence_scores"

    elif method == "RLIF-sgd":
        file = osp.join(folder, "rlif_influence_scores.pkl")
        results = pickle.load(open(file, "rb"))
        key = "influence_scores"

    elif method == "random":
        file = osp.join(folder, "attribution_scores.pkl")
        results = pickle.load(open(file, "rb"))
        key = "random_scores"
    else:
        raise NotImplementedError(f"Unknown method: {method}")

    if group_by != "none":
        key += f"_{group_by}"

    return results[key], key

def _score_to_linear_weight(
    score: float,
    selected_min: float,
    selected_max: float,
    w_min: float = 0.0,
) -> float:
    """
    Linearly map score in [selected_min, selected_max] to weight in [w_min, 1].

    - score == selected_min -> w_min
    - score == selected_max -> 1

    If selected_min == selected_max, return 1 for all selected samples.
    """
    if selected_max <= selected_min:
        return 1.0
    alpha = (score - selected_min) / (selected_max - selected_min)
    return float(w_min + (1.0 - w_min) * alpha)

def build_dataset_from_influence(
    *,
    folder: str,
    model_path: str,
    k: int = 10,
    ed: int = 11,
    n_epochs: int = 10,
    n_updates: int = 320,
    batch_size: int = 64,
    final_ed: int = -1,
    method: str = "TracIn",
    seed: int = 42,
    verbose: bool = True,
    drop_percentile: float = -1.0,
    random_flag: bool = False,
    mode: str = "neg",
    selection_mode: str = "remove",
    order: str = "fixed",
    out_format: str = "reweight",
    out_dir: Optional[str] = None,
    group_by: str = "none",
    w_min: float = 0.0,
) -> str:
    """
    Reproduces the exact behavior of the original script as a function.

    Returns:
        The on-disk dataset directory path (string)
    """
    assert method in ['TracIn','HYDRA','RLIF-opt','random','RLIF-opt-advnorm','RLIF-opt-fixed','TracIn-fixed','RLIF','RLIF-sgd']
    assert mode in ["neg", "bottom", "random", "neg-bottom", "neg-random"]
    assert order in ["fixed", "shuffle"]
    assert out_format in ["reweight", "remove"]
    assert group_by in ["none", "ob_act", "ob_act_prob_adv"]
    assert selection_mode in ["remove", "fixed_weight", "reweight"]

    if final_ed == -1:
        final_ed = ed

    raw_scores, _ = _load_scores(folder, method, group_by)

    items_to_discard: Set[Any] = set()
    begin_rollout_idx = k
    end_rollout_idx = ed

    if group_by != "none":
        d_global: Dict[Any, List[float]] = {}
        for i in range(n_updates * begin_rollout_idx, n_updates * end_rollout_idx):
            batch_i = pickle.load(
                open(osp.join(model_path / f"batch_{i+1}.pkl"), "rb")
            )
            for j in range(batch_size):
                obs = batch_i.observations[j]
                act = batch_i.actions[j]
                prob = batch_i.old_log_prob[j]
                adv = batch_i.advantages[j]
                if group_by == "ob_act":
                    key = flatten_ob_act(obs, act)
                else:
                    key = flatten_ob_act_prob_adv(obs, act, prob, adv)
                d_global[key] = d_global.get(key, []) + [raw_scores[(i, j)]]
    else:
        d_global = raw_scores

    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        logger.info("mode", mode)

    sample_scores: Dict[Any, float] = {}

    for i in range(n_updates * begin_rollout_idx, n_updates * end_rollout_idx):
        batch_i = pickle.load(
            open(osp.join(model_path, f"batch_{i+1}.pkl"), "rb")
        )
        for j in range(batch_size):
            if isinstance(batch_i, dict):
                obs = batch_i["observations"][j]
                act = batch_i["actions"][j]
                prob = batch_i["old_log_prob"][j]
                adv = batch_i["advantages"][j]
            else:
                obs = batch_i.observations[j]
                act = batch_i.actions[j]
                prob = batch_i.old_log_prob[j]
                adv = batch_i.advantages[j]

            if group_by == "none":
                group_key = (i, j)
                score = float(d_global[group_key])
            elif group_by == "ob_act":
                group_key = flatten_ob_act(obs, act)
                score = float(np.mean(d_global[group_key]))
            else:
                group_key = flatten_ob_act_prob_adv(obs, act, prob, adv)
                score = float(np.mean(d_global[group_key]))

            sample_scores[group_key] = score

    all_keys = list(sample_scores.keys())
    all_scores = np.array([sample_scores[k] for k in all_keys], dtype=np.float64)

    selected_keys: Set[Any] = set()

    if mode == "bottom":
        assert drop_percentile >= 0, "drop_percentile must be non-negative for mode='bottom'"
        threshold = np.percentile(all_scores, drop_percentile)
        selected_keys = {k for k, s in sample_scores.items() if s <= threshold}

    elif mode == "neg":
        selected_keys = {k for k, s in sample_scores.items() if s < 0}

    elif mode == "neg-bottom":
        neg_scores = all_scores[all_scores < 0]
        if len(neg_scores) > 0:
            assert drop_percentile >= 0, "drop_percentile must be non-negative for mode='neg-bottom'"
            threshold = np.percentile(neg_scores, drop_percentile)
            selected_keys = {k for k, s in sample_scores.items() if s < 0 and s <= threshold}
        else:
            selected_keys = set()

    elif mode == "random":
        import random
        random.seed(seed)
        n_select = int(len(all_keys) * 0.01 * drop_percentile)
        n_select = min(max(n_select, 0), len(all_keys))
        selected_keys = set(random.sample(all_keys, n_select))

    elif mode == "neg-random":
        import random
        random.seed(seed)
        neg_keys = [k for k, s in sample_scores.items() if s < 0]
        n_select = int(len(neg_keys) * 0.01 * drop_percentile)
        n_select = min(max(n_select, 0), len(neg_keys))
        selected_keys = set(random.sample(neg_keys, n_select))

    else:
        raise NotImplementedError(f"Unsupported mode: {mode}")

    items_to_discard = set(selected_keys)

    weight_map: Dict[Any, float] = {k: 1.0 for k in all_keys}

    if selection_mode == "remove":

        for k in selected_keys:
            weight_map[k] = 0.0

    elif selection_mode == "fixed_weight":

        for k in selected_keys:
            weight_map[k] = w_min

    elif selection_mode == "reweight":

        if mode in ["bottom", "neg", "neg-bottom"]:
            if len(selected_keys) > 0:
                selected_scores = np.array([sample_scores[k] for k in selected_keys], dtype=np.float64)
                selected_min = float(np.min(selected_scores))
                selected_max = float(np.max(selected_scores))

                for k in selected_keys:
                    weight_map[k] = _score_to_linear_weight(
                        score=sample_scores[k],
                        selected_min=selected_min,
                        selected_max=selected_max,
                        w_min=w_min,
                    )
        elif mode in ["random", "neg-random"]:

            for k in selected_keys:
                weight_map[k] = float(w_min)

    if out_format == "remove":

        buckets = defaultdict(list)

        if mode == "random":
            obs_list, act_list, prob_list, adv_list, val_list, ret_list = [], [], [], [], [], []
            for i in range(n_updates * begin_rollout_idx, n_updates * end_rollout_idx):
                batch_i = pickle.load(
                    open(osp.join(model_path / f"batch_{i+1}.pkl"), "rb")
                )
                for j in range(batch_size):
                    if isinstance(batch_i, dict):
                        obs = batch_i["observations"][j]
                        act = batch_i["actions"][j]
                        prob = batch_i["old_log_prob"][j]
                        adv = batch_i["advantages"][j]
                        val = batch_i["old_values"][j]
                        ret = batch_i["returns"][j]
                    else:
                        obs = batch_i.observations[j]
                        act = batch_i.actions[j]
                        prob = batch_i.old_log_prob[j]
                        adv = batch_i.advantages[j]
                        val = batch_i.old_values[j]
                        ret = batch_i.returns[j]
                    if np.random.rand() < 1 - drop_percentile * 0.01:
                        obs_list.append(obs)
                        act_list.append(act)
                        prob_list.append(prob)
                        adv_list.append(adv)
                        val_list.append(val)
                        ret_list.append(ret)
        else:
            assert n_updates % n_epochs == 0
            n_updates_per_epoch = n_updates // n_epochs
            for i in range(n_updates * begin_rollout_idx, n_updates * end_rollout_idx):
                batch_i = pickle.load(
                    open(osp.join(model_path / f"batch_{i+1}.pkl"), "rb")
                )
                rollout_idx, epoch_idx = i // n_updates, (i % n_updates) // n_updates_per_epoch
                for j in range(batch_size):
                    if isinstance(batch_i, dict):
                        obs = batch_i["observations"][j]
                        act = batch_i["actions"][j]
                        prob = batch_i["old_log_prob"][j]
                        adv = batch_i["advantages"][j]
                        val = batch_i["old_values"][j]
                        ret = batch_i["returns"][j]
                    else:
                        obs = batch_i.observations[j]
                        act = batch_i.actions[j]
                        prob = batch_i.old_log_prob[j]
                        adv = batch_i.advantages[j]
                        val = batch_i.old_values[j]
                        ret = batch_i.returns[j]

                    if group_by == "none":
                        group_key = (i, j)
                    elif group_by == "ob_act":
                        group_key = flatten_ob_act(obs, act)
                    else:
                        group_key = flatten_ob_act_prob_adv(obs, act, prob, adv)

                    if group_key not in items_to_discard:
                        buckets[(rollout_idx, epoch_idx)].append((obs, act, prob, adv, val, ret))

        sorted_keys = sorted(buckets.keys())
        t = n_updates * begin_rollout_idx

        for (rollout_idx, epoch_idx) in sorted_keys:
            records = buckets[(rollout_idx, epoch_idx)]
            logger.info(f"Rollout {rollout_idx}, epoch {epoch_idx}: {len(records)} samples")

            indices = list(range(len(records)))
            if order == "shuffle":
                indices = np.random.permutation(indices)

            n_batch = (len(indices) + batch_size - 1) // batch_size
            for b in range(n_batch):
                cur = indices[b * batch_size : min(len(indices), (b + 1) * batch_size)]
                chunk = [records[ii] for ii in cur]

                obs_arr = np.array([x[0] for x in chunk])
                act_arr = np.array([x[1] for x in chunk])
                prob_arr = np.array([x[2] for x in chunk])
                adv_arr = np.array([x[3] for x in chunk])
                val_arr = np.array([x[4] for x in chunk])
                ret_arr = np.array([x[5] for x in chunk])

                out = {
                    "observations": torch.tensor(obs_arr),
                    "actions": torch.tensor(act_arr),
                    "old_values": torch.tensor(val_arr),
                    "old_log_prob": torch.tensor(prob_arr),
                    "advantages": torch.tensor(adv_arr),
                    "returns": torch.tensor(ret_arr),
                }

                pickle.dump(out, open(osp.join(out_dir, f"batch_{t+1}.pkl"), "wb"))
                t += 1

        for i in range(t, ed * n_updates):
            pickle.dump(None, open(osp.join(out_dir, f"batch_{i+1}.pkl"), "wb"))

        if verbose:
            logger.info("Saved dataset to:", out_dir)

    elif out_format == "reweight":
        total_samples = 0
        n_nontrivial = 0

        for i in range(n_updates * begin_rollout_idx, n_updates * end_rollout_idx):
            batch_i = pickle.load(
                open(osp.join(model_path, f"batch_{i+1}.pkl"), "rb")
            )

            obs_list = []
            act_list = []
            prob_list = []
            adv_list = []
            val_list = []
            ret_list = []
            weight_list = []

            for j in range(batch_size):
                if isinstance(batch_i, dict):
                    obs = batch_i["observations"][j]
                    act = batch_i["actions"][j]
                    prob = batch_i["old_log_prob"][j]
                    adv = batch_i["advantages"][j]
                    val = batch_i["old_values"][j]
                    ret = batch_i["returns"][j]
                else:
                    obs = batch_i.observations[j]
                    act = batch_i.actions[j]
                    prob = batch_i.old_log_prob[j]
                    adv = batch_i.advantages[j]
                    val = batch_i.old_values[j]
                    ret = batch_i.returns[j]

                if group_by == "none":
                    group_key = (i, j)
                elif group_by == "ob_act":
                    group_key = flatten_ob_act(obs, act)
                else:
                    group_key = flatten_ob_act_prob_adv(obs, act, prob, adv)

                weight = float(weight_map[group_key])

                obs_list.append(obs)
                act_list.append(act)
                prob_list.append(prob)
                adv_list.append(adv)
                val_list.append(val)
                ret_list.append(ret)
                weight_list.append(weight)

                total_samples += 1
                if weight < 1.0:
                    n_nontrivial += 1

            out = {
                "observations": torch.tensor(np.array(obs_list)),
                "actions": torch.tensor(np.array(act_list)),
                "old_values": torch.tensor(np.array(val_list)),
                "old_log_prob": torch.tensor(np.array(prob_list)),
                "advantages": torch.tensor(np.array(adv_list)),
                "returns": torch.tensor(np.array(ret_list)),
                "weights": torch.tensor(np.array(weight_list), dtype=torch.float32),
            }

            pickle.dump(out, open(osp.join(out_dir, f"batch_{i+1}.pkl"), "wb"))

        if verbose:
            logger.info(f"Saved reweighted dataset to: {out_dir}")
            logger.info(f"Total samples: {total_samples}")
            logger.info(f"Samples with weight < 1: {n_nontrivial}")
            logger.info(f"Selected interval size: {len(selected_keys)}")
            logger.info(f"Selection model: {selection_mode}")
            if len(selected_keys) > 0 and mode in ["bottom", "neg", "neg-bottom"]:
                sel_scores = [sample_scores[k] for k in selected_keys]
                logger.info(f"Selected score range: [{min(sel_scores)}, {max(sel_scores)}]")

    else:
        raise NotImplementedError()

    return out_dir
