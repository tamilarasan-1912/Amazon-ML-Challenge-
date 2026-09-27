"""Validation utilities for Business Entity Resolution."""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

from ..config import THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS, VAL_SPLIT, SEED


def f05_score(precision: float, recall: float) -> float:
    """Compute F_0.5 score."""
    if precision == 0 and recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def per_entity_f05(gt_matches: List[str], pred_matches: List[str]) -> float:
    """
    Compute per-entity F_0.5 as per spec:
    - If truth empty: 1.0 iff prediction empty else 0.0
    - If truth non-empty and prediction empty: 0.0
    - Else: standard F_0.5
    """
    gt_set = set(gt_matches)
    pred_set = set(pred_matches)

    if not gt_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0

    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return f05_score(precision, recall)


def compute_macro_f05(gt_dict: Dict[str, List[str]], pred_dict: Dict[str, List[str]]) -> Tuple[float, float, float, float]:
    """
    Compute macro-averaged F_0.5, precision, recall, and singleton accuracy.
    Returns: (macro_f05, macro_precision, macro_recall, singleton_accuracy)
    """
    all_s1_ids = set(gt_dict.keys()) | set(pred_dict.keys())
    f05_scores = []
    precisions = []
    recalls = []
    singleton_correct = 0
    singleton_total = 0

    for s1_id in all_s1_ids:
        gt = gt_dict.get(s1_id, [])
        pred = pred_dict.get(s1_id, [])

        gt_set = set(gt)
        pred_set = set(pred)

        if not gt_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
                f05_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                f05_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
        elif not pred_set:
            f05_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
        else:
            tp = len(gt_set & pred_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f05 = f05_score(precision, recall)
            f05_scores.append(f05)
            precisions.append(precision)
            recalls.append(recall)

    macro_f05 = np.mean(f05_scores) if f05_scores else 0.0
    macro_precision = np.mean(precisions) if precisions else 0.0
    macro_recall = np.mean(recalls) if recalls else 0.0
    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0

    return macro_f05, macro_precision, macro_recall, singleton_acc


def apply_threshold(candidates: Dict[str, List], probs: np.ndarray,
                    threshold: float) -> Dict[str, List[str]]:
    """Apply threshold to get predictions per S1 entity."""
    predictions = {s1_id: [] for s1_id in candidates.keys()}
    idx = 0
    for s1_id, cand_list in candidates.items():
        for cand in cand_list:
            if probs[idx] >= threshold:
                predictions[s1_id].append(cand.s23_id)
            idx += 1
    return predictions


def find_best_threshold(candidates: Dict[str, List], probs: np.ndarray,
                        gt_dict: Dict[str, List[str]]) -> Tuple[float, Dict]:
    """Find threshold that maximizes validation F_0.5."""
    best_f05 = -1
    best_thresh = THRESHOLD_MIN
    best_metrics = {}

    thresholds = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS)

    for thresh in thresholds:
        pred_dict = apply_threshold(candidates, probs, thresh)
        f05, prec, rec, sing_acc = compute_macro_f05(gt_dict, pred_dict)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_metrics = {
                "f05": f05,
                "precision": prec,
                "recall": rec,
                "singleton_accuracy": sing_acc,
            }

    return best_thresh, best_metrics


def split_s1_entities(s1_ids: List[str], val_split: float = VAL_SPLIT,
                      seed: int = SEED) -> Tuple[List[str], List[str]]:
    """Split S1 entities into train/val by entity (not by pair)."""
    np.random.seed(seed)
    shuffled = np.random.permutation(s1_ids)
    split_idx = int(len(shuffled) * (1 - val_split))
    return list(shuffled[:split_idx]), list(shuffled[split_idx:])


def filter_candidates_by_s1(candidates: Dict[str, List], s1_ids: List[str]) -> Dict[str, List]:
    """Filter candidates to only include given S1 entities."""
    return {s1_id: cands for s1_id, cands in candidates.items() if s1_id in set(s1_ids)}


def analyze_errors(candidates: Dict[str, List], probs: np.ndarray,
                   gt_dict: Dict[str, List[str]], threshold: float,
                   s1_records: List, s23_records: List,
                   top_k: int = 30) -> List[Dict]:
    """Analyze worst errors on validation set."""
    pred_dict = apply_threshold(candidates, probs, threshold)
    errors = []

    s1_dict = {r.entity_id: r for r in s1_records}
    s23_dict = {r.entity_id: r for r in s23_records}

    for s1_id in gt_dict:
        gt = set(gt_dict.get(s1_id, []))
        pred = set(pred_dict.get(s1_id, []))

        false_merges = pred - gt
        missed = gt - pred

        if false_merges or missed:
            s1_rec = s1_dict.get(s1_id)
            error_info = {
                "s1_id": s1_id,
                "s1_name": s1_rec.name["raw"] if s1_rec else "",
                "s1_address": s1_rec.address["raw"] if s1_rec else "",
                "gt_matches": list(gt),
                "pred_matches": list(pred),
                "false_merges": list(false_merges),
                "missed": list(missed),
                "probs": {}
            }
            for cand in candidates.get(s1_id, []):
                # Find index
                pass
            errors.append(error_info)

    return errors[:top_k]