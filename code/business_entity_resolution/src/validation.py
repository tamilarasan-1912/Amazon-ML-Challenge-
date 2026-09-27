"""Validation framework for Business Entity Resolution"""
import polars as pl
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


def grouped_kfold_split(
    s1_ids: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate grouped K-fold splits by S1 entity"""
    unique_s1 = np.unique(s1_ids)
    np.random.seed(random_state)
    np.random.shuffle(unique_s1)
    
    folds = np.array_split(unique_s1, n_splits)
    
    splits = []
    for i in range(n_splits):
        val_s1 = folds[i]
        train_s1 = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        
        train_mask = np.isin(s1_ids, train_s1)
        val_mask = np.isin(s1_ids, val_s1)
        
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        
        splits.append((train_idx, val_idx))
    
    return splits


def leave_country_out_split(
    s1_ids: np.ndarray,
    countries: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Leave-one-country-out splits"""
    unique_countries = np.unique(countries[countries != ""])
    splits = []
    
    for country in unique_countries:
        val_mask = countries == country
        train_mask = ~val_mask
        
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        splits.append((train_idx, val_idx))
    
    return splits


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> Dict:
    """Compute comprehensive metrics"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f05": float(f05),
    }
    
    if y_prob is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except:
            metrics["auc"] = 0.5
    
    return metrics


def compute_macro_f05(
    s1_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute macro-averaged F0.5 per S1 entity"""
    unique_s1 = np.unique(s1_ids)
    f05_scores = []
    
    for s1 in unique_s1:
        mask = s1_ids == s1
        true_labels = y_true[mask]
        pred_labels = y_pred[mask]
        
        tp = np.sum(pred_labels * true_labels)
        fp = np.sum(pred_labels * (1 - true_labels))
        fn = np.sum((1 - pred_labels) * true_labels)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0
        
        f05_scores.append(f05)
    
    return float(np.mean(f05_scores)) if f05_scores else 0.0


def compute_singleton_accuracy(
    s1_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """Compute singleton detection accuracy"""
    unique_s1 = np.unique(s1_ids)
    
    true_singletons = 0
    pred_singletons = 0
    correct_singletons = 0
    false_merges = 0
    
    for s1 in unique_s1:
        mask = s1_ids == s1
        true_labels = y_true[mask]
        pred_labels = y_pred[mask]
        
        is_true_singleton = not true_labels.any()
        is_pred_singleton = not pred_labels.any()
        
        if is_true_singleton:
            true_singletons += 1
            if is_pred_singleton:
                correct_singletons += 1
        
        if is_pred_singleton:
            pred_singletons += 1
        
        # False merge: predicted match but no true match
        if pred_labels.any() and not true_labels.any():
            false_merges += 1
    
    singleton_precision = correct_singletons / pred_singletons if pred_singletons > 0 else 0
    singleton_recall = correct_singletons / true_singletons if true_singletons > 0 else 0
    false_merge_rate = false_merges / len(unique_s1) if len(unique_s1) > 0 else 0
    
    return {
        "singleton_accuracy": singleton_precision,
        "singleton_precision": singleton_precision,
        "singleton_recall": singleton_recall,
        "true_singletons": true_singletons,
        "pred_singletons": pred_singletons,
        "correct_singletons": correct_singletons,
        "false_merge_rate": false_merge_rate,
        "false_merges": false_merges,
    }


def compute_blocking_metrics(
    candidates_df: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
) -> Dict:
    """Compute blocking recall and candidate statistics"""
    # Positive pairs
    positives = ground_truth_df.select(["source1_entity_id", "matched_entity_id"]).unique()
    total_positives = len(positives)
    
    if total_positives == 0:
        return {"blocking_recall": 0.0, "recalled_positives": 0, "total_positives": 0}
    
    candidate_pairs = set(zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"]))
    positive_pairs = set(zip(positives["source1_entity_id"], positives["matched_entity_id"]))
    
    recalled = positive_pairs & candidate_pairs
    recall = len(recalled) / total_positives
    
    # Candidate stats
    counts = candidates_df.group_by("source1_entity_id").len()
    counts_arr = counts["len"].to_numpy()
    
    return {
        "blocking_recall": recall,
        "recalled_positives": len(recalled),
        "total_positives": total_positives,
        "avg_candidates": float(np.mean(counts_arr)),
        "median_candidates": float(np.median(counts_arr)),
        "p95_candidates": float(np.percentile(counts_arr, 95)),
        "max_candidates": int(np.max(counts_arr)),
        "min_candidates": int(np.min(counts_arr)),
        "total_candidates": len(candidates_df),
        "unique_s1": len(counts),
        "candidate_reduction": 1.0 - (len(candidates_df) / (len(ground_truth_df["source1_entity_id"].unique()) * 100)),  # approximate
    }


def run_cv_experiment(
    experiment_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    model_config: Dict,
    n_folds: int = 5,
) -> Dict:
    """Run cross-validation experiment"""
    from models import FinalMatcher
    
    splits = grouped_kfold_split(groups, n_splits=n_folds, random_state=model_config.get("random_state", 42))
    
    oof_preds = np.zeros(len(y))
    oof_probs = np.zeros(len(y))
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Running {experiment_name} fold {fold+1}/{n_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        groups_train = groups[train_idx]
        
        matcher = FinalMatcher(model_config)
        matcher.train(X_train, y_train, groups_train, feature_names)
        
        val_probs = matcher.predict_proba(X_val)
        val_preds = matcher.predict(X_val)
        
        oof_probs[val_idx] = val_probs
        oof_preds[val_idx] = val_preds
        
        metrics = compute_metrics(y_val, val_preds, val_probs)
        fold_metrics.append(metrics)
        logger.info(f"  Fold {fold+1} metrics: {metrics}")
    
    # Overall OOF metrics
    overall_metrics = compute_metrics(y, oof_preds, oof_probs)
    overall_metrics["macro_f05"] = compute_macro_f05(groups, y, oof_preds)
    overall_metrics["singleton"] = compute_singleton_accuracy(groups, y, oof_preds)
    
    result = {
        "experiment": experiment_name,
        "fold_metrics": fold_metrics,
        "overall": overall_metrics,
        "oof_probs": oof_probs.tolist(),
        "oof_preds": oof_preds.tolist(),
    }
    
    return result


def run_leave_country_out_experiment(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    countries: np.ndarray,
    feature_names: List[str],
    model_config: Dict,
) -> Dict:
    """Run leave-country-out experiment"""
    splits = leave_country_out_split(groups, countries)
    
    results = {}
    for country, (train_idx, val_idx) in zip(np.unique(countries[countries != ""]), splits):
        logger.info(f"Leave-country-out: {country}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        groups_train = groups[train_idx]
        
        matcher = FinalMatcher(model_config)
        matcher.train(X_train, y_train, groups_train, feature_names)
        
        val_probs = matcher.predict_proba(X_val)
        val_preds = matcher.predict(X_val)
        
        metrics = compute_metrics(y_val, val_preds, val_probs)
        metrics["macro_f05"] = compute_macro_f05(groups[val_idx], y_val, val_preds)
        metrics["singleton"] = compute_singleton_accuracy(groups[val_idx], y_val, val_preds)
        
        results[country] = metrics
    
    return results


def run_difficult_experiment(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    model_config: Dict,
    difficulty_scores: np.ndarray,
    threshold: float = 0.7,
) -> Dict:
    """Run experiment on difficult/noisy subset"""
    # Select difficult examples (high difficulty score)
    difficult_mask = difficulty_scores > threshold
    easy_mask = ~difficult_mask
    
    logger.info(f"Difficult examples: {difficult_mask.sum()}, Easy: {easy_mask.sum()}")
    
    if difficult_mask.sum() < 10:
        return {"error": "Not enough difficult examples"}
    
    # Train on easy, test on difficult
    X_train, X_test = X[easy_mask], X[difficult_mask]
    y_train, y_test = y[easy_mask], y[difficult_mask]
    groups_train, groups_test = groups[easy_mask], groups[difficult_mask]
    
    matcher = FinalMatcher(model_config)
    matcher.train(X_train, y_train, groups_train, feature_names)
    
    test_probs = matcher.predict_proba(X_test)
    test_preds = matcher.predict(X_test)
    
    metrics = compute_metrics(y_test, test_preds, test_probs)
    metrics["macro_f05"] = compute_macro_f05(groups_test, y_test, test_preds)
    metrics["singleton"] = compute_singleton_accuracy(groups_test, y_test, test_preds)
    
    return metrics


def log_experiment(
    experiment_log_path: str,
    experiment_id: str,
    results: Dict,
):
    """Log experiment results to JSON"""
    log_entry = {
        "experiment_id": experiment_id,
        "results": results,
    }
    
    # Load existing log
    if Path(experiment_log_path).exists():
        with open(experiment_log_path, 'r') as f:
            log = json.load(f)
    else:
        log = []
    
    log.append(log_entry)
    
    with open(experiment_log_path, 'w') as f:
        json.dump(log, f, indent=2)
    
    logger.info(f"Logged experiment {experiment_id}")


def create_experiment_summary(log_path: str) -> pl.DataFrame:
    """Create summary DataFrame from experiment log"""
    with open(log_path, 'r') as f:
        log = json.load(f)
    
    rows = []
    for entry in log:
        exp_id = entry["experiment_id"]
        res = entry["results"]
        
        if "overall" in res:
            row = {"experiment_id": exp_id}
            row.update(res["overall"])
            rows.append(row)
        elif isinstance(res, dict) and all(isinstance(v, dict) for v in res.values()):
            # Leave-country-out
            for country, metrics in res.items():
                row = {"experiment_id": exp_id, "country": country}
                row.update(metrics)
                rows.append(row)
    
    return pl.DataFrame(rows) if rows else pl.DataFrame()
