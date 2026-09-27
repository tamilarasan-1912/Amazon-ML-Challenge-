"""Main training pipeline for Business Entity Resolution."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from .normalize import NormalizedRecord, normalize_dataframe
from .blocking import generate_candidates, compute_blocking_recall
from .features import build_feature_matrix
from .model import train_model, save_model, get_feature_importance
from .validate import (split_s1_entities, filter_candidates_by_s1, find_best_threshold,
                       compute_macro_f05, apply_threshold)
from ..config import (TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
                      MODEL_DIR, MODEL_PATH, VAL_SPLIT, SEED, BLOCKING_RECALL_TARGET,
                      LGBM_PARAMS)


def load_ground_truth(path) -> Dict[str, List[str]]:
    """Load ground truth as dict: s1_id -> list of matched s2/s3 ids."""
    gt_df = pd.read_csv(path, sep="\t")
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        if pd.isna(matched) or matched == "":
            gt_dict[s1_id] = []
        else:
            gt_dict[s1_id] = [m.strip() for m in str(matched).split(",")]
    return gt_dict


def prepare_training_data(s1_records: List[NormalizedRecord],
                          s23_records: List[NormalizedRecord],
                          gt_dict: Dict[str, List[str]],
                          train_s1_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[str, str]]]:
    """Prepare training features and labels."""
    s1_country = {r.entity_id: r.country for r in s1_records}
    s23_country = {r.entity_id: r.country for r in s23_records}

    candidates = generate_candidates(s1_records, s23_records, s1_country, s23_country)
    train_candidates = filter_candidates_by_s1(candidates, train_s1_ids)

    X, feature_names, pair_list = build_feature_matrix(train_candidates, s1_records, s23_records)

    # Build labels
    y = []
    for s1_id, s23_id in pair_list:
        y.append(1 if s23_id in gt_dict.get(s1_id, []) else 0)

    return X, np.array(y), feature_names, pair_list


def evaluate_on_val(s1_records: List[NormalizedRecord],
                    s23_records: List[NormalizedRecord],
                    gt_dict: Dict[str, List[str]],
                    val_s1_ids: List[str],
                    model,
                    feature_names: List[str],
                    threshold: float) -> Dict:
    """Evaluate model on validation set."""
    s1_country = {r.entity_id: r.country for r in s1_records}
    s23_country = {r.entity_id: r.country for r in s23_records}

    candidates = generate_candidates(s1_records, s23_records, s1_country, s23_country)
    val_candidates = filter_candidates_by_s1(candidates, val_s1_ids)

    X, _, pair_list = build_feature_matrix(val_candidates, s1_records, s23_records)

    # Reorder features to match model
    model_features = model.feature_name()
    if feature_names != model_features:
        feat_idx = [feature_names.index(f) for f in model_features]
        X = X[:, feat_idx]

    probs = model.predict(X, num_iteration=model.best_iteration)
    pred_dict = apply_threshold(val_candidates, probs, threshold)

    # Build full val gt dict
    val_gt = {s1_id: gt_dict[s1_id] for s1_id in val_s1_ids if s1_id in gt_dict}

    f05, prec, rec, sing_acc = compute_macro_f05(val_gt, pred_dict)

    return {
        "f05": f05,
        "precision": prec,
        "recall": rec,
        "singleton_accuracy": sing_acc,
        "val_candidates": val_candidates,
        "val_probs": probs,
    }


def run_error_analysis(val_results: Dict, s1_records: List, s23_records: List,
                       gt_dict: Dict, threshold: float, top_k: int = 30) -> None:
    """Print worst errors for analysis."""
    from .validate import apply_threshold

    val_candidates = val_results["val_candidates"]
    val_probs = val_results["val_probs"]
    val_s1_ids = list(val_candidates.keys())

    pred_dict = apply_threshold(val_candidates, val_probs, threshold)

    s1_dict = {r.entity_id: r for r in s1_records}
    s23_dict = {r.entity_id: r for r in s23_records}

    errors = []
    for s1_id in val_s1_ids:
        gt = set(gt_dict.get(s1_id, []))
        pred = set(pred_dict.get(s1_id, []))

        false_merges = pred - gt
        missed = gt - pred

        if false_merges or missed:
            s1_rec = s1_dict.get(s1_id)
            for fm in false_merges:
                m_rec = s23_dict.get(fm)
                prob_idx = list(val_candidates[s1_id]).index(next(c for c in val_candidates[s1_id] if c.s23_id == fm))
                prob = val_probs[prob_idx] if prob_idx < len(val_probs) else 0
                errors.append({
                    "type": "false_merge",
                    "s1_id": s1_id,
                    "s23_id": fm,
                    "prob": prob,
                    "s1_name": s1_rec.name["raw"] if s1_rec else "",
                    "s1_addr": s1_rec.address["raw"] if s1_rec else "",
                    "s23_name": m_rec.name["raw"] if m_rec else "",
                    "s23_addr": m_rec.address["raw"] if m_rec else "",
                })
            for ms in missed:
                m_rec = s23_dict.get(ms)
                prob_idx = list(val_candidates[s1_id]).index(next(c for c in val_candidates[s1_id] if c.s23_id == ms))
                prob = val_probs[prob_idx] if prob_idx < len(val_probs) else 0
                errors.append({
                    "type": "missed_match",
                    "s1_id": s1_id,
                    "s23_id": ms,
                    "prob": prob,
                    "s1_name": s1_rec.name["raw"] if s1_rec else "",
                    "s1_addr": s1_rec.address["raw"] if s1_rec else "",
                    "s23_name": m_rec.name["raw"] if m_rec else "",
                    "s23_addr": m_rec.address["raw"] if m_rec else "",
                })

    # Sort by prob (high prob false merges, low prob missed matches)
    false_merges = [e for e in errors if e["type"] == "false_merge"]
    missed = [e for e in errors if e["type"] == "missed_match"]
    false_merges.sort(key=lambda x: -x["prob"])
    missed.sort(key=lambda x: x["prob"])

    print(f"\n=== TOP {top_k} FALSE MERGES ===")
    for e in false_merges[:top_k]:
        print(f"  {e['s1_id']} -> {e['s23_id']} (prob={e['prob']:.4f})")
        print(f"    S1: {e['s1_name']} | {e['s1_addr']}")
        print(f"    M:  {e['s23_name']} | {e['s23_addr']}")

    print(f"\n=== TOP {top_k} MISSED MATCHES ===")
    for e in missed[:top_k]:
        print(f"  {e['s1_id']} -> {e['s23_id']} (prob={e['prob']:.4f})")
        print(f"    S1: {e['s1_name']} | {e['s1_addr']}")
        print(f"    M:  {e['s23_name']} | {e['s23_addr']}")


def main():
    print("=== Loading Data ===")
    s1_train_df = pd.read_csv(TRAIN_SOURCE1, sep="\t")
    s2_train_df = pd.read_csv(TRAIN_SOURCE2, sep="\t")
    s3_train_df = pd.read_csv(TRAIN_SOURCE3, sep="\t")
    gt_dict = load_ground_truth(TRAIN_GROUND_TRUTH)

    print(f"Train S1: {len(s1_train_df)}, S2: {len(s2_train_df)}, S3: {len(s3_train_df)}")
    print(f"Ground truth entities: {len(gt_dict)}")

    # Normalize
    print("\n=== Normalizing ===")
    s1_records = normalize_dataframe(s1_train_df)
    s2_records = normalize_dataframe(s2_train_df)
    s3_records = normalize_dataframe(s3_train_df)
    s23_records = s2_records + s3_records

    # Split S1 entities
    all_s1_ids = [r.entity_id for r in s1_records]
    train_s1_ids, val_s1_ids = split_s1_entities(all_s1_ids)
    print(f"Train S1: {len(train_s1_ids)}, Val S1: {len(val_s1_ids)}")

    # Check blocking recall on val
    print("\n=== Blocking Recall Check ===")
    s1_country = {r.entity_id: r.country for r in s1_records}
    s23_country = {r.entity_id: r.country for r in s23_records}
    val_candidates = generate_candidates(
        [r for r in s1_records if r.entity_id in val_s1_ids],
        s23_records, s1_country, s23_country
    )
    val_gt = {s1_id: gt_dict[s1_id] for s1_id in val_s1_ids if s1_id in gt_dict}
    recall_ceiling = compute_blocking_recall(val_candidates, val_gt)
    print(f"Blocking recall ceiling: {recall_ceiling:.4f}")
    assert recall_ceiling >= BLOCKING_RECALL_TARGET, f"Recall ceiling {recall_ceiling} < {BLOCKING_RECALL_TARGET}"

    # Train model
    print("\n=== Training Model ===")
    X_train, y_train, feature_names, _ = prepare_training_data(s1_records, s23_records, gt_dict, train_s1_ids)
    print(f"Training pairs: {len(y_train)}, Positives: {y_train.sum()}, Negatives: {len(y_train) - y_train.sum()}")

    model = train_model(X_train, y_train, feature_names, LGBM_PARAMS)

    # Feature importance
    print("\n=== Feature Importances ===")
    for name, imp in get_feature_importance(model, feature_names)[:20]:
        print(f"  {name}: {imp}")

    # Evaluate on val and find best threshold
    print("\n=== Finding Best Threshold ===")
    s1_country = {r.entity_id: r.country for r in s1_records}
    s23_country = {r.entity_id: r.country for r in s23_records}
    val_candidates = generate_candidates(
        [r for r in s1_records if r.entity_id in val_s1_ids],
        s23_records, s1_country, s23_country
    )
    X_val, _, _ = build_feature_matrix(val_candidates, s1_records, s23_records)
    probs_val = model.predict(X_val, num_iteration=model.best_iteration)
    best_thresh, best_metrics = find_best_threshold(val_candidates, probs_val, val_gt)
    print(f"Best threshold: {best_thresh:.4f}")
    print(f"Val F_0.5: {best_metrics['f05']:.4f}, Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}, Singleton Acc: {best_metrics['singleton_accuracy']:.4f}")

    # Error analysis
    print("\n=== Error Analysis ===")
    run_error_analysis({
        "val_candidates": val_candidates,
        "val_probs": probs_val,
    }, s1_records, s23_records, gt_dict, best_thresh)

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, feature_names, best_thresh, best_metrics)
    print(f"\nModel saved to {MODEL_PATH}")

    # Final evaluation on val with best threshold
    pred_dict = apply_threshold(val_candidates, probs_val, best_thresh)
    f05, prec, rec, sing_acc = compute_macro_f05(val_gt, pred_dict)
    print(f"\n=== Final Validation Metrics ===")
    print(f"F_0.5: {f05:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"Singleton Accuracy: {sing_acc:.4f}")


if __name__ == "__main__":
    main()