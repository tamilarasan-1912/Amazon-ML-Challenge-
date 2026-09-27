"""Inference pipeline for Business Entity Resolution."""

import numpy as np
from typing import Dict, List, Tuple
import pandas as pd

from .normalize import NormalizedRecord, normalize_dataframe
from .blocking import CandidatePair, generate_candidates, deduplicate_candidates
from .features import compute_all_features, build_feature_matrix
from .model import load_model, predict_proba
from .validate import apply_threshold
from ..config import MATCHING_RESULTS, CANDIDATE_PAIRS, OUTPUT_DIR, MODEL_PATH


def apply_global_consistency(predictions: Dict[str, List[str]],
                             candidates: Dict[str, List[CandidatePair]],
                             probs: np.ndarray) -> Dict[str, List[str]]:
    """
    If any S2/S3 record is assigned to multiple S1 entities,
    keep only the assignment with highest model probability.
    """
    s23_to_assignments = {}
    idx = 0
    for s1_id, cand_list in candidates.items():
        for cand in cand_list:
            if probs[idx] >= 0:  # all candidates have probs
                if cand.s23_id not in s23_to_assignments:
                    s23_to_assignments[cand.s23_id] = []
                s23_to_assignments[cand.s23_id].append((s1_id, probs[idx], idx))
            idx += 1

    # Find conflicts
    to_remove = set()
    for s23_id, assignments in s23_to_assignments.items():
        if len(assignments) > 1:
            # Keep highest prob, remove others
            assignments.sort(key=lambda x: -x[1])
            for s1_id, prob, orig_idx in assignments[1:]:
                to_remove.add((s1_id, s23_id))

    # Apply removals
    cleaned = {}
    for s1_id, pred_list in predictions.items():
        cleaned[s1_id] = [m for m in pred_list if (s1_id, m) not in to_remove]
    return cleaned


def run_inference(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                  model_path: str = MODEL_PATH) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Run full inference pipeline on test data."""
    # Normalize
    s1_records = normalize_dataframe(s1_df)
    s2_records = normalize_dataframe(s2_df)
    s3_records = normalize_dataframe(s3_df)
    s23_records = s2_records + s3_records

    # Build country maps
    s1_country = {r.entity_id: r.country for r in s1_records}
    s23_country = {r.entity_id: r.country for r in s23_records}

    # Generate candidates
    candidates = generate_candidates(s1_records, s23_records, s1_country, s23_country)

    # Ensure all S1 entities have entries (even if empty candidates)
    for s1_rec in s1_records:
        if s1_rec.entity_id not in candidates:
            candidates[s1_rec.entity_id] = []

    # Build features
    X, feature_names, pair_list = build_feature_matrix(candidates, s1_records, s23_records)

    # Load model
    model, model_feature_names, threshold, _ = load_model(model_path)
    # Ensure feature order matches
    if feature_names != model_feature_names:
        # Reorder X to match model features
        feat_idx = [feature_names.index(f) for f in model_feature_names]
        X = X[:, feat_idx]

    # Predict
    probs = predict_proba(model, X)

    # Apply threshold
    predictions = apply_threshold(candidates, probs, threshold)

    # Apply global consistency (Phase 5)
    predictions = apply_global_consistency(predictions, candidates, probs)

    # Build candidate pairs output (all candidates for each S1)
    candidate_pairs = {}
    for s1_id, cand_list in candidates.items():
        candidate_pairs[s1_id] = [c.s23_id for c in cand_list]

    return predictions, candidate_pairs


def write_outputs(predictions: Dict[str, List[str]],
                  candidate_pairs: Dict[str, List[str]],
                  s1_test_ids: List[str]) -> None:
    """Write matching_results.tsv and candidate_pairs.tsv with assertions."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sort by source1_entity_id for deterministic output
    sorted_s1_ids = sorted(s1_test_ids)

    # Assertions
    for s1_id in sorted_s1_ids:
        assert s1_id in predictions, f"Missing prediction for {s1_id}"
        assert s1_id in candidate_pairs, f"Missing candidates for {s1_id}"
        pred = predictions[s1_id]
        cand = candidate_pairs[s1_id]
        assert len(pred) == len(set(pred)), f"Duplicate matches for {s1_id}"
        assert len(cand) == len(set(cand)), f"Duplicate candidates for {s1_id}"
        assert set(pred).issubset(set(cand)), f"Match not in candidates for {s1_id}"

    # Write matching_results.tsv
    with open(MATCHING_RESULTS, "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted_s1_ids:
            matches = predictions.get(s1_id, [])
            matches_str = ",".join(sorted(matches))
            f.write(f"{s1_id}\t{matches_str}\n")

    # Write candidate_pairs.tsv
    with open(CANDIDATE_PAIRS, "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted_s1_ids:
            cands = candidate_pairs.get(s1_id, [])
            cands_str = ",".join(sorted(cands))
            f.write(f"{s1_id}\t{cands_str}\n")

    print(f"Outputs written to {MATCHING_RESULTS} and {CANDIDATE_PAIRS}")