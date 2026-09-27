"""Validate submission for Business Entity Resolution competition."""

import argparse
import pandas as pd
import sys


def validate_submission(matching_path: str, candidate_path: str, test_dir: str) -> bool:
    """Validate submission files against requirements."""
    
    # Load test data
    s1_test = pd.read_csv(f"{test_dir}/test_source1.tsv", sep="\t")
    s2_test = pd.read_csv(f"{test_dir}/test_source2.tsv", sep="\t")
    s3_test = pd.read_csv(f"{test_dir}/test_source3.tsv", sep="\t")
    
    s1_ids = set(s1_test["entity_id"])
    s2_ids = set(s2_test["entity_id"])
    s3_ids = set(s3_test["entity_id"])
    all_s23_ids = s2_ids | s3_ids
    
    # Load matching results
    matching = pd.read_csv(matching_path, sep="\t")
    candidate = pd.read_csv(candidate_path, sep="\t")
    
    errors = []
    
    # Check columns
    if list(matching.columns) != ["source1_entity_id", "matched_entity_ids"]:
        errors.append(f"matching_results.tsv has wrong columns: {list(matching.columns)}")
    if list(candidate.columns) != ["source1_entity_id", "candidate_entity_ids"]:
        errors.append(f"candidate_pairs.tsv has wrong columns: {list(candidate.columns)}")
    
    # Check row count - exactly one row per test S1 entity
    if len(matching) != len(s1_ids):
        errors.append(f"matching_results.tsv has {len(matching)} rows, expected {len(s1_ids)}")
    if len(candidate) != len(s1_ids):
        errors.append(f"candidate_pairs.tsv has {len(candidate)} rows, expected {len(s1_ids)}")
    
    # Check each S1 entity appears exactly once
    matching_s1_ids = set(matching["source1_entity_id"])
    candidate_s1_ids = set(candidate["source1_entity_id"])
    
    if matching_s1_ids != s1_ids:
        missing = s1_ids - matching_s1_ids
        extra = matching_s1_ids - s1_ids
        if missing:
            errors.append(f"Missing S1 entities in matching_results: {missing}")
        if extra:
            errors.append(f"Extra S1 entities in matching_results: {extra}")
    
    if candidate_s1_ids != s1_ids:
        missing = s1_ids - candidate_s1_ids
        extra = candidate_s1_ids - s1_ids
        if missing:
            errors.append(f"Missing S1 entities in candidate_pairs: {missing}")
        if extra:
            errors.append(f"Extra S1 entities in candidate_pairs: {extra}")
    
    # Check matched IDs are valid S2/S3 IDs and no duplicates
    for _, row in matching.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]
        if pd.isna(matched_str) or matched_str == "":
            matched = []
        else:
            matched = [m.strip() for m in str(matched_str).split(",")]
        
        # Check duplicates
        if len(matched) != len(set(matched)):
            errors.append(f"Duplicate matched IDs for {s1_id}: {matched}")
        
        # Check all matched IDs exist in test S2/S3
        for m in matched:
            if m not in all_s23_ids:
                errors.append(f"Matched ID {m} for {s1_id} not found in test S2/S3")
    
    # Check candidate IDs are valid S2/S3 IDs and no duplicates
    for _, row in candidate.iterrows():
        s1_id = row["source1_entity_id"]
        cand_str = row["candidate_entity_ids"]
        if pd.isna(cand_str) or cand_str == "":
            cand = []
        else:
            cand = [c.strip() for c in str(cand_str).split(",")]
        
        if len(cand) != len(set(cand)):
            errors.append(f"Duplicate candidate IDs for {s1_id}: {cand}")
        
        for c in cand:
            if c not in all_s23_ids:
                errors.append(f"Candidate ID {c} for {s1_id} not found in test S2/S3")
    
    # Check matches ⊆ candidates
    for _, row in matching.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]
        if pd.isna(matched_str) or matched_str == "":
            matched = set()
        else:
            matched = set(m.strip() for m in str(matched_str).split(","))
        
        cand_row = candidate[candidate["source1_entity_id"] == s1_id]
        if len(cand_row) == 0:
            errors.append(f"No candidate row for {s1_id}")
            continue
        cand_str = cand_row.iloc[0]["candidate_entity_ids"]
        if pd.isna(cand_str) or cand_str == "":
            cand = set()
        else:
            cand = set(c.strip() for c in str(cand_str).split(","))
        
        if not matched.issubset(cand):
            extra = matched - cand
            errors.append(f"Matched IDs not in candidates for {s1_id}: {extra}")
    
    # Check deterministic ordering (sorted by source1_entity_id)
    matching_sorted = matching["source1_entity_id"].tolist()
    if matching_sorted != sorted(matching_sorted):
        errors.append("matching_results.tsv not sorted by source1_entity_id")
    
    candidate_sorted = candidate["source1_entity_id"].tolist()
    if candidate_sorted != sorted(candidate_sorted):
        errors.append("candidate_pairs.tsv not sorted by source1_entity_id")
    
    if errors:
        print("VALIDATION FAILED:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("PASS")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to test data directory")
    args = parser.parse_args()
    
    success = validate_submission(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()