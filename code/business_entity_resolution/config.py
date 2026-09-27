"""Centralized configuration for Business Entity Resolution pipeline."""

from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# Output paths
OUTPUT_DIR = ROOT / "output"
MATCHING_RESULTS = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS = OUTPUT_DIR / "candidate_pairs.tsv"

# Model artifact
MODEL_DIR = ROOT / "code" / "business_entity_resolution" / "models"
MODEL_PATH = MODEL_DIR / "matching_model.txt"

# Random seed
SEED = 42

# Blocking parameters
BLOCKING_TOP_K = 100
BLOCKING_MAX_CANDIDATES = 150
BLOCKING_RECALL_TARGET = 0.995

# Model parameters
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": SEED,
    "num_threads": 4,
}

# Validation split
VAL_SPLIT = 0.2

# Threshold search
THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.99
THRESHOLD_STEPS = 200