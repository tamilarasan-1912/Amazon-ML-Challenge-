# Business Entity Resolution Pipeline

End-to-end Entity Resolution system for matching business entities across three sources. Achieves **F_0.5 = 1.0000** on validation (target ≥ 0.99, stretch 0.999).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training (generates model artifact)
cd code/business_entity_resolution
PYTHONPATH=../.. python3 -m src.train

# 3. Run inference on test set
PYTHONPATH=../.. python3 -c "
from src.predict import run_inference, write_outputs
import pandas as pd
s1 = pd.read_csv('../../dataset/test/test_source1.tsv', sep='\t')
s2 = pd.read_csv('../../dataset/test/test_source2.tsv', sep='\t')
s3 = pd.read_csv('../../dataset/test/test_source3.tsv', sep='\t')
pred, cand = run_inference(s1, s2, s3)
write_outputs(pred, cand, list(s1['entity_id']))
"

# 4. Validate outputs
python3 ../../utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test
# Expected: PASS
```

## Project Structure

```
├── code/business_entity_resolution/
│   ├── config.py              # Centralized paths & hyperparameters
│   ├── src/
│   │   ├── __init__.py
│   │   ├── normalize.py       # Phase 1: Country-agnostic normalization
│   │   ├── blocking.py        # Phase 2: Multi-scheme candidate generation
│   │   ├── features.py        # Phase 3: 34 pairwise features
│   │   ├── model.py           # Phase 4: LightGBM training/loading
│   │   ├── validate.py        # Phase 4: F_0.5 scorer, threshold selection
│   │   ├── train.py           # Main training pipeline
│   │   └── predict.py         # Phase 7: Inference & output writing
│   └── models/                # Model artifacts (auto-created)
├── dataset/
│   ├── train/                 # Training data (TSV)
│   └── test/                  # Test data (TSV)
├── output/                    # Generated outputs (TSV)
├── utils/validate_submission.py  # Competition validator
├── Documentation_template.md  # Full methodology documentation
└── requirements.txt           # Pinned dependencies
```

## Pipeline Phases

| Phase | Description | Key Result |
|-------|-------------|------------|
| -1 | Data acquisition & verification | Verified local dataset |
| 0 | Data audit | 1-to-1 S2/S3→S1 mapping confirmed |
| 1 | Normalization library | Unicode, legal suffixes, address parsing |
| 2 | Blocking | 5 schemes, recall ceiling **1.0000** |
| 3 | Pairwise features | 34 features (name, address, cross) |
| 4 | Matching model | LightGBM, threshold=0.01, **F_0.5=1.0000** |
| 5 | Global consistency | Greedy conflict resolution (no change) |
| 6 | Error analysis | Zero errors on validation |
| 7 | Inference | Deterministic TSV outputs |
| 8 | Packaging | README, requirements, methodology |

## Output Format

### `output/matching_results.tsv`
```
source1_entity_id	matched_entity_ids
S1-10001	S2-10001,S3-10001
S1-10002	S2-10002,S3-10002
...
S1-10031	
```
- Exactly one row per test S1 entity (35 rows)
- Comma-separated matched S2/S3 IDs (no spaces)
- Empty for singletons (e.g., French entities)

### `output/candidate_pairs.tsv`
```
source1_entity_id	candidate_entity_ids
S1-10001	S2-10001,S2-10002,...,S3-10050
...
```
- Final candidate sets scored by model
- Every match ⊆ candidates

## Key Design Decisions

- **No country hard-coding**: Works for unseen countries (France in test)
- **No external data/APIs**: Pure in-process computation
- **MIT/Apache-2.0 licensed model**: LightGBM (<8B params)
- **Deterministic**: Fixed seeds, sorted outputs
- **Reproducible**: Single config.py, centralized paths

## Requirements

See `requirements.txt` for pinned versions. Key packages:
- `pandas`, `numpy`, `scikit-learn`
- `lightgbm` (model)
- `rapidfuzz` (string similarity)
- `metaphone` (phonetic blocking)

## Methodology Details

See `Documentation_template.md` for full methodology including:
- Blocking strategy & measured recall ceiling
- Complete feature list with definitions
- Model hyperparameters & training protocol
- Threshold selection methodology
- Validation scores & error analysis
- Failure modes observed (none on this dataset)