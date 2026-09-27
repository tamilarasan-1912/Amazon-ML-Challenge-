"""Business Entity Resolution package."""

from .normalize import (
    normalize_text,
    tokenize,
    normalize_name,
    normalize_address,
    normalize_record,
    normalize_dataframe,
    NormalizedRecord,
)
from .blocking import (
    generate_candidates,
    compute_blocking_recall,
    CandidatePair,
)
from .features import (
    compute_all_features,
    build_feature_matrix,
)
from .model import (
    train_model,
    save_model,
    load_model,
    predict_proba,
    get_feature_importance,
)
from .validate import (
    per_entity_f05,
    compute_macro_f05,
    find_best_threshold,
    apply_threshold,
    split_s1_entities,
)
from .predict import (
    run_inference,
    write_outputs,
)

__all__ = [
    "normalize_text",
    "tokenize",
    "normalize_name",
    "normalize_address",
    "normalize_record",
    "normalize_dataframe",
    "NormalizedRecord",
    "generate_candidates",
    "compute_blocking_recall",
    "CandidatePair",
    "compute_all_features",
    "build_feature_matrix",
    "train_model",
    "save_model",
    "load_model",
    "predict_proba",
    "get_feature_importance",
    "per_entity_f05",
    "compute_macro_f05",
    "find_best_threshold",
    "apply_threshold",
    "split_s1_entities",
    "run_inference",
    "write_outputs",
]