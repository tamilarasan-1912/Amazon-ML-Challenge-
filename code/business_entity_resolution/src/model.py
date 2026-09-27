"""Matching model for Business Entity Resolution using LightGBM."""

import json
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..config import LGBM_PARAMS, MODEL_PATH, SEED


@dataclass
class ModelArtifact:
    model: lgb.Booster
    feature_names: List[str]
    threshold: float
    val_metrics: Dict[str, float]


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                feature_names: List[str],
                params: Dict = None) -> lgb.Booster:
    """Train LightGBM binary classifier."""
    if params is None:
        params = LGBM_PARAMS.copy()

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    model = lgb.train(params, train_data, num_boost_round=200, valid_sets=[train_data])
    return model


def predict_proba(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """Get prediction probabilities."""
    return model.predict(X, num_iteration=model.best_iteration)


def save_model(model: lgb.Booster, feature_names: List[str], threshold: float,
               val_metrics: Dict[str, float], path: str = MODEL_PATH) -> None:
    """Save model artifact."""
    model.save_model(str(path))
    meta_path = str(path).replace(".txt", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "feature_names": feature_names,
            "threshold": threshold,
            "val_metrics": val_metrics,
        }, f)


def load_model(path: str = MODEL_PATH) -> Tuple[lgb.Booster, List[str], float, Dict]:
    """Load model artifact."""
    model = lgb.Booster(model_file=str(path))
    meta_path = str(path).replace(".txt", "_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta["feature_names"], meta["threshold"], meta["val_metrics"]


def get_feature_importance(model: lgb.Booster, feature_names: List[str]) -> List[Tuple[str, float]]:
    """Get feature importances sorted by importance."""
    importance = model.feature_importance(importance_type="gain")
    return sorted(zip(feature_names, importance), key=lambda x: -x[1])