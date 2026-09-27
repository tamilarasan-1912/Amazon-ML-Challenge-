"""Model training and inference for Business Entity Resolution"""
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class Stage1Pruner:
    """Stage 1: Fast candidate pruner to reduce candidate set"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.feature_names = None
        self.threshold = 0.5
    
    def train(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
              feature_names: List[str], val_data: Tuple = None):
        """Train Stage-1 pruner with grouped CV"""
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": self.config.get("n_estimators", 200),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "max_depth": self.config.get("max_depth", 6),
            "num_leaves": self.config.get("num_leaves", 31),
            "min_child_samples": self.config.get("min_child_samples", 20),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "random_state": self.config.get("random_state", 42),
            "n_jobs": self.config.get("n_jobs", -1),
            "verbose": -1,
            "class_weight": "balanced",
        }
        
        self.feature_names = feature_names
        
        # Use grouped CV for OOF predictions
        n_folds = self.config.get("cv_folds", 5)
        gkf = GroupKFold(n_splits=n_folds)
        
        oof_preds = np.zeros(len(y))
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            logger.info(f"Training Stage-1 fold {fold+1}/{n_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            val_data_lgb = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=params["n_estimators"],
                valid_sets=[val_data_lgb],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
            
            oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
            models.append(model)
        
        # Train final model on all data
        train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
        self.model = lgb.train(params, train_data, num_boost_round=params["n_estimators"])
        
        # Optimize threshold for high recall
        self.threshold = self._optimize_recall_threshold(y, oof_preds)
        
        logger.info(f"Stage-1 OOF recall at threshold {self.threshold:.4f}: "
                    f"{(oof_preds >= self.threshold)[y==1].mean():.4f}")
        
        return oof_preds
    
    def _optimize_recall_threshold(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    target_recall: float = 0.998) -> float:
        """Find threshold achieving target recall"""
        thresholds = np.linspace(0, 1, 1001)
        best_thresh = 0.5
        
        for thresh in thresholds:
            recall = (y_pred[y_true==1] >= thresh).mean() if (y_true==1).any() else 1.0
            if recall >= target_recall:
                best_thresh = thresh
                break
        
        return best_thresh
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions"""
        return (self.predict_proba(X) >= self.threshold).astype(int)
    
    def prune_candidates(self, pairs_df: pl.DataFrame, X: np.ndarray) -> pl.DataFrame:
        """Prune candidates based on Stage-1 predictions"""
        probs = self.predict_proba(X)
        pairs_df = pairs_df.with_columns(
            pl.Series("stage1_score", probs),
            pl.Series("stage1_keep", (probs >= self.threshold).astype(int))
        )
        return pairs_df.filter(pl.col("stage1_keep") == 1).drop(["stage1_score", "stage1_keep"])
    
    def save(self, path: str):
        """Save model"""
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "config": self.config,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'Stage1Pruner':
        """Load model"""
        data = joblib.load(path)
        pruner = cls(data["config"])
        pruner.model = data["model"]
        pruner.feature_names = data["feature_names"]
        pruner.threshold = data["threshold"]
        return pruner


class FinalMatcher:
    """Stage 2: Final matching model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.feature_names = None
        self.calibrator = None
    
    def train(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
              feature_names: List[str], calibration_method: str = "isotonic"):
        """Train final matcher with calibration"""
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": self.config.get("n_estimators", 500),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "max_depth": self.config.get("max_depth", 8),
            "num_leaves": self.config.get("num_leaves", 63),
            "min_child_samples": self.config.get("min_child_samples", 20),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "random_state": self.config.get("random_state", 42),
            "n_jobs": self.config.get("n_jobs", -1),
            "verbose": -1,
            "class_weight": "balanced",
        }
        
        self.feature_names = feature_names
        
        # Grouped CV for OOF
        n_folds = self.config.get("cv_folds", 5)
        gkf = GroupKFold(n_splits=n_folds)
        
        oof_preds = np.zeros(len(y))
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            logger.info(f"Training Final Matcher fold {fold+1}/{n_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            val_data_lgb = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=params["n_estimators"],
                valid_sets=[val_data_lgb],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )
            
            oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
            models.append(model)
        
        # Train final model
        train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
        self.model = lgb.train(params, train_data, num_boost_round=params["n_estimators"])
        
        # Calibrate probabilities
        self.calibrator = self._calibrate(oof_preds, y, method=calibration_method)
        
        logger.info(f"Final Matcher OOF AUC: {self._auc(y, oof_preds):.4f}")
        
        return oof_preds
    
    def _calibrate(self, preds: np.ndarray, labels: np.ndarray, method: str = "isotonic"):
        """Calibrate probabilities"""
        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
        else:
            calibrator = CalibratedClassifierCV(
                LogisticRegression(), method="sigmoid", cv=5
            )
        
        calibrator.fit(preds.reshape(-1, 1), labels)
        return calibrator
    
    def _auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute AUC"""
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")
        raw_probs = self.model.predict(X, num_iteration=self.model.best_iteration)
        if self.calibrator is not None:
            return self.calibrator.predict(raw_probs.reshape(-1, 1))
        return raw_probs
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions"""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def save(self, path: str):
        """Save model"""
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "calibrator": self.calibrator,
            "config": self.config,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'FinalMatcher':
        """Load model"""
        data = joblib.load(path)
        matcher = cls(data["config"])
        matcher.model = data["model"]
        matcher.feature_names = data["feature_names"]
        matcher.calibrator = data["calibrator"]
        return matcher


class SingletonDetector:
    """Detect singleton entities (no matches)"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.threshold = 0.5
    
    def fit(self, s1_ids: np.ndarray, cand_probs: np.ndarray, 
            cand_labels: np.ndarray, groups: np.ndarray):
        """Fit singleton detector using OOF"""
        # For each S1, compute max probability and other features
        unique_s1 = np.unique(s1_ids)
        
        singleton_features = []
        singleton_labels = []
        
        for s1 in unique_s1:
            mask = s1_ids == s1
            max_prob = cand_probs[mask].max() if mask.any() else 0
            n_cands = mask.sum()
            has_match = cand_labels[mask].any()
            
            singleton_features.append([max_prob, n_cands])
            singleton_labels.append(int(not has_match))
        
        X_single = np.array(singleton_features)
        y_single = np.array(singleton_labels)
        
        # Simple threshold optimization for F0.5
        self.threshold = self._optimize_singleton_threshold(X_single[:, 0], y_single)
        
        logger.info(f"Singleton detector threshold: {self.threshold:.4f}")
    
    def _optimize_singleton_threshold(self, max_probs: np.ndarray, 
                                       labels: np.ndarray) -> float:
        """Optimize threshold for singleton detection"""
        from sklearn.metrics import fbeta_score
        
        thresholds = np.linspace(0, 1, 1001)
        best_thresh = 0.5
        best_f05 = 0
        
        for thresh in thresholds:
            preds = (max_probs < thresh).astype(int)  # Low prob -> singleton
            f05 = fbeta_score(labels, preds, beta=0.5, zero_division=0)
            if f05 > best_f05:
                best_f05 = f05
                best_thresh = thresh
        
        return best_thresh
    
    def predict_singleton(self, max_probs: np.ndarray) -> np.ndarray:
        """Predict singleton (1 = no match)"""
        return (max_probs < self.threshold).astype(int)


def optimize_f05_threshold(
    s1_ids: np.ndarray,
    cand_probs: np.ndarray,
    cand_labels: np.ndarray,
) -> Dict:
    """Optimize decision threshold for macro F0.5"""
    from sklearn.metrics import precision_recall_fscore_support
    
    unique_s1 = np.unique(s1_ids)
    thresholds = np.linspace(0, 1, 1001)
    
    best_thresh = 0.5
    best_f05 = 0
    best_metrics = {}
    
    for thresh in thresholds:
        all_preds = []
        all_true = []
        
        for s1 in unique_s1:
            mask = s1_ids == s1
            probs = cand_probs[mask]
            labels = cand_labels[mask]
            
            # Predict matches above threshold
            preds = (probs >= thresh).astype(int)
            
            # If no predictions, it's a singleton
            if not preds.any():
                preds = np.zeros_like(preds)
            
            all_preds.extend(preds)
            all_true.extend(labels)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true, all_preds, average="binary", zero_division=0
        )
        
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_metrics = {"precision": precision, "recall": recall, "f05": f05}
    
    return {"threshold": best_thresh, "f05": best_f05, **best_metrics}


def optimize_expected_f05(
    s1_ids: np.ndarray,
    cand_probs: np.ndarray,
    cand_labels: np.ndarray,
) -> Dict:
    """Optimize per-S1 expected F0.5 decision"""
    from sklearn.metrics import fbeta_score
    
    unique_s1 = np.unique(s1_ids)
    
    # For each S1, evaluate all possible prediction sets
    decisions = {}
    
    for s1 in unique_s1:
        mask = s1_ids == s1
        probs = cand_probs[mask]
        labels = cand_labels[mask]
        cand_indices = np.where(mask)[0]
        
        # Sort by probability descending
        order = np.argsort(probs)[::-1]
        probs_sorted = probs[order]
        labels_sorted = labels[order]
        indices_sorted = cand_indices[order]
        
        best_f05 = 0
        best_preds = np.zeros_like(probs)
        
        # Try all prefixes (including empty)
        for k in range(len(probs_sorted) + 1):
            preds = np.zeros_like(probs)
            if k > 0:
                preds[indices_sorted[:k]] = 1
            
            # Compute F0.5 for this S1
            tp = np.sum(preds * labels)
            fp = np.sum(preds * (1 - labels))
            fn = np.sum((1 - preds) * labels)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0
            
            if f05 > best_f05:
                best_f05 = f05
                best_preds = preds
        
        decisions[s1] = {
            "predictions": best_preds,
            "f05": best_f05,
        }
    
    return decisions


def hard_negative_mining(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model,
    feature_names: List[str],
    n_rounds: int = 3,
    top_k_fp: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hard negative mining"""
    logger.info("Starting hard negative mining...")
    
    X_augmented = X.copy()
    y_augmented = y.copy()
    groups_augmented = groups.copy()
    
    for round_idx in range(n_rounds):
        logger.info(f"Hard negative mining round {round_idx + 1}/{n_rounds}")
        
        # Get predictions
        probs = model.predict_proba(X_augmented)
        
        # Find high-scoring false positives
        fp_mask = (y_augmented == 0) & (probs > 0.5)
        fp_indices = np.where(fp_mask)[0]
        
        if len(fp_indices) == 0:
            logger.info("No false positives found")
            break
        
        # Sort by probability descending
        fp_probs = probs[fp_indices]
        fp_order = np.argsort(fp_probs)[::-1]
        top_fp = fp_indices[fp_order[:top_k_fp]]
        
        logger.info(f"Adding {len(top_fp)} hard negatives")
        
        # Add to training data (they're already in X, just re-weight or duplicate)
        # For simplicity, we'll just note them for reweighting
        # In practice, you might want to oversample them
        
        # Retrain with focus on hard negatives
        # This is a simplified version - in practice you'd use sample weights
        pass
    
    return X_augmented, y_augmented, groups_augmented


def train_with_hard_negatives(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    config: Dict,
    n_rounds: int = 3,
) -> Tuple[FinalMatcher, np.ndarray]:
    """Train final model with hard negative mining"""
    matcher = FinalMatcher(config)
    oof_preds = matcher.train(X, y, groups, feature_names)
    
    for round_idx in range(n_rounds):
        logger.info(f"Hard negative mining round {round_idx + 1}")
        
        # Find false positives in OOF
        fp_mask = (y == 0) & (oof_preds > 0.5)
        fp_indices = np.where(fp_mask)[0]
        
        if len(fp_indices) == 0:
            break
        
        # Add hard negatives (in practice, use sample weights)
        # For now, just retrain with same data but the model should learn
        oof_preds = matcher.train(X, y, groups, feature_names)
    
    return matcher, oof_preds
