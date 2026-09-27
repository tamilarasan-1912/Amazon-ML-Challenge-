"""Configuration for the Business Entity Resolution pipeline"""
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


@dataclass
class PathConfig:
    """File paths configuration"""
    train_dir: str = "dataset/train"
    test_dir: str = "dataset/test"
    output_dir: str = "kaggle/working/output"
    models_dir: str = "models"
    reports_dir: str = "reports"
    features_dir: str = "features"
    experiments_dir: str = "reports/experiments"
    
    train_source1: str = "train_source1.tsv"
    train_source2: str = "train_source2.tsv"
    train_source3: str = "train_source3.tsv"
    train_ground_truth: str = "train_ground_truth.tsv"
    test_source1: str = "test_source1.tsv"
    test_source2: str = "test_source2.tsv"
    test_source3: str = "test_source3.tsv"
    
    matching_results: str = "matching_results.tsv"
    candidate_pairs: str = "candidate_pairs.tsv"
    run_config: str = "run_config.json"
    experiment_log: str = "experiment_log.json"


@dataclass
class BlockingConfig:
    """Blocking configuration"""
    tfidf_char_ngrams: List[int] = None
    tfidf_word_ngrams: List[int] = None
    top_k_candidates: List[int] = None
    exact_key_combinations: List[List[str]] = None
    
    def __post_init__(self):
        if self.tfidf_char_ngrams is None:
            self.tfidf_char_ngrams = [3, 4, 5]
        if self.tfidf_word_ngrams is None:
            self.tfidf_word_ngrams = [1, 2]
        if self.top_k_candidates is None:
            self.top_k_candidates = [5, 10, 20, 30, 50]
        if self.exact_key_combinations is None:
            self.exact_key_combinations = [
                ["country", "name_normalized"],
                ["country", "postal_code"],
                ["postal_code", "house_number"],
                ["country", "house_number", "street_token"],
                ["name_prefix", "postal_code"],
                ["core_name", "city"],
                ["street_number", "street_token"],
            ]


@dataclass
class ModelConfig:
    """Model configuration"""
    stage1_model: str = "lightgbm"
    final_model: str = "lightgbm"
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 8
    num_leaves: int = 63
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    n_jobs: int = -1
    calibration_method: str = "isotonic"  # "platt" or "isotonic"
    calibration_cv: int = 5


@dataclass
class ValidationConfig:
    """Validation configuration"""
    n_folds: int = 5
    grouped_by: str = "source1_entity_id"
    leave_country_out: bool = True
    difficult_experiment: bool = True
    random_state: int = 42


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    paths: PathConfig = None
    blocking: BlockingConfig = None
    model: ModelConfig = None
    validation: ValidationConfig = None
    random_seed: int = 42
    target_blocking_recall: float = 0.998
    max_candidates_per_s1: int = 100
    
    def __post_init__(self):
        if self.paths is None:
            self.paths = PathConfig()
        if self.blocking is None:
            self.blocking = BlockingConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.validation is None:
            self.validation = ValidationConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paths": asdict(self.paths),
            "blocking": asdict(self.blocking),
            "model": asdict(self.model),
            "validation": asdict(self.validation),
            "random_seed": self.random_seed,
            "target_blocking_recall": self.target_blocking_recall,
            "max_candidates_per_s1": self.max_candidates_per_s1,
        }
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'PipelineConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        config = cls()
        config.paths = PathConfig(**data.get("paths", {}))
        config.blocking = BlockingConfig(**data.get("blocking", {}))
        config.model = ModelConfig(**data.get("model", {}))
        config.validation = ValidationConfig(**data.get("validation", {}))
        config.random_seed = data.get("random_seed", 42)
        config.target_blocking_recall = data.get("target_blocking_recall", 0.998)
        config.max_candidates_per_s1 = data.get("max_candidates_per_s1", 100)
        return config


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration"""
    return PipelineConfig()
