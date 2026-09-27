"""Data loading utilities for Business Entity Resolution"""
import polars as pl
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def discover_dataset(root_dir: str = ".") -> Dict[str, Path]:
    """Automatically locate dataset files"""
    root = Path(root_dir)
    train_dir = root / "dataset" / "train"
    test_dir = root / "dataset" / "test"
    
    files = {
        "train_source1": train_dir / "train_source1.tsv",
        "train_source2": train_dir / "train_source2.tsv",
        "train_source3": train_dir / "train_source3.tsv",
        "train_ground_truth": train_dir / "train_ground_truth.tsv",
        "test_source1": test_dir / "test_source1.tsv",
        "test_source2": test_dir / "test_source2.tsv",
        "test_source3": test_dir / "test_source3.tsv",
    }
    
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    
    logger.info(f"Discovered dataset files:")
    for key, path in files.items():
        logger.info(f"  {key}: {path}")
    
    return files


def load_tsv(path: Path) -> pl.DataFrame:
    """Load TSV file with proper parsing"""
    return pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=10000,
        try_parse_dates=False,
        null_values=["", "NULL", "null", "NA", "na", "NaN", "nan"],
    )


def load_all_data(root_dir: str = ".") -> Dict[str, pl.DataFrame]:
    """Load all dataset files"""
    files = discover_dataset(root_dir)
    
    data = {}
    for key, path in files.items():
        logger.info(f"Loading {key} from {path}")
        data[key] = load_tsv(path)
        logger.info(f"  Shape: {data[key].shape}")
        logger.info(f"  Columns: {data[key].columns}")
    
    return data


def inspect_dataframe(df: pl.DataFrame, name: str) -> Dict:
    """Comprehensive dataframe inspection"""
    info = {
        "name": name,
        "shape": df.shape,
        "columns": df.columns,
        "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        "null_counts": df.null_count().to_dict(as_series=False),
        "unique_counts": {col: df[col].n_unique() for col in df.columns},
    }
    
    # String length distributions for string columns
    str_lengths = {}
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            lengths = df[col].str.len_chars().drop_nulls()
            if len(lengths) > 0:
                str_lengths[col] = {
                    "mean": lengths.mean(),
                    "median": lengths.median(),
                    "min": lengths.min(),
                    "max": lengths.max(),
                    "std": lengths.std(),
                }
    info["string_lengths"] = str_lengths
    
    # Country distribution
    if "country" in df.columns:
        info["country_distribution"] = df["country"].value_counts().to_dict(as_series=False)
    
    # Duplicate statistics
    info["exact_duplicates"] = df.n_unique() == df.height
    info["duplicate_rows"] = df.height - df.n_unique()
    
    return info


def print_data_inspection(info: Dict):
    """Pretty print data inspection results"""
    print(f"\n{'='*60}")
    print(f"DATA INSPECTION: {info['name']}")
    print(f"{'='*60}")
    print(f"Shape: {info['shape']}")
    print(f"Columns: {info['columns']}")
    print(f"Dtypes: {info['dtypes']}")
    print(f"Null counts: {info['null_counts']}")
    print(f"Unique counts: {info['unique_counts']}")
    print(f"String lengths: {info['string_lengths']}")
    print(f"Country distribution: {info.get('country_distribution', 'N/A')}")
    print(f"Exact duplicates: {not info['exact_duplicates']} (count: {info['duplicate_rows']})")
    print(f"{'='*60}\n")


def prepare_source_data(df: pl.DataFrame, source_name: str) -> pl.DataFrame:
    """Prepare source data with consistent column names"""
    df = df.clone()
    
    # Ensure entity_id column exists
    if "entity_id" not in df.columns:
        # Try to find ID column
        id_cols = [c for c in df.columns if "id" in c.lower() or "entity" in c.lower()]
        if id_cols:
            df = df.rename({id_cols[0]: "entity_id"})
        else:
            raise ValueError(f"No entity_id column found in {source_name}")
    
    # Add source identifier
    df = df.with_columns(pl.lit(source_name).alias("source"))
    
    return df


def parse_ground_truth(gt_df: pl.DataFrame) -> pl.DataFrame:
    """Parse ground truth into long format"""
    # ground truth has: source1_entity_id, matched_entity_ids (comma-separated)
    if "matched_entity_ids" not in gt_df.columns:
        raise ValueError("Ground truth must have 'matched_entity_ids' column")
    
    # Split comma-separated matches
    gt_long = gt_df.with_columns(
        pl.col("matched_entity_ids").str.split(",").alias("matches")
    ).explode("matches").rename({"matches": "matched_entity_id"})
    
    # Filter out empty matches
    gt_long = gt_long.filter(pl.col("matched_entity_id").str.len_chars() > 0)
    
    # Add source info for matched entities
    gt_long = gt_long.with_columns(
        pl.col("matched_entity_id").str.slice(0, 3).alias("matched_source")
    )
    
    return gt_long


def create_training_pairs(
    s1_df: pl.DataFrame,
    s2_df: pl.DataFrame,
    s3_df: pl.DataFrame,
    gt_df: pl.DataFrame,
) -> pl.DataFrame:
    """Create training pair table with labels"""
    # Combine S2 and S3
    candidates = pl.concat([s2_df, s3_df], how="vertical")
    
    # Get positive pairs from ground truth
    positives = gt_df.select(["source1_entity_id", "matched_entity_id"]).unique()
    positives = positives.with_columns(pl.lit(1).alias("label"))
    
    # For negative pairs, we'll generate them during blocking
    # This function returns the positive pairs for reference
    return positives


def get_data_stats(data: Dict[str, pl.DataFrame]) -> Dict:
    """Get comprehensive statistics for all datasets"""
    stats = {}
    for name, df in data.items():
        stats[name] = inspect_dataframe(df, name)
    return stats
