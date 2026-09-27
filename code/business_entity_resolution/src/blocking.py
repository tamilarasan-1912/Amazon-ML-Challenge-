"""Multi-channel blocking for Business Entity Resolution"""
import polars as pl
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)


class BlockingIndex:
    """Inverted index for exact matching"""
    def __init__(self):
        self.index: Dict[str, List[int]] = defaultdict(list)
        self.ids: List[str] = []
    
    def build(self, keys: List[str], ids: List[str]):
        """Build inverted index"""
        self.index.clear()
        self.ids = ids
        for i, key in enumerate(keys):
            if key and str(key).strip():
                self.index[str(key).strip()].append(i)
    
    def query(self, key: str) -> List[int]:
        """Query index for exact matches"""
        return self.index.get(str(key).strip(), [])


class TFIDFRetriever:
    """TF-IDF based retrieval for approximate matching"""
    def __init__(self, analyzer: str = "char_wb", ngram_range: Tuple[int, int] = (3, 5), 
                 max_features: int = 50000, top_k: int = 50):
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True,
            strip_accents="unicode",
        )
        self.top_k = top_k
        self.corpus_matrix: Optional[csr_matrix] = None
        self.ids: List[str] = []
        self.fitted = False
    
    def fit(self, texts: List[str], ids: List[str]):
        """Fit TF-IDF on corpus"""
        self.ids = ids
        self.corpus_matrix = self.vectorizer.fit_transform(texts)
        self.fitted = True
    
    def query(self, query_text: str, k: int = None) -> List[Tuple[int, float]]:
        """Retrieve top-k similar documents"""
        if not self.fitted:
            return []
        k = k or self.top_k
        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.corpus_matrix).flatten()
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        return [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0]


def build_exact_name_index(s2_df: pl.DataFrame, s3_df: pl.DataFrame, 
                           name_col: str = "name_normalized") -> BlockingIndex:
    """CHANNEL A: Exact normalized name index"""
    candidates = pl.concat([s2_df, s3_df], how="vertical")
    keys = candidates[name_col].fill_null("").to_list()
    ids = candidates["entity_id"].to_list()
    index = BlockingIndex()
    index.build(keys, ids)
    return index


def build_core_name_index(s2_df: pl.DataFrame, s3_df: pl.DataFrame,
                          name_col: str = "name_core") -> BlockingIndex:
    """CHANNEL B: Core name (legal suffix stripped) index"""
    candidates = pl.concat([s2_df, s3_df], how="vertical")
    keys = candidates[name_col].fill_null("").to_list()
    ids = candidates["entity_id"].to_list()
    index = BlockingIndex()
    index.build(keys, ids)
    return index


def build_char_tfidf_retriever(s2_df: pl.DataFrame, s3_df: pl.DataFrame,
                               name_col: str = "name_normalized",
                               ngram_range: Tuple[int, int] = (3, 5),
                               top_k: int = 50) -> TFIDFRetriever:
    """CHANNEL C: Character n-gram TF-IDF"""
    candidates = pl.concat([s2_df, s3_df], how="vertical")
    texts = candidates[name_col].fill_null("").to_list()
    ids = candidates["entity_id"].to_list()
    retriever = TFIDFRetriever(analyzer="char_wb", ngram_range=ngram_range, top_k=top_k)
    retriever.fit(texts, ids)
    return retriever


def build_word_tfidf_retriever(s2_df: pl.DataFrame, s3_df: pl.DataFrame,
                               name_col: str = "name_tokens",
                               ngram_range: Tuple[int, int] = (1, 2),
                               top_k: int = 50) -> TFIDFRetriever:
    """CHANNEL D: Word TF-IDF"""
    candidates = pl.concat([s2_df, s3_df], how="vertical")
    texts = candidates[name_col].fill_null("").to_list()
    ids = candidates["entity_id"].to_list()
    retriever = TFIDFRetriever(analyzer="word", ngram_range=ngram_range, top_k=top_k)
    retriever.fit(texts, ids)
    return retriever


def build_combined_tfidf_retriever(s2_df: pl.DataFrame, s3_df: pl.DataFrame,
                                   name_col: str = "name_normalized",
                                   address_col: str = "address_normalized",
                                   name_weight: float = 3.0,
                                   top_k: int = 50) -> TFIDFRetriever:
    """CHANNEL E: Combined name + weighted address"""
    candidates = pl.concat([s2_df, s3_df], how="vertical")
    
    # Create combined text with name weighted more
    name_texts = candidates[name_col].fill_null("").to_list()
    addr_texts = candidates[address_col].fill_null("").to_list()
    
    combined_texts = []
    for n, a in zip(name_texts, addr_texts):
        if n and a:
            combined = " ".join([n] * int(name_weight) + [a])
        elif n:
            combined = n
        else:
            combined = a
        combined_texts.append(combined)
    
    ids = candidates["entity_id"].to_list()
    retriever = TFIDFRetriever(analyzer="char_wb", ngram_range=(3, 5), top_k=top_k)
    retriever.fit(combined_texts, ids)
    return retriever


def build_address_retriever(s2_df: pl.DataFrame, s3_df: pl.DataFrame,
                            address_col: str = "address_normalized",
                            top_k: int = 50) -> TFIDFRetriever:
    """CHANNEL F: Address-based retrieval"""
    candidates = pl.concat([s2_df, s3_df], how="vertical")
    texts = candidates[address_col].fill_null("").to_list()
    ids = candidates["entity_id"].to_list()
    retriever = TFIDFRetriever(analyzer="char_wb", ngram_range=(3, 5), top_k=top_k)
    retriever.fit(texts, ids)
    return retriever


def build_exact_key_indices(s2_df: pl.DataFrame, s3_df: pl.DataFrame,
                            key_combinations: List[List[str]]) -> Dict[str, BlockingIndex]:
    """CHANNEL G: Exact structural keys"""
    candidates = pl.concat([s2_df, s3_df], how="vertical")
    indices = {}
    
    for combo in key_combinations:
        key_name = "_".join(combo)
        # Build composite key
        keys = []
        for row in candidates.iter_rows(named=True):
            parts = []
            for col in combo:
                val = row.get(col, "")
                if val is not None and str(val).strip():
                    parts.append(str(val).strip())
            keys.append("|".join(parts) if parts else "")
        
        ids = candidates["entity_id"].to_list()
        index = BlockingIndex()
        index.build(keys, ids)
        indices[key_name] = index
    
    return indices


def build_token_rescue_index(s2_df: pl.DataFrame, s3_df: pl.DataFrame,
                             name_col: str = "name_tokens") -> BlockingIndex:
    """CHANNEL H: Token/phonetic rescue - first significant tokens"""
    candidates = pl.concat([s2_df, s3_df], how="vertical")
    keys = []
    for row in candidates.iter_rows(named=True):
        tokens = row.get(name_col, "").split()
        # First 2 significant tokens (length > 2)
        sig_tokens = [t for t in tokens if len(t) > 2][:2]
        keys.append(" ".join(sig_tokens) if sig_tokens else "")
    
    ids = candidates["entity_id"].to_list()
    index = BlockingIndex()
    index.build(keys, ids)
    return index


def build_reverse_indices(s1_df: pl.DataFrame, s2_df: pl.DataFrame, s3_df: pl.DataFrame,
                          name_col: str = "name_normalized") -> Tuple[BlockingIndex, BlockingIndex]:
    """CHANNEL I: Reverse retrieval indices (S2/S3 -> S1)"""
    # S2/S3 -> S1
    s23_df = pl.concat([s2_df, s3_df], how="vertical")
    keys_23 = s23_df[name_col].fill_null("").to_list()
    ids_23 = s23_df["entity_id"].to_list()
    index_23_to_1 = BlockingIndex()
    index_23_to_1.build(keys_23, ids_23)
    
    # S1 -> S2/S3
    keys_1 = s1_df[name_col].fill_null("").to_list()
    ids_1 = s1_df["entity_id"].to_list()
    index_1_to_23 = BlockingIndex()
    index_1_to_23.build(keys_1, ids_1)
    
    return index_1_to_23, index_23_to_1


def retrieve_candidates_multi_channel(
    s1_df: pl.DataFrame,
    s2_df: pl.DataFrame,
    s3_df: pl.DataFrame,
    config: Dict,
    name_col: str = "name_normalized",
    core_name_col: str = "name_core",
    address_col: str = "address_normalized",
    tokens_col: str = "name_tokens",
) -> pl.DataFrame:
    """Multi-channel candidate retrieval"""
    all_candidates = []
    
    # Build all indices
    logger.info("Building blocking indices...")
    
    # CHANNEL A: Exact normalized name
    exact_name_idx = build_exact_name_index(s2_df, s3_df, name_col)
    
    # CHANNEL B: Core name
    core_name_idx = build_core_name_index(s2_df, s3_df, core_name_col)
    
    # CHANNEL C: Char n-gram TF-IDF
    char_tfidf = build_char_tfidf_retriever(s2_df, s3_df, name_col, 
                                            ngram_range=(3, 5), top_k=config.get("top_k", 50))
    
    # CHANNEL D: Word TF-IDF
    word_tfidf = build_word_tfidf_retriever(s2_df, s3_df, tokens_col,
                                            ngram_range=(1, 2), top_k=config.get("top_k", 50))
    
    # CHANNEL E: Combined name + address
    combined_tfidf = build_combined_tfidf_retriever(s2_df, s3_df, name_col, address_col,
                                                     top_k=config.get("top_k", 50))
    
    # CHANNEL F: Address retrieval
    address_tfidf = build_address_retriever(s2_df, s3_df, address_col,
                                             top_k=config.get("top_k", 50))
    
    # CHANNEL G: Exact structural keys
    key_combos = config.get("exact_key_combinations", [
        ["country_normalized", "name_normalized"],
        ["country_normalized", "postal_code_normalized"],
        ["postal_code_normalized", "address_house_number"],
        ["country_normalized", "address_house_number", "address_street_tokens"],
        ["name_core", "city_normalized"],
    ])
    exact_key_indices = build_exact_key_indices(s2_df, s3_df, key_combos)
    
    # CHANNEL H: Token rescue
    token_rescue_idx = build_token_rescue_index(s2_df, s3_df, tokens_col)
    
    # For each S1 entity, retrieve candidates
    logger.info(f"Retrieving candidates for {len(s1_df)} S1 entities...")
    
    for row in s1_df.iter_rows(named=True):
        s1_id = row["entity_id"]
        candidate_set = set()
        channel_hits = defaultdict(list)
        
        # CHANNEL A
        matches = exact_name_idx.query(row.get(name_col, ""))
        for idx in matches:
            candidate_set.add(idx)
            channel_hits["exact_name"].append(idx)
        
        # CHANNEL B
        matches = core_name_idx.query(row.get(core_name_col, ""))
        for idx in matches:
            candidate_set.add(idx)
            channel_hits["core_name"].append(idx)
        
        # CHANNEL C
        results = char_tfidf.query(row.get(name_col, ""), k=config.get("top_k", 50))
        for idx, score in results:
            candidate_set.add(idx)
            channel_hits["char_tfidf"].append((idx, score))
        
        # CHANNEL D
        results = word_tfidf.query(row.get(tokens_col, ""), k=config.get("top_k", 50))
        for idx, score in results:
            candidate_set.add(idx)
            channel_hits["word_tfidf"].append((idx, score))
        
        # CHANNEL E
        results = combined_tfidf.query(
            " ".join([row.get(name_col, "")] * 3 + [row.get(address_col, "")]),
            k=config.get("top_k", 50)
        )
        for idx, score in results:
            candidate_set.add(idx)
            channel_hits["combined_tfidf"].append((idx, score))
        
        # CHANNEL F
        results = address_tfidf.query(row.get(address_col, ""), k=config.get("top_k", 50))
        for idx, score in results:
            candidate_set.add(idx)
            channel_hits["address_tfidf"].append((idx, score))
        
        # CHANNEL G
        for key_name, idx in exact_key_indices.items():
            key_parts = []
            for col in key_name.split("_"):
                key_parts.append(str(row.get(col, "")))
            key = "_".join(key_parts)
            matches = idx.query(key)
            for m_idx in matches:
                candidate_set.add(m_idx)
                channel_hits[key_name].append(m_idx)
        
        # CHANNEL H
        matches = token_rescue_idx.query(" ".join([t for t in row.get(tokens_col, "").split() if len(t) > 2][:2]))
        for idx in matches:
            candidate_set.add(idx)
            channel_hits["token_rescue"].append(idx)
        
        # Convert to list of candidates
        candidates_df = pl.concat([s2_df, s3_df], how="vertical")
        for cand_idx in candidate_set:
            cand_row = candidates_df.row(cand_idx, named=True)
            all_candidates.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_row["entity_id"],
                "candidate_source": cand_row["source"],
                "channels": ",".join(channel_hits.keys()),
            })
    
    return pl.DataFrame(all_candidates) if all_candidates else pl.DataFrame({
        "source1_entity_id": [],
        "candidate_entity_id": [],
        "candidate_source": [],
        "channels": [],
    })


def compute_blocking_recall(
    candidates_df: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
) -> Dict:
    """Compute blocking recall metrics"""
    # Get positive pairs
    positives = ground_truth_df.select(["source1_entity_id", "matched_entity_id"]).unique()
    total_positives = len(positives)
    
    if total_positives == 0:
        return {"blocking_recall": 0.0, "recalled_positives": 0, "total_positives": 0}
    
    # Check how many positives are in candidates
    candidate_pairs = set(zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"]))
    positive_pairs = set(zip(positives["source1_entity_id"], positives["matched_entity_id"]))
    
    recalled = positive_pairs & candidate_pairs
    recall = len(recalled) / total_positives
    
    return {
        "blocking_recall": recall,
        "recalled_positives": len(recalled),
        "total_positives": total_positives,
        "missed_positives": list(positive_pairs - candidate_pairs)[:10],
    }


def compute_candidate_stats(candidates_df: pl.DataFrame) -> Dict:
    """Compute candidate set statistics"""
    if len(candidates_df) == 0:
        return {"avg_candidates": 0, "median_candidates": 0, "p95_candidates": 0, 
                "max_candidates": 0, "total_candidates": 0, "unique_s1": 0}
    
    counts = candidates_df.group_by("source1_entity_id").len()
    counts_arr = counts["len"].to_numpy()
    
    return {
        "avg_candidates": float(np.mean(counts_arr)),
        "median_candidates": float(np.median(counts_arr)),
        "p95_candidates": float(np.percentile(counts_arr, 95)),
        "max_candidates": int(np.max(counts_arr)),
        "min_candidates": int(np.min(counts_arr)),
        "total_candidates": len(candidates_df),
        "unique_s1": len(counts),
    }


def deduplicate_candidates(candidates_df: pl.DataFrame) -> pl.DataFrame:
    """Deduplicate candidate pairs"""
    return candidates_df.unique(subset=["source1_entity_id", "candidate_entity_id"])


def filter_candidates_by_channels(candidates_df: pl.DataFrame, 
                                   min_channels: int = 1) -> pl.DataFrame:
    """Filter candidates by minimum number of blocking channels"""
    if len(candidates_df) == 0:
        return candidates_df
    
    # Count channels per pair
    def count_channels(channels_str: str) -> int:
        if not channels_str:
            return 0
        return len(set(channels_str.split(",")))
    
    candidates_df = candidates_df.with_columns(
        pl.col("channels").map_elements(count_channels, return_dtype=pl.Int32).alias("n_channels")
    )
    
    return candidates_df.filter(pl.col("n_channels") >= min_channels).drop("n_channels")
