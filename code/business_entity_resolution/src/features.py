"""Feature engineering for Business Entity Resolution"""
import polars as pl
import numpy as np
from rapidfuzz import fuzz, process
from typing import List, Dict, Tuple, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def jaro_winkler(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity"""
    if not s1 or not s2:
        return 0.0
    return fuzz.jaro_winkler(s1, s2)


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity"""
    if not s1 or not s2:
        return 0.0
    return fuzz.ratio(s1, s2) / 100.0


def token_jaccard(s1: str, s2: str) -> float:
    """Token Jaccard similarity"""
    if not s1 or not s2:
        return 0.0
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def token_overlap(s1: str, s2: str) -> int:
    """Count of overlapping tokens"""
    if not s1 or not s2:
        return 0
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    return len(tokens1 & tokens2)


def token_containment(s1: str, s2: str) -> float:
    """Token containment: s1 tokens in s2"""
    if not s1 or not s2:
        return 0.0
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1)


def token_set_ratio(s1: str, s2: str) -> float:
    """Token set ratio (RapidFuzz)"""
    if not s1 or not s2:
        return 0.0
    return fuzz.token_set_ratio(s1, s2) / 100.0


def token_sort_ratio(s1: str, s2: str) -> float:
    """Token sort ratio (RapidFuzz)"""
    if not s1 or not s2:
        return 0.0
    return fuzz.token_sort_ratio(s1, s2) / 100.0


def char_ngram_cosine(s1: str, s2: str, n: int = 3) -> float:
    """Character n-gram cosine similarity"""
    if not s1 or not s2:
        return 0.0
    
    def get_ngrams(text: str, n: int) -> Counter:
        text = f" {text} "
        return Counter(text[i:i+n] for i in range(len(text) - n + 1))
    
    c1 = get_ngrams(s1, n)
    c2 = get_ngrams(s2, n)
    
    if not c1 and not c2:
        return 1.0
    if not c1 or not c2:
        return 0.0
    
    # Cosine similarity
    dot = sum(c1[k] * c2[k] for k in c1 if k in c2)
    norm1 = np.sqrt(sum(v*v for v in c1.values()))
    norm2 = np.sqrt(sum(v*v for v in c2.values()))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def common_token_count(s1: str, s2: str) -> int:
    """Count of common tokens"""
    if not s1 or not s2:
        return 0
    return len(set(s1.split()) & set(s2.split()))


def rare_token_overlap(s1: str, s2: str, idf_dict: Dict[str, float] = None) -> float:
    """IDF-weighted rare token overlap"""
    if not s1 or not s2:
        return 0.0
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    common = tokens1 & tokens2
    if not common:
        return 0.0
    if idf_dict is None:
        return len(common)
    return sum(idf_dict.get(t, 1.0) for t in common)


def first_token_eq(s1: str, s2: str) -> int:
    """First token equality"""
    if not s1 or not s2:
        return 0
    t1 = s1.split()[0] if s1.split() else ""
    t2 = s2.split()[0] if s2.split() else ""
    return int(t1 == t2)


def first_two_tokens_eq(s1: str, s2: str) -> int:
    """First two tokens equality"""
    if not s1 or not s2:
        return 0
    t1 = " ".join(s1.split()[:2])
    t2 = " ".join(s2.split()[:2])
    return int(t1 == t2)


def last_token_eq(s1: str, s2: str) -> int:
    """Last token equality"""
    if not s1 or not s2:
        return 0
    t1 = s1.split()[-1] if s1.split() else ""
    t2 = s2.split()[-1] if s2.split() else ""
    return int(t1 == t2)


def length_diff(s1: str, s2: str) -> int:
    """Length difference"""
    return abs(len(s1) - len(s2))


def length_ratio(s1: str, s2: str) -> float:
    """Length ratio (min/max)"""
    if not s1 or not s2:
        return 0.0
    l1, l2 = len(s1), len(s2)
    return min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 0.0


def token_count_diff(s1: str, s2: str) -> int:
    """Token count difference"""
    return abs(len(s1.split()) - len(s2.split()))


def legal_form_agreement(s1: str, s2: str) -> int:
    """Check if legal forms agree"""
    legal_forms = {"inc", "corp", "llc", "ltd", "pvt", "plc", "co", "company",
                   "incorporated", "corporation", "limited", "private"}
    t1 = set(s1.split()) & legal_forms
    t2 = set(s2.split()) & legal_forms
    if not t1 and not t2:
        return 1  # Both have no legal form
    if not t1 or not t2:
        return 0  # Only one has legal form
    return int(t1 == t2)


def exact_eq(s1: str, s2: str) -> int:
    """Exact string equality"""
    return int(s1 == s2)


def partial_eq(s1: str, s2: str) -> int:
    """Partial string containment"""
    if not s1 or not s2:
        return 0
    return int(s1 in s2 or s2 in s1)


def numeric_token_overlap(s1: str, s2: str) -> int:
    """Overlap of numeric tokens"""
    if not s1 or not s2:
        return 0
    nums1 = {t for t in s1.split() if t.isdigit()}
    nums2 = {t for t in s2.split() if t.isdigit()}
    return len(nums1 & nums2)


def soft_token_match(s1: str, s2: str, idf_dict: Dict[str, float] = None) -> float:
    """Dictionary-free soft token matching with edit distance"""
    if not s1 or not s2:
        return 0.0
    tokens1 = s1.split()
    tokens2 = s2.split()
    
    if idf_dict is None:
        idf_dict = {}
    
    total_weight = 0.0
    matched_weight = 0.0
    
    for t1 in tokens1:
        w1 = idf_dict.get(t1, 1.0)
        total_weight += w1
        best_match = 0.0
        for t2 in tokens2:
            # Exact match
            if t1 == t2:
                best_match = 1.0
                break
            # Near edit match (Levenshtein)
            ratio = fuzz.ratio(t1, t2) / 100.0
            if ratio >= 0.8:
                best_match = max(best_match, ratio * 0.8)
            # Abbreviation compatible (one is prefix of other, min len 3)
            if len(t1) >= 3 and len(t2) >= 3:
                if t1.startswith(t2) or t2.startswith(t1):
                    best_match = max(best_match, 0.6)
        matched_weight += w1 * best_match
    
    if total_weight == 0:
        return 0.0
    return matched_weight / total_weight


def compute_idf(tokens_list: List[str]) -> Dict[str, float]:
    """Compute IDF weights from token corpus"""
    doc_count = len(tokens_list)
    token_docs = Counter()
    
    for tokens in tokens_list:
        unique_tokens = set(tokens.split())
        for t in unique_tokens:
            token_docs[t] += 1
    
    idf = {}
    for token, df in token_docs.items():
        idf[token] = np.log(doc_count / df + 1)
    
    return idf


def create_pair_features(
    s1_row: Dict,
    cand_row: Dict,
    idf_dict: Dict[str, float] = None,
) -> Dict[str, float]:
    """Create feature vector for a pair"""
    features = {}
    
    # Name features
    s1_name = s1_row.get("name_normalized", "")
    c_name = cand_row.get("name_normalized", "")
    s1_core = s1_row.get("name_core", "")
    c_core = cand_row.get("name_core", "")
    s1_tokens = s1_row.get("name_tokens", "")
    c_tokens = cand_row.get("name_tokens", "")
    s1_sorted = s1_row.get("name_sorted_tokens", "")
    c_sorted = cand_row.get("name_sorted_tokens", ")
    s1_token_set = s1_row.get("name_token_set", "")
    c_token_set = cand_row.get("name_token_set", "")
    s1_compact = s1_row.get("name_compact", "")
    c_compact = cand_row.get("name_compact", "")
    
    features["name_exact_eq"] = exact_eq(s1_name, c_name)
    features["name_core_eq"] = exact_eq(s1_core, c_core)
    features["name_jaro_winkler"] = jaro_winkler(s1_name, c_name)
    features["name_levenshtein_ratio"] = levenshtein_ratio(s1_name, c_name)
    features["name_token_jaccard"] = token_jaccard(s1_tokens, c_tokens)
    features["name_token_overlap"] = token_overlap(s1_tokens, c_tokens)
    features["name_token_containment"] = token_containment(s1_tokens, c_tokens)
    features["name_token_set_ratio"] = token_set_ratio(s1_tokens, c_tokens)
    features["name_token_sort_ratio"] = token_sort_ratio(s1_tokens, c_tokens)
    features["name_char_3gram_cosine"] = char_ngram_cosine(s1_name, c_name, 3)
    features["name_char_4gram_cosine"] = char_ngram_cosine(s1_name, c_name, 4)
    features["name_char_5gram_cosine"] = char_ngram_cosine(s1_name, c_name, 5)
    features["name_common_token_count"] = common_token_count(s1_tokens, c_tokens)
    features["name_rare_token_overlap"] = rare_token_overlap(s1_tokens, c_tokens, idf_dict)
    features["name_first_token_eq"] = first_token_eq(s1_tokens, c_tokens)
    features["name_first_two_token_eq"] = first_two_tokens_eq(s1_tokens, c_tokens)
    features["name_last_token_eq"] = last_token_eq(s1_tokens, c_tokens)
    features["name_length_diff"] = length_diff(s1_name, c_name)
    features["name_length_ratio"] = length_ratio(s1_name, c_name)
    features["name_token_count_diff"] = token_count_diff(s1_tokens, c_tokens)
    features["name_legal_form_agree"] = legal_form_agreement(s1_tokens, c_tokens)
    features["name_soft_token_match"] = soft_token_match(s1_tokens, c_tokens, idf_dict)
    
    # Address features
    s1_addr = s1_row.get("address_normalized", "")
    c_addr = cand_row.get("address_normalized", "")
    s1_addr_tokens = s1_row.get("address_tokens", "")
    c_addr_tokens = cand_row.get("address_tokens", "")
    s1_addr_compact = s1_row.get("address_compact", "")
    c_addr_compact = cand_row.get("address_compact", "")
    s1_postal = s1_row.get("address_postal_code", "")
    c_postal = cand_row.get("address_postal_code", "")
    s1_house = s1_row.get("address_house_number", "")
    c_house = cand_row.get("address_house_number", "")
    s1_street = s1_row.get("address_street_tokens", "")
    c_street = cand_row.get("address_street_tokens", "")
    s1_city = s1_row.get("city_normalized", "")
    c_city = cand_row.get("city_normalized", "")
    s1_state = s1_row.get("state_normalized", "")
    c_state = cand_row.get("state_normalized", "")
    
    features["address_exact_eq"] = exact_eq(s1_addr, c_addr)
    features["address_char_3gram_cosine"] = char_ngram_cosine(s1_addr, c_addr, 3)
    features["address_char_4gram_cosine"] = char_ngram_cosine(s1_addr, c_addr, 4)
    features["address_char_5gram_cosine"] = char_ngram_cosine(s1_addr, c_addr, 5)
    features["address_token_jaccard"] = token_jaccard(s1_addr_tokens, c_addr_tokens)
    features["address_token_overlap"] = token_overlap(s1_addr_tokens, c_addr_tokens)
    features["address_token_containment"] = token_containment(s1_addr_tokens, c_addr_tokens)
    features["address_levenshtein_ratio"] = levenshtein_ratio(s1_addr, c_addr)
    features["address_postal_eq"] = exact_eq(s1_postal, c_postal)
    features["address_postal_partial"] = partial_eq(s1_postal, c_postal)
    features["address_house_eq"] = exact_eq(s1_house, c_house)
    features["address_street_overlap"] = token_overlap(s1_street, c_street)
    features["address_city_eq"] = exact_eq(s1_city, c_city)
    features["address_state_eq"] = exact_eq(s1_state, c_state)
    features["address_length_ratio"] = length_ratio(s1_addr, c_addr)
    features["address_token_count_diff"] = token_count_diff(s1_addr_tokens, c_addr_tokens)
    features["address_numeric_overlap"] = numeric_token_overlap(s1_addr_tokens, c_addr_tokens)
    
    # Cross-field features
    features["name_addr_sim_product"] = features["name_jaro_winkler"] * features["address_char_3gram_cosine"]
    features["high_name_high_addr"] = int(features["name_jaro_winkler"] > 0.8 and features["address_char_3gram_cosine"] > 0.7)
    features["high_name_low_addr"] = int(features["name_jaro_winkler"] > 0.8 and features["address_char_3gram_cosine"] <= 0.3)
    features["low_name_high_addr"] = int(features["name_jaro_winkler"] <= 0.3 and features["address_char_3gram_cosine"] > 0.7)
    features["exact_name_exact_postal"] = int(features["name_exact_eq"] and features["address_postal_eq"])
    features["exact_postal_house_eq"] = int(features["address_postal_eq"] and features["address_house_eq"])
    
    # Country features
    s1_country = s1_row.get("country_normalized", "")
    c_country = cand_row.get("country_normalized", "")
    features["same_country"] = int(s1_country == c_country and s1_country != "")
    features["country_missing"] = int(s1_country == "" or c_country == "")
    features["country_conflict"] = int(s1_country != "" and c_country != "" and s1_country != c_country)
    
    # Name similarity conditioned on country
    if features["same_country"]:
        features["name_sim_same_country"] = features["name_jaro_winkler"]
        features["addr_sim_same_country"] = features["address_char_3gram_cosine"]
    else:
        features["name_sim_same_country"] = 0.0
        features["addr_sim_same_country"] = 0.0
    
    return features


def compute_blocking_features(
    s1_id: str,
    cand_id: str,
    candidate_pairs_df: pl.DataFrame,
) -> Dict[str, float]:
    """Compute blocking-related features for a pair"""
    features = {}
    
    # Get all candidates for this S1
    s1_candidates = candidate_pairs_df.filter(pl.col("source1_entity_id") == s1_id)
    candidate_count = len(s1_candidates)
    features["candidate_count"] = candidate_count
    
    # Get channels for this pair
    pair_row = s1_candidates.filter(pl.col("candidate_entity_id") == cand_id)
    if len(pair_row) > 0:
        channels_str = pair_row["channels"][0] if "channels" in pair_row.columns else ""
        channels = set(channels_str.split(",")) if channels_str else set()
        features["n_blocking_channels"] = len(channels)
        
        # Exact key indicators
        for ch in ["exact_name", "core_name", "exact_postal", "exact_house"]:
            features[f"channel_{ch}"] = int(ch in channels)
    else:
        features["n_blocking_channels"] = 0
        for ch in ["exact_name", "core_name", "exact_postal", "exact_house"]:
            features[f"channel_{ch}"] = 0
    
    # Candidates sharing same normalized name
    same_name = s1_candidates.filter(pl.col("candidate_entity_id") == cand_id)
    if len(same_name) > 0 and "name_normalized" in s1_candidates.columns:
        pass  # Would need join with candidate details
    
    return features


def create_difficulty_features(
    s1_row: Dict,
    all_candidates_df: pl.DataFrame,
    s1_id: str,
) -> Dict[str, float]:
    """Create difficulty features for an S1 entity"""
    features = {}
    
    s1_candidates = all_candidates_df.filter(pl.col("source1_entity_id") == s1_id)
    features["candidate_count"] = len(s1_candidates)
    
    # These would require joining with candidate details
    # Placeholder for now
    features["same_name_candidates"] = 0
    features["same_postal_candidates"] = 0
    features["same_address_token_candidates"] = 0
    
    return features


def build_feature_matrix(
    pairs_df: pl.DataFrame,
    s1_df: pl.DataFrame,
    s23_df: pl.DataFrame,
    idf_dict: Dict[str, float] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build feature matrix for all pairs"""
    # Create lookup dictionaries
    s1_lookup = {row["entity_id"]: row for row in s1_df.iter_rows(named=True)}
    s23_lookup = {row["entity_id"]: row for row in s23_df.iter_rows(named=True)}
    
    feature_list = []
    feature_names = None
    
    for pair in pairs_df.iter_rows(named=True):
        s1_id = pair["source1_entity_id"]
        cand_id = pair["candidate_entity_id"]
        
        s1_row = s1_lookup.get(s1_id, {})
        cand_row = s23_lookup.get(cand_id, {})
        
        features = create_pair_features(s1_row, cand_row, idf_dict)
        
        if feature_names is None:
            feature_names = sorted(features.keys())
        
        feature_list.append([features.get(name, 0.0) for name in feature_names])
    
    return np.array(feature_list, dtype=np.float32), feature_names


def compute_tfidf_similarity(
    texts1: List[str],
    texts2: List[str],
    analyzer: str = "char_wb",
    ngram_range: Tuple[int, int] = (3, 5),
) -> np.ndarray:
    """Compute TF-IDF cosine similarity between two lists of texts"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=50000,
    )
    
    all_texts = texts1 + texts2
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    n1 = len(texts1)
    sim_matrix = cosine_similarity(tfidf_matrix[:n1], tfidf_matrix[n1:])
    
    return np.diag(sim_matrix)
