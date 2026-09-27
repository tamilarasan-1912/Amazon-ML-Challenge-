"""Pairwise feature computation for Business Entity Resolution."""

import re
from typing import Dict, List, Tuple, Set
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalize import NormalizedRecord, normalize_name, normalize_address


def jaro_winkler_sim(a: str, b: str) -> float:
    return fuzz.ratio(a, b) / 100.0 * 0.0 + fuzz.WRatio(a, b) / 100.0


def levenshtein_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1.0 - (Levenshtein.distance(a, b) / max(len(a), len(b)))


def token_set_jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    set_a, set_b = set(a_tokens), set(b_tokens)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def token_sort_ratio(a: str, b: str) -> float:
    a_sorted = " ".join(sorted(a.split()))
    b_sorted = " ".join(sorted(b.split()))
    return fuzz.ratio(a_sorted, b_sorted) / 100.0


def tfidf_char_ngram_cosine(texts: List[str]) -> np.ndarray:
    """Compute TF-IDF char n-gram cosine similarity matrix."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
    tfidf = vectorizer.fit_transform(texts)
    sim = (tfidf * tfidf.T).toarray()
    return sim


def exact_match_flag(a: str, b: str) -> int:
    return int(a.strip().lower() == b.strip().lower())


def contains_substring(a: str, b: str) -> int:
    a_low, b_low = a.lower(), b.lower()
    return int(a_low in b_low or b_low in a_low)


def first_token_match(a_tokens: List[str], b_tokens: List[str]) -> int:
    if not a_tokens or not b_tokens:
        return 0
    return int(a_tokens[0] == b_tokens[0])


def digit_run_overlap(a_digits: List[str], b_digits: List[str]) -> float:
    if not a_digits and not b_digits:
        return 1.0
    if not a_digits or not b_digits:
        return 0.0
    set_a, set_b = set(a_digits), set(b_digits)
    return len(set_a & set_b) / len(set_a | set_b)


def landmark_overlap(a_landmarks: List[str], b_landmarks: List[str]) -> int:
    if not a_landmarks and not b_landmarks:
        return 1
    if not a_landmarks or not b_landmarks:
        return 0
    return int(bool(set(a_landmarks) & set(b_landmarks)))


def component_overlap_jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    return token_set_jaccard(a_tokens, b_tokens)


def length_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return min(len(a), len(b)) / max(len(a), len(b))


def compute_name_features(s1_name: Dict, s23_name: Dict) -> Dict[str, float]:
    """Compute all name-based features."""
    feats = {}

    n1_full = s1_name["full_normalized"]
    n2_full = s23_name["full_normalized"]
    n1_core = s1_name["suffix_stripped"]
    n2_core = s23_name["suffix_stripped"]
    n1_raw = s1_name["raw"]
    n2_raw = s23_name["raw"]

    feats["name_jaro_winkler"] = jaro_winkler_sim(n1_full, n2_full)
    feats["name_levenshtein_ratio"] = levenshtein_ratio(n1_full, n2_full)
    feats["name_token_set_jaccard"] = token_set_jaccard(s1_name["tokens"], s23_name["tokens"])
    feats["name_token_sort_ratio"] = token_sort_ratio(n1_full, n2_full)
    feats["name_exact_match_raw"] = exact_match_flag(n1_raw, n2_raw)
    feats["name_exact_match_norm"] = exact_match_flag(n1_full, n2_full)
    feats["name_contains_substring"] = contains_substring(n1_full, n2_full)
    feats["name_first_token_match"] = first_token_match(s1_name["tokens"], s23_name["tokens"])
    feats["name_token_count_diff"] = abs(s1_name["token_count"] - s23_name["token_count"])

    # Suffix-stripped variants
    feats["name_core_jaro_winkler"] = jaro_winkler_sim(n1_core, n2_core)
    feats["name_core_levenshtein_ratio"] = levenshtein_ratio(n1_core, n2_core)
    feats["name_core_token_set_jaccard"] = token_set_jaccard(s1_name["core_tokens"], s23_name["core_tokens"])
    feats["name_core_token_sort_ratio"] = token_sort_ratio(n1_core, n2_core)
    feats["name_core_exact_match"] = exact_match_flag(n1_core, n2_core)

    return feats


def compute_address_features(s1_addr: Dict, s23_addr: Dict) -> Dict[str, float]:
    """Compute all address-based features."""
    feats = {}

    a1_full = s1_addr["expanded_normalized"]
    a2_full = s23_addr["expanded_normalized"]
    a1_raw = s1_addr["raw"]
    a2_raw = s23_addr["raw"]

    feats["addr_jaro_winkler"] = jaro_winkler_sim(a1_full, a2_full)
    feats["addr_levenshtein_ratio"] = levenshtein_ratio(a1_full, a2_full)
    feats["addr_token_set_jaccard"] = token_set_jaccard(s1_addr["tokens"], s23_addr["tokens"])
    feats["addr_token_sort_ratio"] = token_sort_ratio(a1_full, a2_full)
    feats["addr_exact_match_raw"] = exact_match_flag(a1_raw, a2_raw)
    feats["addr_exact_match_norm"] = exact_match_flag(a1_full, a2_full)
    feats["addr_contains_substring"] = contains_substring(a1_full, a2_full)
    feats["addr_first_token_match"] = first_token_match(s1_addr["tokens"], s23_addr["tokens"])
    feats["addr_token_count_diff"] = abs(s1_addr["token_count"] - s23_addr["token_count"])

    # Structured components
    feats["street_number_match"] = int(s1_addr["street_number"] == s23_addr["street_number"] and s1_addr["street_number"] != "")
    feats["postal_code_match"] = int(bool(set(s1_addr["postal_codes"]) & set(s23_addr["postal_codes"])))
    feats["digit_run_overlap"] = digit_run_overlap(s1_addr["postal_codes"], s23_addr["postal_codes"])
    feats["landmark_overlap"] = landmark_overlap(s1_addr["landmark_tokens"], s23_addr["landmark_tokens"])
    feats["addr_component_jaccard"] = component_overlap_jaccard(s1_addr["tokens"], s23_addr["tokens"])
    feats["addr_length_ratio"] = length_ratio(a1_full, a2_full)

    return feats


def compute_cross_features(name_feats: Dict, addr_feats: Dict,
                           s1_country: str, s23_country: str,
                           s23_source: str) -> Dict[str, float]:
    """Compute cross features."""
    feats = {}

    # Interactions
    feats["name_addr_jw_product"] = name_feats["name_jaro_winkler"] * addr_feats["addr_jaro_winkler"]
    feats["name_addr_tsj_product"] = name_feats["name_token_set_jaccard"] * addr_feats["addr_token_set_jaccard"]
    feats["name_core_addr_jw_product"] = name_feats["name_core_jaro_winkler"] * addr_feats["addr_jaro_winkler"]

    # Country equality (string equality only)
    feats["country_match"] = int(s1_country == s23_country and s1_country != "")

    # Source indicator
    feats["source_is_s2"] = int(s23_source == "S2")
    feats["source_is_s3"] = int(s23_source == "S3")

    return feats


def compute_all_features(s1_rec: NormalizedRecord,
                         s23_rec: NormalizedRecord,
                         tfidf_vectorizer: TfidfVectorizer = None) -> Dict[str, float]:
    """Compute all features for a candidate pair."""
    feats = {}

    name_feats = compute_name_features(s1_rec.name, s23_rec.name)
    addr_feats = compute_address_features(s1_rec.address, s23_rec.address)
    cross_feats = compute_cross_features(name_feats, addr_feats,
                                          s1_rec.country, s23_rec.country,
                                          "S2" if s23_rec.entity_id.startswith("S2") else "S3")

    feats.update(name_feats)
    feats.update(addr_feats)
    feats.update(cross_feats)

    # TF-IDF cosine on combined text (if vectorizer provided)
    if tfidf_vectorizer is not None:
        texts = [s1_rec.full_normalized, s23_rec.full_normalized]
        tfidf = tfidf_vectorizer.transform(texts)
        cos_sim = (tfidf[0] * tfidf[1].T).toarray()[0, 0]
        feats["tfidf_cosine"] = float(cos_sim)
    else:
        feats["tfidf_cosine"] = 0.0

    return feats


def build_feature_matrix(candidates: Dict[str, List],
                         s1_records: List[NormalizedRecord],
                         s23_records: List[NormalizedRecord]) -> Tuple[np.ndarray, List[str], List[Tuple[str, str]]]:
    """Build feature matrix for all candidate pairs."""
    s1_dict = {r.entity_id: r for r in s1_records}
    s23_dict = {r.entity_id: r for r in s23_records}

    # Collect all texts for TF-IDF
    all_texts = []
    pair_list = []
    for s1_id, cand_list in candidates.items():
        for cand in cand_list:
            s1_rec = s1_dict[s1_id]
            s23_rec = s23_dict[cand.s23_id]
            pair_list.append((s1_id, cand.s23_id))
            all_texts.append(s1_rec.full_normalized)
            all_texts.append(s23_rec.full_normalized)

    # Fit TF-IDF on all texts
    tfidf_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
    tfidf_vectorizer.fit(all_texts)

    # Compute features
    feature_rows = []
    for s1_id, s23_id in pair_list:
        s1_rec = s1_dict[s1_id]
        s23_rec = s23_dict[s23_id]
        feats = compute_all_features(s1_rec, s23_rec, tfidf_vectorizer)
        feature_rows.append(feats)

    # Convert to matrix
    feature_names = sorted(feature_rows[0].keys()) if feature_rows else []
    X = np.array([[row.get(fn, 0.0) for fn in feature_names] for row in feature_rows], dtype=np.float32)

    return X, feature_names, pair_list