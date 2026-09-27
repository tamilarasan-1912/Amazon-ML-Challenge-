"""Blocking and candidate generation for Business Entity Resolution."""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from metaphone import doublemetaphone
except ImportError:
    def doublemetaphone(text):
        return (text[:4], text[:4])

from .normalize import NormalizedRecord, normalize_dataframe
from ..config import BLOCKING_TOP_K, BLOCKING_MAX_CANDIDATES, SEED


@dataclass
class CandidatePair:
    s1_id: str
    s23_id: str
    source: str  # "S2" or "S3"
    score: float = 0.0  # blocking similarity score


def build_tfidf_candidates(s1_records: List[NormalizedRecord],
                           s23_records: List[NormalizedRecord],
                           top_k: int = BLOCKING_TOP_K) -> List[CandidatePair]:
    """TF-IDF char n-gram blocking on combined name+address."""
    s1_texts = [r.full_normalized for r in s1_records]
    s23_texts = [r.full_normalized for r in s23_records]

    if not s1_texts or not s23_texts:
        return []

    # Fit on all documents to avoid max_df issues with small S1 sets
    all_texts = s1_texts + s23_texts
    n_docs = len(all_texts)
    max_df_val = min(0.95, 1.0 - 1.0 / n_docs) if n_docs > 1 else 1.0

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_df=max_df_val,
        sublinear_tf=True,
    )
    tfidf_all = vectorizer.fit_transform(all_texts)
    tfidf_s1 = tfidf_all[:len(s1_texts)]
    tfidf_s23 = tfidf_all[len(s1_texts):]

    sim_matrix = cosine_similarity(tfidf_s1, tfidf_s23)

    candidates = []
    for i, s1_rec in enumerate(s1_records):
        sims = sim_matrix[i]
        if len(sims) == 0:
            continue
        k = min(top_k, len(sims))
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
        for j in top_indices:
            if sims[j] > 0:
                s23_rec = s23_records[j]
                candidates.append(CandidatePair(
                    s1_id=s1_rec.entity_id,
                    s23_id=s23_rec.entity_id,
                    source="S2" if s23_rec.entity_id.startswith("S2") else "S3",
                    score=float(sims[j])
                ))
    return candidates


def build_inverted_index_candidates(s1_records: List[NormalizedRecord],
                                     s23_records: List[NormalizedRecord]) -> List[CandidatePair]:
    """Inverted index on rare normalized name tokens."""
    token_to_s23 = defaultdict(list)
    token_counts = defaultdict(int)

    for rec in s23_records:
        for tok in set(rec.name["core_tokens"]):
            token_to_s23[tok].append(rec.entity_id)
            token_counts[tok] += 1

    total_s23 = len(s23_records)
    rare_threshold = max(1, total_s23 * 0.1)
    rare_tokens = {tok for tok, cnt in token_counts.items() if cnt <= rare_threshold}

    candidates = []
    for s1_rec in s1_records:
        matched_ids = set()
        for tok in set(s1_rec.name["core_tokens"]):
            if tok in rare_tokens:
                matched_ids.update(token_to_s23[tok])
        for s23_id in matched_ids:
            candidates.append(CandidatePair(
                s1_id=s1_rec.entity_id,
                s23_id=s23_id,
                source="S2" if s23_id.startswith("S2") else "S3",
                score=1.0
            ))
    return candidates


def build_address_anchor_candidates(s1_records: List[NormalizedRecord],
                                     s23_records: List[NormalizedRecord]) -> List[CandidatePair]:
    """Address anchors: same postal code OR same street_number+postal_code."""
    postal_to_s23 = defaultdict(list)
    street_postal_to_s23 = defaultdict(list)

    for rec in s23_records:
        for pc in rec.address["postal_codes"]:
            postal_to_s23[pc].append(rec.entity_id)
            if rec.address["street_number"]:
                street_postal_to_s23[(rec.address["street_number"], pc)].append(rec.entity_id)

    candidates = []
    for s1_rec in s1_records:
        matched_ids = set()
        for pc in s1_rec.address["postal_codes"]:
            matched_ids.update(postal_to_s23.get(pc, []))
            if s1_rec.address["street_number"]:
                matched_ids.update(street_postal_to_s23.get((s1_rec.address["street_number"], pc), []))
        for s23_id in matched_ids:
            candidates.append(CandidatePair(
                s1_id=s1_rec.entity_id,
                s23_id=s23_id,
                source="S2" if s23_id.startswith("S2") else "S3",
                score=1.0
            ))
    return candidates


def build_phonetic_candidates(s1_records: List[NormalizedRecord],
                               s23_records: List[NormalizedRecord]) -> List[CandidatePair]:
    """Phonetic key (double metaphone) of normalized name + first address token."""
    def get_phonetic_key(rec: NormalizedRecord) -> str:
        name_key = doublemetaphone(rec.name["normalized"])[0]
        addr_tokens = rec.address["tokens"]
        addr_key = doublemetaphone(addr_tokens[0])[0] if addr_tokens else ""
        return f"{name_key}|{addr_key}"

    phonetic_to_s23 = defaultdict(list)
    for rec in s23_records:
        phonetic_to_s23[get_phonetic_key(rec)].append(rec.entity_id)

    candidates = []
    for s1_rec in s1_records:
        key = get_phonetic_key(s1_rec)
        for s23_id in phonetic_to_s23.get(key, []):
            candidates.append(CandidatePair(
                s1_id=s1_rec.entity_id,
                s23_id=s23_id,
                source="S2" if s23_id.startswith("S2") else "S3",
                score=1.0
            ))
    return candidates


def build_sorted_token_candidates(s1_records: List[NormalizedRecord],
                                   s23_records: List[NormalizedRecord]) -> List[CandidatePair]:
    """Sorted-token signature exact match on name core tokens."""
    sig_to_s23 = defaultdict(list)
    for rec in s23_records:
        sig_to_s23[rec.name["sorted_tokens"]].append(rec.entity_id)

    candidates = []
    for s1_rec in s1_records:
        for s23_id in sig_to_s23.get(s1_rec.name["sorted_tokens"], []):
            candidates.append(CandidatePair(
                s1_id=s1_rec.entity_id,
                s23_id=s23_id,
                source="S2" if s23_id.startswith("S2") else "S3",
                score=1.0
            ))
    return candidates


def deduplicate_candidates(candidates: List[CandidatePair],
                           max_per_s1: int = BLOCKING_MAX_CANDIDATES) -> Dict[str, List[CandidatePair]]:
    """Deduplicate and cap candidates per S1 entity by max score."""
    by_s1 = defaultdict(dict)
    for c in candidates:
        key = (c.s1_id, c.s23_id)
        if key not in by_s1[c.s1_id] or c.score > by_s1[c.s1_id][key].score:
            by_s1[c.s1_id][key] = c

    result = {}
    for s1_id, cand_dict in by_s1.items():
        sorted_cands = sorted(cand_dict.values(), key=lambda x: -x.score)
        result[s1_id] = sorted_cands[:max_per_s1]
    return result


def generate_candidates(s1_records: List[NormalizedRecord],
                        s23_records: List[NormalizedRecord],
                        s1_country_map: Dict[str, str],
                        s23_country_map: Dict[str, str]) -> Dict[str, List[CandidatePair]]:
    """Generate candidate pairs using union of blocking schemes, restricted to same country."""
    s23_by_country = defaultdict(list)
    for rec in s23_records:
        s23_by_country[rec.country].append(rec)

    all_candidates = []

    for s1_rec in s1_records:
        country = s1_rec.country
        country_s23 = s23_by_country.get(country, [])

        if not country_s23:
            continue

        all_candidates.extend(build_tfidf_candidates([s1_rec], country_s23))
        all_candidates.extend(build_inverted_index_candidates([s1_rec], country_s23))
        all_candidates.extend(build_address_anchor_candidates([s1_rec], country_s23))
        all_candidates.extend(build_phonetic_candidates([s1_rec], country_s23))
        all_candidates.extend(build_sorted_token_candidates([s1_rec], country_s23))

    return deduplicate_candidates(all_candidates)


def compute_blocking_recall(candidates: Dict[str, List[CandidatePair]],
                            ground_truth: Dict[str, List[str]]) -> float:
    """Compute blocking recall ceiling: fraction of GT matches present in candidates."""
    total_gt = 0
    found_gt = 0
    for s1_id, gt_matches in ground_truth.items():
        if not gt_matches:
            continue
        total_gt += len(gt_matches)
        cand_ids = {c.s23_id for c in candidates.get(s1_id, [])}
        for m in gt_matches:
            if m in cand_ids:
                found_gt += 1
    return found_gt / total_gt if total_gt > 0 else 1.0