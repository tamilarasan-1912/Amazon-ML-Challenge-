"""Normalization library for Business Entity Resolution - country-agnostic."""

import re
import unicodedata
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass


LEGAL_SUFFIX_MAP: Dict[str, str] = {
    "corp": "corporation",
    "corporation": "corporation",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "ltd": "limited",
    "limited": "limited",
    "pvt": "private",
    "private": "private",
    "llc": "llc",
    "llp": "llp",
    "co": "company",
    "company": "company",
    "sarl": "sarl",
    "sas": "sas",
    "eurl": "eurl",
    "sa": "sa",
    "gmbh": "gmbh",
    "ag": "ag",
    "kg": "kg",
    "ohg": "ohg",
    "bv": "bv",
    "nv": "nv",
    "plc": "plc",
    "lp": "lp",
    "llp": "llp",
    "pc": "pc",
    "pa": "pa",
    "lllp": "lllp",
    "pllc": "pllc",
}

ADDRESS_ABBREV_MAP: Dict[str, str] = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "pkwy": "parkway",
    "hwy": "highway",
    "cir": "circle",
    "tr": "trail",
    "way": "way",
    "sq": "square",
    "ter": "terrace",
    "byp": "bypass",
    "exp": "expressway",
    "ext": "extension",
    "jct": "junction",
    "mt": "mount",
    "mtn": "mountain",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    "ap": "apartment",
    "apt": "apartment",
    "ste": "suite",
    "unit": "unit",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "rm": "room",
    "opp": "opposite",
    "near": "near",
    "beside": "beside",
    "behind": "behind",
}


def normalize_text(text: str) -> str:
    """Normalize text: NFKD, strip accents, lowercase, collapse whitespace, handle &."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Split normalized text into tokens."""
    return normalize_text(text).split()


def rewrite_legal_suffixes(tokens: List[str]) -> List[str]:
    """Rewrite legal suffix tokens to canonical form."""
    return [LEGAL_SUFFIX_MAP.get(tok, tok) for tok in tokens]


def strip_trailing_suffixes(tokens: List[str]) -> Tuple[List[str], List[str]]:
    """Strip trailing legal suffix tokens, return (core_tokens, stripped_suffixes)."""
    suffixes = []
    core = tokens[:]
    while core and core[-1] in LEGAL_SUFFIX_MAP:
        suffixes.append(core.pop())
    return core, suffixes


def normalize_name(name: str) -> Dict[str, object]:
    """Normalize business name, return variants and tokens."""
    norm = normalize_text(name)
    tokens = tokenize(name)
    tokens_rewritten = rewrite_legal_suffixes(tokens)
    core_tokens, stripped_suffixes = strip_trailing_suffixes(tokens_rewritten)
    core_norm = " ".join(core_tokens)
    full_norm = " ".join(tokens_rewritten)
    sorted_tokens = tuple(sorted(core_tokens))
    return {
        "raw": name,
        "normalized": norm,
        "full_normalized": full_norm,
        "suffix_stripped": core_norm,
        "tokens": tokens_rewritten,
        "core_tokens": core_tokens,
        "stripped_suffixes": stripped_suffixes,
        "sorted_tokens": sorted_tokens,
        "token_count": len(tokens_rewritten),
    }


def expand_address_abbrev(tokens: List[str]) -> List[str]:
    """Expand address abbreviation tokens."""
    return [ADDRESS_ABBREV_MAP.get(tok, tok) for tok in tokens]


def extract_digit_runs(text: str, min_len: int = 4, max_len: int = 6) -> List[str]:
    """Extract digit runs of length min_len to max_len (postal codes, etc.)."""
    return re.findall(rf"\d{{{min_len},{max_len}}}", text)


def extract_street_number(address: str) -> str:
    """Extract leading street number from address."""
    match = re.match(r"^(\d+)\s", address)
    return match.group(1) if match else ""


def extract_landmark_tokens(tokens: List[str]) -> List[str]:
    """Extract landmark reference tokens (near/opp/opposite/beside + following)."""
    landmarks = []
    landmark_triggers = {"near", "opp", "opposite", "beside", "behind", "next", "adjacent"}
    for i, tok in enumerate(tokens):
        if tok in landmark_triggers and i + 1 < len(tokens):
            landmarks.append(tok + "_" + tokens[i + 1])
    return landmarks


def normalize_address(address: str) -> Dict[str, object]:
    """Normalize address, extract structured components."""
    norm = normalize_text(address)
    tokens = tokenize(address)
    tokens_expanded = expand_address_abbrev(tokens)
    expanded_norm = " ".join(tokens_expanded)

    postal_codes = extract_digit_runs(address)
    street_number = extract_street_number(address)
    landmark_tokens = extract_landmark_tokens(tokens_expanded)

    sorted_tokens = tuple(sorted(tokens_expanded))

    return {
        "raw": address,
        "normalized": norm,
        "expanded_normalized": expanded_norm,
        "tokens": tokens_expanded,
        "postal_codes": postal_codes,
        "street_number": street_number,
        "landmark_tokens": landmark_tokens,
        "sorted_tokens": sorted_tokens,
        "token_count": len(tokens_expanded),
    }


@dataclass
class NormalizedRecord:
    entity_id: str
    country: str
    name: Dict
    address: Dict
    full_normalized: str  # combined name + address for TF-IDF

    def __post_init__(self):
        self.full_normalized = f"{self.name['full_normalized']} {self.address['expanded_normalized']}"


def normalize_record(entity_id: str, name: str, address: str, city: str, state: str, postal_code: str, country: str) -> NormalizedRecord:
    """Normalize a full record from source data."""
    full_address = " ".join(filter(None, [address, city, state, str(postal_code)]))
    name_norm = normalize_name(name)
    addr_norm = normalize_address(full_address)
    return NormalizedRecord(
        entity_id=entity_id,
        country=country if isinstance(country, str) else "",
        name=name_norm,
        address=addr_norm,
        full_normalized="",
    )


def normalize_dataframe(df) -> List[NormalizedRecord]:
    """Normalize all records in a dataframe."""
    records = []
    for _, row in df.iterrows():
        records.append(normalize_record(
            row["entity_id"],
            row["name"],
            row["address"],
            row.get("city", ""),
            row.get("state", ""),
            row.get("postal_code", ""),
            row.get("country", ""),
        ))
    return records