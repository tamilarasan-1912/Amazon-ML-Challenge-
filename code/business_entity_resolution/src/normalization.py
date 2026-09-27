"""Normalization module for Business Entity Resolution"""
import re
import unicodedata
from typing import List, Set, Dict, Optional, Tuple
import polars as pl
from rapidfuzz import fuzz, process


# Legal form normalization mappings
LEGAL_FORMS = {
    "incorporated": "inc",
    "incorporation": "inc",
    "corporation": "corp",
    "corp": "corp",
    "company": "co",
    "co": "co",
    "limited": "ltd",
    "ltd": "ltd",
    "ltd.": "ltd",
    "private": "pvt",
    "pvt": "pvt",
    "pvt.": "pvt",
    "plc": "plc",
    "llc": "llc",
    "l.l.c.": "llc",
    "l.l.c": "llc",
    "llp": "llp",
    "l.l.p.": "llp",
    "gmbh": "gmbh",
    "ag": "ag",
    "sa": "sa",
    "s.a.": "sa",
    "s.r.l.": "srl",
    "srl": "srl",
    "bv": "bv",
    "b.v.": "bv",
    "nv": "nv",
    "n.v.": "nv",
    "kg": "kg",
    "ohg": "ohg",
    "ug": "ug",
    "k.k.": "kk",
    "kk": "kk",
}

# Common abbreviations
ABBREVIATIONS = {
    "street": "st",
    "st": "st",
    "avenue": "ave",
    "ave": "ave",
    "road": "rd",
    "rd": "rd",
    "drive": "dr",
    "dr": "dr",
    "lane": "ln",
    "ln": "ln",
    "boulevard": "blvd",
    "blvd": "blvd",
    "court": "ct",
    "ct": "ct",
    "place": "pl",
    "pl": "pl",
    "highway": "hwy",
    "hwy": "hwy",
    "parkway": "pkwy",
    "pkwy": "pkwy",
    "circle": "cir",
    "cir": "cir",
    "way": "way",
    "terrace": "ter",
    "ter": "ter",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
    "suite": "ste",
    "ste": "ste",
    "apartment": "apt",
    "apt": "apt",
    "unit": "unit",
    "floor": "fl",
    "fl": "fl",
    "building": "bldg",
    "bldg": "bldg",
    "department": "dept",
    "dept": "dept",
    "associates": "assoc",
    "assoc": "assoc",
    "international": "intl",
    "intl": "intl",
    "national": "natl",
    "natl": "natl",
    "global": "glbl",
    "technologies": "tech",
    "technology": "tech",
    "tech": "tech",
    "solutions": "sol",
    "solution": "sol",
    "services": "svcs",
    "service": "svc",
    "systems": "sys",
    "system": "sys",
    "industries": "ind",
    "industry": "ind",
    "manufacturing": "mfg",
    "mfg": "mfg",
    "corporation": "corp",
    "corp": "corp",
    "incorporated": "inc",
    "inc": "inc",
    "enterprises": "ent",
    "enterprise": "ent",
    "holdings": "hldgs",
    "holding": "hldg",
    "group": "grp",
    "grp": "grp",
    "partners": "prtnrs",
    "partner": "prtnr",
    "associates": "assoc",
    "assoc": "assoc",
}


def unicode_normalize(text: str) -> str:
    """Unicode NFC normalization"""
    if text is None:
        return ""
    return unicodedata.normalize("NFC", str(text))


def lowercase(text: str) -> str:
    """Lowercase conversion"""
    if text is None:
        return ""
    return str(text).lower()


def remove_punctuation(text: str, keep: str = "") -> str:
    """Remove punctuation except specified characters"""
    if text is None:
        return ""
    if keep:
        pattern = f"[^\\w\\s{re.escape(keep)}]"
    else:
        pattern = r"[^\w\s]"
    return re.sub(pattern, " ", str(text))


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace"""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_ampersand(text: str) -> str:
    """Normalize & and and"""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"\s+and\s+", " and ", text)
    return text


def normalize_legal_forms(text: str) -> str:
    """Normalize legal form suffixes"""
    if text is None:
        return ""
    words = str(text).split()
    if not words:
        return ""
    # Check last word for legal form
    last_word = words[-1].lower().rstrip(".,;:")
    if last_word in LEGAL_FORMS:
        words[-1] = LEGAL_FORMS[last_word]
    return " ".join(words)


def expand_abbreviations(text: str, abbrev_dict: Dict = None) -> str:
    """Expand common abbreviations"""
    if text is None:
        return ""
    if abbrev_dict is None:
        abbrev_dict = ABBREVIATIONS
    words = str(text).split()
    expanded = [abbrev_dict.get(w.lower().rstrip(".,;:"), w) for w in words]
    return " ".join(expanded)


def collapse_abbreviations(text: str, abbrev_dict: Dict = None) -> str:
    """Collapse words to abbreviations"""
    if text is None:
        return ""
    if abbrev_dict is None:
        abbrev_dict = {v: k for k, v in ABBREVIATIONS.items()}
    words = str(text).split()
    collapsed = [abbrev_dict.get(w.lower().rstrip(".,;:"), w) for w in words]
    return " ".join(collapsed)


def remove_repeated_chars(text: str, max_repeat: int = 2) -> str:
    """Remove excessive repeated characters"""
    if text is None:
        return ""
    pattern = f"(.)\\1{{{max_repeat},}}"
    return re.sub(pattern, r"\1" * max_repeat, str(text))


def transliterate_ascii(text: str) -> str:
    """Basic ASCII transliteration"""
    if text is None:
        return ""
    # Basic latin-1 supplement transliteration
    translit_map = {
        'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a', 'æ': 'ae',
        'ç': 'c', 'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
        'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
        'ñ': 'n', 'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o',
        'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
        'ý': 'y', 'ÿ': 'y', 'ß': 'ss',
        'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A', 'Å': 'A', 'Æ': 'AE',
        'Ç': 'C', 'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E',
        'Ì': 'I', 'Í': 'I', 'Î': 'I', 'Ï': 'I',
        'Ñ': 'N', 'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O', 'Ø': 'O',
        'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U',
        'Ý': 'Y', 'Ÿ': 'Y',
    }
    return "".join(translit_map.get(c, c) for c in str(text))


def tokenize(text: str) -> List[str]:
    """Tokenize text into words"""
    if text is None:
        return []
    text = normalize_whitespace(text)
    return text.split() if text else []


def sort_tokens(text: str) -> str:
    """Sort tokens alphabetically"""
    tokens = tokenize(text)
    return " ".join(sorted(tokens))


def token_set(text: str) -> str:
    """Unique tokens sorted"""
    tokens = tokenize(text)
    return " ".join(sorted(set(tokens)))


def compact_representation(text: str) -> str:
    """Compact representation: remove spaces and punctuation"""
    if text is None:
        return ""
    text = unicode_normalize(text)
    text = lowercase(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    return text.replace(" ", "")


def extract_house_number(address: str) -> Optional[str]:
    """Extract house/building number from address"""
    if address is None:
        return None
    # Match leading number
    match = re.match(r"^(\d+[a-zA-Z]?)", address.strip())
    if match:
        return match.group(1)
    return None


def extract_street_tokens(address: str) -> List[str]:
    """Extract street-related tokens"""
    if address is None:
        return []
    addr = lowercase(unicode_normalize(address))
    addr = remove_punctuation(addr)
    tokens = tokenize(addr)
    # Filter for street-like tokens
    street_keywords = {"st", "street", "ave", "avenue", "rd", "road", "dr", "drive",
                       "ln", "lane", "blvd", "boulevard", "ct", "court", "pl", "place",
                       "hwy", "highway", "pkwy", "parkway", "cir", "circle", "way",
                       "ter", "terrace", "n", "s", "e", "w", "ne", "nw", "se", "sw"}
    return [t for t in tokens if t in street_keywords or t.isdigit()]


def extract_postal_code(address: str, country: str = "") -> Optional[str]:
    """Extract postal/ZIP code"""
    if address is None:
        return None
    addr = str(address)
    # US ZIP
    if country.lower() in ["usa", "us", "united states"]:
        match = re.search(r"\b\d{5}(?:-\d{4})?\b", addr)
        if match:
            return match.group(0)
    # India PIN
    if country.lower() in ["india", "in"]:
        match = re.search(r"\b\d{6}\b", addr)
        if match:
            return match.group(0)
    # France postal code
    if country.lower() in ["france", "fr"]:
        match = re.search(r"\b\d{5}\b", addr)
        if match:
            return match.group(0)
    # Generic: 5-6 digits
    match = re.search(r"\b\d{5,6}\b", addr)
    if match:
        return match.group(0)
    return None


def normalize_country(country: str) -> str:
    """Normalize country string"""
    if country is None:
        return ""
    c = unicode_normalize(str(country)).lower().strip()
    c = remove_punctuation(c)
    c = normalize_whitespace(c)
    # Common mappings
    country_map = {
        "united states": "usa",
        "united states of america": "usa",
        "america": "usa",
        "us": "usa",
        "u.s.": "usa",
        "u.s.a.": "usa",
        "india": "india",
        "in": "india",
        "france": "france",
        "fr": "france",
        "uk": "uk",
        "united kingdom": "uk",
        "great britain": "uk",
        "canada": "canada",
        "ca": "canada",
        "germany": "germany",
        "de": "germany",
        "deutschland": "germany",
    }
    return country_map.get(c, c)


def normalize_name(text: str) -> Dict[str, str]:
    """Create multiple name representations"""
    if text is None:
        text = ""
    
    raw = str(text)
    unicode_norm = unicode_normalize(raw)
    lower = lowercase(unicode_norm)
    no_punct = remove_punctuation(lower)
    ws_norm = normalize_whitespace(no_punct)
    amp_norm = normalize_ampersand(ws_norm)
    legal_norm = normalize_legal_forms(amp_norm)
    abbrev_expanded = expand_abbreviations(legal_norm)
    no_repeats = remove_repeated_chars(abbrev_expanded)
    final_norm = normalize_whitespace(no_repeats)
    translit = transliterate_ascii(final_norm)
    tokens = tokenize(final_norm)
    sorted_toks = sort_tokens(final_norm)
    token_set_repr = token_set(final_norm)
    compact = compact_representation(final_norm)
    
    # Core name (without legal forms)
    core_tokens = [t for t in tokens if t not in LEGAL_FORMS.values()]
    core_name = " ".join(core_tokens)
    
    return {
        "name_raw": raw,
        "name_unicode": unicode_norm,
        "name_lower": lower,
        "name_no_punct": no_punct,
        "name_ws_norm": ws_norm,
        "name_amp_norm": amp_norm,
        "name_legal_norm": legal_norm,
        "name_abbrev_expanded": abbrev_expanded,
        "name_no_repeats": no_repeats,
        "name_normalized": final_norm,
        "name_transliterated": translit,
        "name_tokens": " ".join(tokens),
        "name_sorted_tokens": sorted_toks,
        "name_token_set": token_set_repr,
        "name_compact": compact,
        "name_core": core_name,
    }


def normalize_address(text: str, country: str = "") -> Dict[str, str]:
    """Create multiple address representations"""
    if text is None:
        text = ""
    
    raw = str(text)
    unicode_norm = unicode_normalize(raw)
    lower = lowercase(unicode_norm)
    no_punct = remove_punctuation(lower)
    ws_norm = normalize_whitespace(no_punct)
    abbrev_expanded = expand_abbreviations(ws_norm)
    no_repeats = remove_repeated_chars(abbrev_expanded)
    final_norm = normalize_whitespace(no_repeats)
    translit = transliterate_ascii(final_norm)
    tokens = tokenize(final_norm)
    sorted_toks = sort_tokens(final_norm)
    token_set_repr = token_set(final_norm)
    compact = compact_representation(final_norm)
    alpha_numeric = re.sub(r"[^a-z0-9]", "", final_norm)
    
    # Extract components
    house_num = extract_house_number(final_norm)
    street_tokens = extract_street_tokens(final_norm)
    postal = extract_postal_code(final_norm, country)
    
    return {
        "address_raw": raw,
        "address_unicode": unicode_norm,
        "address_lower": lower,
        "address_no_punct": no_punct,
        "address_ws_norm": ws_norm,
        "address_abbrev_expanded": abbrev_expanded,
        "address_no_repeats": no_repeats,
        "address_normalized": final_norm,
        "address_transliterated": translit,
        "address_tokens": " ".join(tokens),
        "address_sorted_tokens": sorted_toks,
        "address_token_set": token_set_repr,
        "address_compact": compact,
        "address_alpha_numeric": alpha_numeric,
        "address_house_number": house_num or "",
        "address_street_tokens": " ".join(street_tokens),
        "address_postal_code": postal or "",
    }


def normalize_record(record: Dict, is_source1: bool = False) -> Dict:
    """Normalize a complete record"""
    result = dict(record)
    
    # Normalize name
    name = record.get("name", "")
    name_norm = normalize_name(name)
    result.update(name_norm)
    
    # Normalize address
    address = record.get("address", "")
    country = record.get("country", "")
    addr_norm = normalize_address(address, country)
    result.update(addr_norm)
    
    # Normalize country
    result["country_normalized"] = normalize_country(country)
    result["country_raw"] = country
    
    # Other fields
    for field in ["city", "state", "postal_code"]:
        val = record.get(field, "")
        result[f"{field}_normalized"] = normalize_whitespace(lowercase(unicode_normalize(str(val))))
        result[f"{field}_raw"] = val
    
    return result


def normalize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Apply normalization to entire dataframe"""
    rows = df.to_dicts()
    normalized_rows = [normalize_record(row) for row in rows]
    return pl.DataFrame(normalized_rows)
