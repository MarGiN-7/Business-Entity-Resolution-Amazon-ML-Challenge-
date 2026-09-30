import re
import unicodedata
from collections import Counter
from typing import List, Set, Tuple

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "llp", "lp", "plc", "gmbh", "ag", "bv", "nv", "sa", "sarl", "pte", "pvt",
    "private", "public", "co", "company", "associates", "assn", "group",
    "holdings", "holding", "trust", "industries", "international", "intl",
    "sys", "systems", "tech", "technologies", "solutions", "services",
    "consulting", "consultants", "enterprises", "enterprise", "business",
    "partners", "partner", "and", "&",
}

ADDR_ABBREV = {
    "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "rd": "road", "blvd": "boulevard", "dr": "drive", "ln": "lane",
    "way": "way", "ct": "court", "pl": "place", "sq": "square",
    "ter": "terrace", "trl": "trail", "hwy": "highway", "pkwy": "parkway",
    "cir": "circle", "bldg": "building", "ste": "suite", "fl": "floor",
    "ft": "fort", "mt": "mount", "n": "north", "s": "south",
    "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "no": "number", "nr": "near",
}

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "need", "dare", "ought", "used", "it", "its", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "new", "old", "near", "opp", "opposite", "behind", "beside", "next",
    "above", "below", "between", "among", "through", "across", "along",
    "around", "against", "during", "before", "after", "since", "until",
    "while", "about", "into", "within", "without", "along",
}


def normalize_unicode(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii", "ignore")
    return text


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return clean_text(text).split()


def remove_suffixes(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in LEGAL_SUFFIXES and len(t) > 1]


def expand_abbrev(tokens: List[str]) -> List[str]:
    return [ADDR_ABBREV.get(t, t) for t in tokens]


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS]


def canonical_name(text: str) -> str:
    toks = tokenize(text)
    toks = remove_suffixes(toks)
    toks = remove_stopwords(toks)
    return " ".join(sorted(set(toks)))


def canonical_address(text: str) -> str:
    toks = tokenize(text)
    toks = expand_abbrev(toks)
    toks = remove_stopwords(toks)
    nums = [t for t in toks if t.isdigit()]
    words = [t for t in toks if not t.isdigit() and len(t) > 1]
    return " ".join(nums + sorted(words))


def get_ngrams(text: str, n_min: int = 2, n_max: int = 3) -> Set[str]:
    toks = tokenize(text)
    if not toks:
        return set()
    ngrams = set()
    joined = " " + " ".join(toks) + " "
    for n in range(n_min, n_max + 1):
        for i in range(len(joined) - n + 1):
            ng = joined[i:i + n]
            if ng.strip():
                ngrams.add(ng)
    for tok in toks:
        if len(tok) >= 3:
            for n in range(n_min, min(n_max, len(tok)) + 1):
                for i in range(len(tok) - n + 1):
                    ngrams.add(tok[i:i + n])
    return ngrams


def get_tokens_set(text: str, remove_suf: bool = False, rm_stop: bool = False) -> Set[str]:
    toks = tokenize(text)
    if remove_suf:
        toks = remove_suffixes(toks)
    if rm_stop:
        toks = remove_stopwords(toks)
    return set(t for t in toks if len(t) > 1)


def soundex(token: str) -> str:
    if not token:
        return ""
    token = token.upper()
    first = token[0]
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }
    rest = [mapping.get(c, "0") for c in token[1:]]
    result = [first]
    prev = "0"
    for code in rest:
        if code != prev and code != "0":
            result.append(code)
        prev = code
    while len(result) < 4:
        result.append("0")
    return "".join(result[:4])


def name_soundex_keys(text: str) -> List[str]:
    toks = remove_suffixes(tokenize(text))
    toks = [t for t in toks if t.isalpha() and len(t) >= 3]
    return [soundex(t) for t in toks[:4]]


def levenshtein_ratio(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1
    prev = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        curr = [i] + [0] * len2
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[len2]
    return 1.0 - dist / max(len1, len2)


def jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    max_dist = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    jaro = (matches / len1 + matches / len2 +
            (matches - transpositions / 2) / matches) / 3.0
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * prefix_weight * (1.0 - jaro)


def jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def dice_coeff(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    return 2.0 * inter / (len(set_a) + len(set_b))


def overlap_coeff(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    return inter / min(len(set_a), len(set_b))


def containment(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a:
        return 1.0 if not set_b else 0.0
    inter = len(set_a & set_b)
    return inter / len(set_a)
