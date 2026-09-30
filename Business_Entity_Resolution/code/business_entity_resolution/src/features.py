from typing import Dict, List, Tuple
import numpy as np
from .text_utils import (
    clean_text, tokenize, remove_suffixes, remove_stopwords, expand_abbrev,
    get_ngrams, get_tokens_set, canonical_name, canonical_address,
    levenshtein_ratio, jaro_winkler, jaccard, dice_coeff, overlap_coeff,
    containment, name_soundex_keys,
)
from .config import NGRAM_MIN, NGRAM_MAX


class FeatureEngineer:
    def __init__(self):
        self.feature_names: List[str] = []
        self._build_feature_names()

    def _build_feature_names(self) -> None:
        names = []
        for prefix in ("name", "addr"):
            names.extend([
                f"{prefix}_jaccard_tokens",
                f"{prefix}_dice_tokens",
                f"{prefix}_overlap_tokens",
                f"{prefix}_containment_a_b",
                f"{prefix}_containment_b_a",
                f"{prefix}_jaccard_ngrams",
                f"{prefix}_dice_ngrams",
                f"{prefix}_overlap_ngrams",
                f"{prefix}_levenshtein_canon",
                f"{prefix}_jarowinkler_canon",
                f"{prefix}_levenshtein_raw",
                f"{prefix}_jarowinkler_raw",
                f"{prefix}_common_tokens_cnt",
                f"{prefix}_common_tokens_ratio",
                f"{prefix}_first_token_match",
                f"{prefix}_length_diff",
                f"{prefix}_length_ratio",
                f"{prefix}_all_digits_match",
            ])
        names.extend([
            "country_match",
            "name_addr_cross_jaccard",
            "soundex_match_count",
            "name_3gram_overlap_top5",
            "addr_3gram_overlap_top5",
            "combined_quick_score",
            "either_empty_name",
            "either_empty_addr",
            "both_empty_name",
            "both_empty_addr",
            "name_exact_match",
            "addr_exact_match",
            "canon_name_exact",
            "canon_addr_exact",
        ])
        self.feature_names = names

    def _name_tokens(self, text: str):
        return set(t for t in remove_suffixes(tokenize(text)) if len(t) > 1)

    def _addr_tokens(self, text: str):
        toks = expand_abbrev(tokenize(text))
        toks = [t for t in toks if t.isdigit() or len(t) > 1]
        return set(toks)

    def _digit_tokens(self, text: str) -> set:
        return set(t for t in tokenize(text) if t.isdigit())

    def _extract_pair(self, rec_a: Dict, rec_b: Dict) -> np.ndarray:
        n_raw_a = rec_a.get("business_name", "")
        n_raw_b = rec_b.get("business_name", "")
        a_raw_a = rec_a.get("business_address", "")
        a_raw_b = rec_b.get("business_address", "")
        country_a = rec_a.get("country", "").strip().lower()
        country_b = rec_b.get("country", "").strip().lower()

        n_canon_a = canonical_name(n_raw_a)
        n_canon_b = canonical_name(n_raw_b)
        a_canon_a = canonical_address(a_raw_a)
        a_canon_b = canonical_address(a_raw_b)

        n_tok_a = self._name_tokens(n_raw_a)
        n_tok_b = self._name_tokens(n_raw_b)
        a_tok_a = self._addr_tokens(a_raw_a)
        a_tok_b = self._addr_tokens(a_raw_b)

        n_ng_a = get_ngrams(n_canon_a, NGRAM_MIN, NGRAM_MAX)
        n_ng_b = get_ngrams(n_canon_b, NGRAM_MIN, NGRAM_MAX)
        a_ng_a = get_ngrams(a_canon_a, NGRAM_MIN, NGRAM_MAX)
        a_ng_b = get_ngrams(a_canon_b, NGRAM_MIN, NGRAM_MAX)

        features = []

        def add_text_sim(raw_a, raw_b, canon_a, canon_b, tok_a, tok_b, ng_a, ng_b):
            features.append(jaccard(tok_a, tok_b))
            features.append(dice_coeff(tok_a, tok_b))
            features.append(overlap_coeff(tok_a, tok_b))
            features.append(containment(tok_a, tok_b))
            features.append(containment(tok_b, tok_a))
            features.append(jaccard(ng_a, ng_b))
            features.append(dice_coeff(ng_a, ng_b))
            features.append(overlap_coeff(ng_a, ng_b))
            features.append(levenshtein_ratio(canon_a, canon_b))
            features.append(jaro_winkler(canon_a, canon_b))
            features.append(levenshtein_ratio(clean_text(raw_a), clean_text(raw_b)))
            features.append(jaro_winkler(clean_text(raw_a), clean_text(raw_b)))
            common = len(tok_a & tok_b)
            features.append(common)
            denom = min(len(tok_a), len(tok_b))
            features.append(common / denom if denom else 0.0)
            first_a = next(iter(tok_a), "")
            first_b = next(iter(tok_b), "")
            features.append(1.0 if first_a and first_a == first_b else 0.0)
            la, lb = len(canon_a), len(canon_b)
            features.append(abs(la - lb) / max(la, lb, 1))
            features.append(min(la, lb) / max(la, lb, 1))
            digits_a = self._digit_tokens(raw_a)
            digits_b = self._digit_tokens(raw_b)
            features.append(1.0 if digits_a == digits_b and digits_a else 0.0)

        add_text_sim(n_raw_a, n_raw_b, n_canon_a, n_canon_b, n_tok_a, n_tok_b, n_ng_a, n_ng_b)
        add_text_sim(a_raw_a, a_raw_b, a_canon_a, a_canon_b, a_tok_a, a_tok_b, a_ng_a, a_ng_b)

        c_match = 1.0 if country_a and country_a == country_b else (0.5 if not country_a or not country_b else 0.0)
        features.append(c_match)

        cross = jaccard(n_tok_a & a_tok_a, n_tok_b & a_tok_b)
        features.append(cross)

        sx_a = set(name_soundex_keys(n_raw_a))
        sx_b = set(name_soundex_keys(n_raw_b))
        if sx_a or sx_b:
            features.append(len(sx_a & sx_b) / max(len(sx_a | sx_b), 1))
        else:
            features.append(0.0)

        top_n_ng_a = sorted(n_ng_a, key=len, reverse=True)[:5]
        top_n_ng_b = sorted(n_ng_b, key=len, reverse=True)[:5]
        features.append(jaccard(set(top_n_ng_a), set(top_n_ng_b)))

        top_a_ng_a = sorted(a_ng_a, key=len, reverse=True)[:5]
        top_a_ng_b = sorted(a_ng_b, key=len, reverse=True)[:5]
        features.append(jaccard(set(top_a_ng_a), set(top_a_ng_b)))

        quick = (
            0.35 * jaccard(n_tok_a, n_tok_b) +
            0.15 * jaccard(n_ng_a, n_ng_b) +
            0.20 * jaccard(a_tok_a, a_tok_b) +
            0.10 * jaccard(a_ng_a, a_ng_b) +
            0.20 * c_match
        )
        features.append(quick)

        features.append(1.0 if not n_tok_a or not n_tok_b else 0.0)
        features.append(1.0 if not a_tok_a or not a_tok_b else 0.0)
        features.append(1.0 if not n_tok_a and not n_tok_b else 0.0)
        features.append(1.0 if not a_tok_a and not a_tok_b else 0.0)

        features.append(1.0 if clean_text(n_raw_a) == clean_text(n_raw_b) and clean_text(n_raw_a) else 0.0)
        features.append(1.0 if clean_text(a_raw_a) == clean_text(a_raw_b) and clean_text(a_raw_a) else 0.0)
        features.append(1.0 if n_canon_a == n_canon_b and n_canon_a else 0.0)
        features.append(1.0 if a_canon_a == a_canon_b and a_canon_a else 0.0)

        return np.array(features, dtype=np.float32)

    def transform_pairs(self, pairs: List[Tuple[Dict, Dict]]) -> np.ndarray:
        if not pairs:
            return np.zeros((0, len(self.feature_names)), dtype=np.float32)
        X = np.zeros((len(pairs), len(self.feature_names)), dtype=np.float32)
        for i, (a, b) in enumerate(pairs):
            X[i] = self._extract_pair(a, b)
        return X
