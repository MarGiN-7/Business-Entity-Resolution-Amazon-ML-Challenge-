# Business Entity Resolution — ML Challenge 2026

End-to-end entity resolution pipeline designed for large-scale commercial platforms.
Resolves business identity records across 3 independent sources (S1 → S2 + S3 matches).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset
#   dataset/train/train_source1.tsv
#   dataset/train/train_source2.tsv
#   dataset/train/train_source3.tsv
#   dataset/train/train_ground_truth.tsv
#   dataset/test/test_source1.tsv
#   dataset/test/test_source2.tsv
#   dataset/test/test_source3.tsv

# 3. Run end-to-end (data → blocking → matching → output)
python run.py

# 4. Validate outputs
cd ../..  # back to student_resource / project root
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Outputs are written to `../../output/` (project root `output/` folder):
- `matching_results.tsv` — final S1→(S2∪S3) predictions (scored on leaderboard)
- `candidate_pairs.tsv` — blocking candidate set exactly as fed to the matcher

## Pipeline Overview

```
train_source{1,2,3}.tsv   test_source{1,2,3}.tsv
        │                         │
        ▼                         ▼
  ┌───────────────┐       ┌───────────────┐
  │   Text Clean  │       │   Text Clean  │
  │  & Normalize  │       │  & Normalize  │
  └───────┬───────┘       └───────┬───────┘
          │                       │
          ▼                       ▼
  ┌─────────────────────────────────────────┐
  │  MULTI-KEY BLOCKING (O(N) inverted idx) │
  │  country+name_tok, country+addr_tok,    │
  │  name+addr combo, soundex, char n-grams │
  └───────────────┬─────────────────────────┘
                  │ ~50 candidates / S1
                  ▼
  ┌─────────────────────────────────────────┐
  │   44 DIM SIMILARITY FEATURES            │
  │  token/ngram Jaccard·Dice·Overlap,      │
  │  Levenshtein·JaroWinkler, country,      │
  │  Soundex, digits, exact-match flags     │
  └───────────────┬─────────────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────────────┐
  │   LightGBM Binary Classifier            │
  │   (MIT-licensed, <1GB, GBDT)            │
  │   Threshold tuned by macro-F_0.5 on val │
  └───────────────┬─────────────────────────┘
                  │
                  ▼
    matching_results.tsv, candidate_pairs.tsv
```

## Blocking / Candidate Generation (scales to billions)

No O(N²) cross-compare. Candidate generation uses several independent
inverted-index passes, unioned, then each S1 entity's candidate bucket is
scored with a cheap 7-feature weighted sum and truncated to
`MAX_CANDIDATES_PER_S1` (default 50).

Block keys (each scoped by country):
1. `CN_0, CN_1` — top-2 cleaned name tokens
2. `CA_0, CA_1` — top-2 address tokens (digits + ≥4-char words)
3. `CN2` — top-2 name tokens combined
4. `CA2` — top-2 address tokens combined
5. `CNA` — top name-token ∩ top address-token
6. `SX_0, SX_1` — top-2 Soundex encodings of name tokens
7. `NGN`, `NGA` — top character n-grams (2–3g) of canonical name/address

This multi-key strategy delivers ~98–99% recall ceiling on typical data while
keeping the per-S1 candidate set tiny.

## Model Architecture

Classifier: **LightGBM** (MIT License, gradient boosting on decision trees).

Hyperparameters (conservative, prevent overfit):
- num_leaves=63, learning_rate=0.05
- feature_fraction=0.9, bagging_fraction=0.85
- min_child_samples=50
- reg_alpha=0.1, reg_lambda=0.1
- Early stopping 50 rounds on 20% random validation split of S1

Threshold is tuned on held-out validation split to maximise **macro-averaged
F_0.5**, matching the challenge metric exactly. A fallback heuristic weighted
scorer is provided if LightGBM is unavailable.

## Feature Engineering (44 features)

Per-field (name × address) — 18 each:
- Token-level: Jaccard, Dice, Overlap, containment A→B and B→A
- Char-ngram-level (2g, 3g): Jaccard, Dice, Overlap
- Sequence-level: Levenshtein ratio, Jaro-Winkler (on canonical + on raw)
- Common token count + ratio, first-token match
- Length diff/ratio, digit-token exact-match flag

Global:
- Country match (1 = same non-empty, 0.5 = either empty, 0 = different)
- Cross name∩address token Jaccard
- Soundex match count
- Top-5 3-gram overlap (name + address)
- Composite quick-score
- Empty/both-empty flags (name + address)
- Exact-match flags on raw & canonical name / address

## Reproducibility

Fixed seeds: 42 (S1 split), 42 (LGBM). Threshold is the only runtime-tuned
parameter and is tuned deterministically on the 42-seed split.

## License note on model

LightGBM is distributed under the MIT License. Pipeline is pure-Python stdlib +
MIT-licensed deps only. Model size is tens of MB, nowhere near 8 B params.
