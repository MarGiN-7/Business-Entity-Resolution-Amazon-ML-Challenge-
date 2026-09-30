# Business Entity Resolution — Amazon ML Challenge

A machine-learning project for matching the same business entity across multiple data sources.

This solution uses a scalable two-stage entity-resolution pipeline:

1. **Candidate generation (blocking):** efficiently finds likely matching records without comparing every possible pair.
2. **Machine-learning matching:** uses similarity features and a LightGBM classifier to determine whether two business records represent the same entity.

## Features

- Multi-key blocking using business names, addresses, country, Soundex, and character n-grams
- Text normalization for punctuation, abbreviations, legal suffixes, typos, and word-order differences
- Similarity-based feature engineering for names and addresses
- LightGBM binary classification with a heuristic fallback
- Validation and submission-output utilities
- Designed to avoid expensive all-to-all record comparisons

## Project Structure

```text
Business_Entity_Resolution/
├── code/
│   └── business_entity_resolution/
│       ├── run.py
│       ├── requirements.txt
│       ├── README.md
│       └── src/
│           ├── blocking.py
│           ├── config.py
│           ├── data_loader.py
│           ├── embeddings.py
│           ├── features.py
│           ├── matcher.py
│           ├── pipeline.py
│           ├── scorer.py
│           └── text_utils.py
├── dataset/
│   ├── train/
│   └── test/
├── output/
├── submission_stage/
├── utils/
│   └── validate_submission.py
└── Documentation_template.md

Installation
Clone the repository:
git clone https://github.com/MarGiN-7/Business-Entity-Resolution-Amazon-ML-Challenge-.git
cd Business-Entity-Resolution-Amazon-ML-Challenge-
Install the dependencies:
cd Business_Entity_Resolution/code/business_entity_resolution
pip install -r requirements.txt
Run the Pipeline
Run the complete entity-resolution pipeline:
python run.py
The pipeline reads the training and test TSV files, generates candidate pairs, trains the matching model, and creates prediction outputs.
Validate Submission Files
From the Business_Entity_Resolution folder, run:
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
Technologies Used
- Python
- Pandas
- NumPy
- LightGBM
- Text normalization
- String similarity metrics
- Soundex and character n-gram matching

