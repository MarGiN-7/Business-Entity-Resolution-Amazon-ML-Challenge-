# Business Entity Resolution — Amazon ML Challenge

A machine-learning project for identifying and matching the same business entity across multiple data sources.

This project implements a scalable, two-stage entity-resolution pipeline:

1. **Candidate Generation (Blocking):** Efficiently identifies likely matching records without comparing every possible pair.
2. **Machine Learning Matching:** Uses similarity-based features and a LightGBM classifier to determine whether two business records represent the same entity.

## Features

* Multi-key blocking using:

  * Business names
  * Addresses
  * Countries
  * Soundex
  * Character n-grams
* Text normalization for:

  * Punctuation
  * Abbreviations
  * Legal suffixes
  * Common typos
  * Word-order differences
* Similarity-based feature engineering for business names and addresses
* LightGBM binary classification with a heuristic fallback
* Validation utilities for submission files
* Candidate-pair and prediction-output generation
* Designed to avoid expensive all-to-all record comparisons
* Modular pipeline architecture for easier experimentation and extension

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
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MarGiN-7/Business-Entity-Resolution-Amazon-ML-Challenge-.git
cd Business-Entity-Resolution-Amazon-ML-Challenge-
```

### 2. Install Dependencies

Navigate to the project directory:

```bash
cd Business_Entity_Resolution/code/business_entity_resolution
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Run the complete entity-resolution pipeline with:

```bash
python run.py
```

The pipeline performs the following steps:

1. Loads the training and test TSV files.
2. Normalizes and preprocesses business information.
3. Generates candidate record pairs using blocking techniques.
4. Extracts similarity-based features.
5. Trains the LightGBM matching model.
6. Scores candidate pairs.
7. Generates entity-matching predictions and output files.

Generated files are stored in the configured output directories.

## Validating Submission Files

The project includes a validation utility for checking the generated submission and candidate files.

From the `Business_Entity_Resolution` directory, run:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

This validates the generated matching results against the expected test-data structure.

## Methodology

The entity-resolution system follows a two-stage approach to balance accuracy and computational efficiency.

### 1. Candidate Generation

Comparing every business record against every other record is computationally expensive. The blocking stage reduces the search space by generating candidate pairs using multiple keys, including:

* Normalized business names
* Normalized addresses
* Country information
* Soundex representations
* Character n-grams

Multiple blocking strategies help identify potential matches even when records contain spelling variations, formatting differences, abbreviations, or missing information.

### 2. Feature Engineering

For each candidate pair, the pipeline generates similarity features based on business attributes such as:

* Business-name similarity
* Address similarity
* Token-level similarity
* Character-level similarity
* Normalized string comparisons
* Word-order-independent comparisons
* Phonetic similarity

These features are used by the matching model to distinguish true entity matches from non-matches.

### 3. Machine Learning Matching

A **LightGBM binary classifier** is used to predict whether a candidate pair represents the same business entity.

When the machine-learning model cannot be used or does not produce a suitable prediction, the pipeline can fall back to heuristic-based matching logic.

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **LightGBM**
* **Text normalization**
* **String similarity metrics**
* **Soundex**
* **Character n-grams**
* **Machine-learning-based entity matching**

## Design Goals

The main goals of this project are:

* **Scalability:** Reduce the number of record comparisons through efficient blocking.
* **Robustness:** Handle spelling differences, abbreviations, punctuation, and formatting variations.
* **Accuracy:** Combine multiple similarity signals with machine-learning classification.
* **Modularity:** Keep data loading, blocking, feature engineering, matching, and scoring components separated.
* **Reproducibility:** Provide a structured pipeline for generating and validating outputs.

## Amazon ML Challenge

This project was developed as part of the **Amazon ML Challenge** and focuses on the problem of business entity resolution across multiple data sources.

The approach combines traditional record-linkage techniques with machine learning to efficiently identify records that refer to the same underlying business entity.
