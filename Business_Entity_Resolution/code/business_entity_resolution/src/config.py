import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = BASE_DIR
CODE_DIR = os.path.dirname(BASE_DIR)
SUBMISSION_CODE_DIR = os.path.dirname(CODE_DIR)
PROJECT_ROOT = os.path.dirname(SUBMISSION_CODE_DIR)

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

for d in [OUTPUT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

MATCHING_OUTPUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUTPUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

MATCHING_THRESHOLD = 0.55
MAX_CANDIDATES_PER_S1 = 50
NGRAM_MIN = 2
NGRAM_MAX = 3
