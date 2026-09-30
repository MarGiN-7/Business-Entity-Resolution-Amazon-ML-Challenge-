import os
import pandas as pd
from typing import Dict, List, Tuple


def load_tsv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"TSV not found: {path}")
    df = pd.read_csv(
    path,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    encoding="utf-8"
    )
    return df.fillna("")


def load_sources(train: bool = True,
                 data_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from .config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TEST_S1, TEST_S2, TEST_S3, TRAIN_DIR, TEST_DIR

    if train:
        s1_path, s2_path, s3_path = TRAIN_S1, TRAIN_S2, TRAIN_S3
    else:
        s1_path, s2_path, s3_path = TEST_S1, TEST_S2, TEST_S3

    if data_dir is not None:
        if train:
            base = data_dir if os.path.basename(data_dir) == "train" else os.path.join(data_dir, "train")
        else:
            base = data_dir if os.path.basename(data_dir) == "test" else os.path.join(data_dir, "test")
        prefix = "train_" if train else "test_"
        s1_path = os.path.join(base, f"{prefix}source1.tsv")
        s2_path = os.path.join(base, f"{prefix}source2.tsv")
        s3_path = os.path.join(base, f"{prefix}source3.tsv")

    s1 = load_tsv(s1_path)
    s2 = load_tsv(s2_path)
    s3 = load_tsv(s3_path)

    for df in (s1, s2, s3):
        for col in ("business_name", "business_address", "country"):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

    return s1, s2, s3


def load_ground_truth(
    path: str = None,
    data_dir: str = None
) -> Dict[str, set]:

    from .config import TRAIN_GT

    if path is None:

        if data_dir is not None:

            base = (
                data_dir
                if os.path.basename(
                    os.path.normpath(data_dir)
                ).lower() == "train"
                else os.path.join(
                    data_dir,
                    "train"
                )
            )

            path = os.path.join(
                base,
                "train_ground_truth.tsv"
            )

        else:
            path = TRAIN_GT

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Ground truth not found: {path}"
        )

    gt_df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    ).fillna("")

    gt = {}

    for _, row in gt_df.iterrows():

        s1_id = row[
            "source1_entity_id"
        ].strip()

        matched = row[
            "matched_entity_ids"
        ].strip()

        if matched:

            ids = {
                x.strip()
                for x in matched.split(",")
                if x.strip()
            }

        else:
            ids = set()

        gt[s1_id] = ids

    return gt

def concat_sources(s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([s2, s3], axis=0, ignore_index=True)
    return df.reset_index(drop=True)


def records_to_dict(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    result = {}
    cols = df.columns
    for _, row in df.iterrows():
        rid = row["entity_id"]
        result[rid] = {c: str(row[c]) if pd.notna(row[c]) else "" for c in cols}
    return result
