import os
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from .config import (
    MATCHING_OUTPUT,
    CANDIDATE_OUTPUT,
    MATCHING_THRESHOLD,
    MAX_CANDIDATES_PER_S1,
)
from .data_loader import load_sources, load_ground_truth
from .blocking import MultiKeyBlocker
from .features import FeatureEngineer
from .matcher import EntityMatcher
from .scorer import compute_fbeta, blocking_stats


def _s1_splits(
    s1_df: pd.DataFrame,
    seed: int = 42,
    val_frac: float = 0.2,
):
    n = len(s1_df)
    rng = np.random.RandomState(seed)

    idx = np.arange(n)
    rng.shuffle(idx)

    split = int(n * (1.0 - val_frac))

    tr_idx = idx[:split]
    va_idx = idx[split:]

    return (
        s1_df.iloc[tr_idx].reset_index(drop=True),
        s1_df.iloc[va_idx].reset_index(drop=True),
    )


def _build_pairs(
    s1_df: pd.DataFrame,
    candidates: Dict[str, List[str]],
    records_cand: Dict[str, Dict],
    gt: Dict[str, Set[str]] = None,
):
    pairs = []
    labels = []
    info = []

    for row in s1_df.to_dict("records"):

        s1_id = row["entity_id"]
        s1_rec = row

        cands = candidates.get(s1_id, [])

        gt_for_s1 = (
            gt.get(s1_id, set())
            if gt is not None
            else None
        )

        for cid in cands:

            c_rec = records_cand.get(cid)

            if c_rec is None:
                continue

            pairs.append(
                (
                    s1_rec,
                    c_rec,
                )
            )

            labels.append(
                1.0
                if (
                    gt_for_s1 is not None
                    and cid in gt_for_s1
                )
                else 0.0
            )

            info.append(
                (
                    s1_id,
                    cid,
                    gt_for_s1,
                )
            )

    return (
        pairs,
        np.array(labels, dtype=np.float32),
        info,
    )


def _save_predictions(
    s1_ids: List[str],
    s1_to_matches: Dict[str, Set[str]],
    path: str,
    col_name: str,
) -> None:

    rows = []

    for s1 in s1_ids:

        mid = sorted(
            s1_to_matches.get(
                s1,
                set(),
            )
        )

        rows.append(
            {
                "source1_entity_id": s1,
                col_name: ",".join(mid),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id",
            col_name,
        ],
    )

    parent = os.path.dirname(path)

    if parent:
        os.makedirs(
            parent,
            exist_ok=True,
        )

    df.to_csv(
        path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    print(
        f"[pipeline] Wrote "
        f"{len(df)} rows -> {path}"
    )


def run_pipeline(
    data_dir: str = None,
    train: bool = True,
    predict_test: bool = True,
    threshold: float = MATCHING_THRESHOLD,
    max_candidates: int = MAX_CANDIDATES_PER_S1,
) -> Dict:

    t0 = time.time()

    print("=" * 60)
    print("Business Entity Resolution Pipeline")
    print("=" * 60)

    # ============================================================
    # [1/6] LOAD TRAINING DATA
    # ============================================================

    print("\n[1/6] Loading data...")

    (
        use_train_s1,
        use_train_s2,
        use_train_s3,
    ) = load_sources(
        train=True,
        data_dir=data_dir,
    )

    gt = None

    try:

        gt = load_ground_truth(
            data_dir=data_dir
        )

        print(
            f"    Train: "
            f"S1={len(use_train_s1)}, "
            f"S2={len(use_train_s2)}, "
            f"S3={len(use_train_s3)}, "
            f"GT={len(gt)}"
        )

    except FileNotFoundError:

        print(
            "    train_ground_truth.tsv not found - "
            "skipping validation scoring"
        )

    train_s1_tr, train_s1_va = _s1_splits(
        use_train_s1,
        seed=42,
        val_frac=0.2,
    )

    print(
        f"    S1 split: "
        f"train={len(train_s1_tr)}, "
        f"val={len(train_s1_va)}"
    )

    # ============================================================
    # [2/6] BUILD BLOCKER
    # ============================================================

    print(
        "\n[2/6] Building blocker "
        "(fit on S2/S3)..."
    )

    blocker = MultiKeyBlocker(
        max_candidates=max_candidates
    )

    blocker.fit(
        use_train_s2,
        use_train_s3,
    )

    print(
        "    Building train candidates..."
    )

    cand_tr = blocker.transform(
        train_s1_tr
    )

    print(
        "    Building validation candidates..."
    )

    cand_va = blocker.transform(
        train_s1_va
    )

    # The blocker retains only candidate records,
    # not all 10M+ S2/S3 records.
    records_train_candidates = (
        blocker.candidate_records
    )

    # ============================================================
    # BLOCKING VALIDATION
    # ============================================================

    if gt is not None:

        cand_va_sets = {
            k: set(v)
            for k, v in cand_va.items()
        }

        va_s1_set = set(
            train_s1_va["entity_id"].tolist()
        )

        bstats = blocking_stats(
            gt,
            cand_va_sets,
            s1_subset=va_s1_set,
        )

        print(
            f"    [blocking-val] "
            f"recall_ceiling="
            f"{bstats['recall_ceiling']:.4f}, "
            f"avg_cand="
            f"{bstats['avg_candidates_per_s1']:.2f}, "
            f"found="
            f"{bstats['found_true_matches']}/"
            f"{bstats['total_true_matches']} "
            f"true matches"
        )

    # ============================================================
    # [3/6] BUILD TRAINING FEATURES
    # ============================================================

    print(
        "\n[3/6] Building training features..."
    )

    fe = FeatureEngineer()

    print(
        f"    Features: "
        f"{len(fe.feature_names)}"
    )

    pairs_tr, y_tr, info_tr = _build_pairs(
        train_s1_tr,
        cand_tr,
        records_train_candidates,
        gt,
    )

    pairs_va, y_va, info_va = _build_pairs(
        train_s1_va,
        cand_va,
        records_train_candidates,
        gt,
    )

    print(
        f"    Train pairs: "
        f"{len(pairs_tr)} "
        f"(pos={int(y_tr.sum())})"
    )

    print(
        f"    Val pairs:   "
        f"{len(pairs_va)} "
        f"(pos={int(y_va.sum())})"
    )

    X_tr = fe.transform_pairs(
        pairs_tr
    )

    X_va = fe.transform_pairs(
        pairs_va
    )

    # ============================================================
    # [4/6] TRAIN MATCHER
    # ============================================================

    print(
        "\n[4/6] Training matcher..."
    )

    matcher = EntityMatcher(
        threshold=threshold
    )

    matcher.train(
        X_tr,
        y_tr,
        val_X=X_va,
        val_y=y_va,
    )

    if len(info_va) > 0:

        matcher.tune_threshold(
            X_va,
            y_va,
            info_va,
            beta=0.5,
        )

    # ============================================================
    # VALIDATION SCORING
    # ============================================================

    f05 = None

    if (
        gt is not None
        and len(info_va) > 0
    ):

        print(
            "\n    [val scoring]"
        )

        proba_va = matcher.predict_proba(
            X_va
        )

        t = matcher.threshold

        pred_va = {}
        cand_va_out = {}

        va_s1_set = set(
            train_s1_va["entity_id"].tolist()
        )

        for s1_id in va_s1_set:

            pred_va[s1_id] = set()
            cand_va_out[s1_id] = set()

        for idx, (
            s1_id,
            cid,
            _,
        ) in enumerate(info_va):

            cand_va_out[
                s1_id
            ].add(cid)

            if (
                idx < len(proba_va)
                and proba_va[idx] >= t
            ):

                pred_va[
                    s1_id
                ].add(cid)

        f05, _ = compute_fbeta(
            gt,
            pred_va,
            beta=0.5,
            s1_subset=va_s1_set,
        )

        print(
            f"    Val F_0.5 = "
            f"{f05:.4f}"
        )

        bstats2 = blocking_stats(
            gt,
            cand_va_out,
            s1_subset=va_s1_set,
        )

        print(
            f"    Val blocking recall ceiling = "
            f"{bstats2['recall_ceiling']:.4f} "
            f"({bstats2['found_true_matches']}/"
            f"{bstats2['total_true_matches']})"
        )

        print(
            f"    Selected matching threshold = "
            f"{matcher.threshold:.6f}"
        )

    # ============================================================
    # TRAIN-ONLY MODE
    # ============================================================

    if not predict_test:

        return {
            "val_f05": f05,
            "threshold": matcher.threshold,
        }

    # ============================================================
    # [5/6] LOAD TEST SET
    # ============================================================

    print(
        "\n[5/6] Running on TEST set..."
    )

    try:

        (
            test_s1,
            test_s2,
            test_s3,
        ) = load_sources(
            train=False,
            data_dir=data_dir,
        )

        print(
            f"    Test: "
            f"S1={len(test_s1)}, "
            f"S2={len(test_s2)}, "
            f"S3={len(test_s3)}"
        )

    except FileNotFoundError as e:

        raise RuntimeError(
            "Test dataset could not be loaded. "
            "Check --data-dir and make sure these "
            "files exist: "
            "test/test_source1.tsv, "
            "test/test_source2.tsv, "
            "test/test_source3.tsv"
        ) from e

    # ============================================================
    # TEST BLOCKING
    # ============================================================

    print(
        "    Building candidates for test S1..."
    )

    blocker_test = MultiKeyBlocker(
        max_candidates=max_candidates
    )

    blocker_test.fit(
        test_s2,
        test_s3,
    )

    cand_test = blocker_test.transform(
        test_s1
    )

    records_test_candidates = (
        blocker_test.candidate_records
    )

    # ============================================================
    # TEST FEATURES
    # ============================================================

    print(
        "    Extracting features for "
        "test pairs..."
    )

    pairs_test = []
    test_info = []

    for row in test_s1.to_dict(
        "records"
    ):

        s1_id = row["entity_id"]

        for cid in cand_test.get(
            s1_id,
            [],
        ):

            c_rec = (
                records_test_candidates.get(
                    cid
                )
            )

            if c_rec is None:
                continue

            pairs_test.append(
                (
                    row,
                    c_rec,
                )
            )

            test_info.append(
                (
                    s1_id,
                    cid,
                )
            )

    print(
        f"    Test candidate pairs: "
        f"{len(pairs_test)}"
    )

    X_test = fe.transform_pairs(
        pairs_test
    )

    # ============================================================
    # TEST SCORING
    # ============================================================

    print(
        "    Scoring..."
    )

    proba_test = matcher.predict_proba(
        X_test
    )

    t = matcher.threshold

    # ============================================================
    # BUILD OUTPUT DICTIONARIES
    # ============================================================

    candidate_out = {}
    matches_out = {}

    # EVERY Source-1 test entity gets a row.

    for s1_id in test_s1[
        "entity_id"
    ].tolist():

        candidate_out[s1_id] = set()
        matches_out[s1_id] = set()

    for i, (
        s1_id,
        cid,
    ) in enumerate(test_info):

        candidate_out[
            s1_id
        ].add(cid)

        if (
            i < len(proba_test)
            and proba_test[i] >= t
        ):

            matches_out[
                s1_id
            ].add(cid)

    # ============================================================
    # [6/6] SAVE OUTPUTS
    # ============================================================

    print(
        "\n[6/6] Saving outputs..."
    )

    test_s1_ids = test_s1[
        "entity_id"
    ].tolist()

    _save_predictions(
        test_s1_ids,
        candidate_out,
        CANDIDATE_OUTPUT,
        "candidate_entity_ids",
    )

    _save_predictions(
        test_s1_ids,
        matches_out,
        MATCHING_OUTPUT,
        "matched_entity_ids",
    )

    # ============================================================
    # SUMMARY
    # ============================================================

    elapsed = time.time() - t0

    print(
        f"\n=== Done in "
        f"{elapsed:.1f}s ==="
    )

    n_cand = sum(
        len(v)
        for v in candidate_out.values()
    )

    n_match = sum(
        len(v)
        for v in matches_out.values()
    )

    print(
        f"Total candidates: "
        f"{n_cand}"
    )

    print(
        f"Total matches: "
        f"{n_match}"
    )

    print(
        f"Avg candidates/S1: "
        f"{n_cand / max(1, len(candidate_out)):.2f}"
    )

    print(
        f"Avg matches/S1:    "
        f"{n_match / max(1, len(matches_out)):.4f}"
    )

    return {
        "val_f05": f05,
        "threshold": matcher.threshold,
        "n_test_s1": len(test_s1_ids),
        "total_candidates": n_cand,
        "total_matches": n_match,
    }


if __name__ == "__main__":
    run_pipeline()