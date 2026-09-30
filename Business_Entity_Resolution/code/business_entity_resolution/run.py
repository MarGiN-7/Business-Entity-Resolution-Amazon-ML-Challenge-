#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Business Entity Resolution Runner")
    parser.add_argument("--data-dir", default=None, help="Override dataset directory")
    parser.add_argument("--threshold", type=float, default=None, help="Override matching threshold")
    parser.add_argument("--max-candidates", type=int, default=None, help="Max candidates per S1")
    parser.add_argument("--train-only", action="store_true", help="Train only, skip test prediction")
    args = parser.parse_args()

    kwargs = {}
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir
    if args.threshold is not None:
        kwargs["threshold"] = args.threshold
    if args.max_candidates is not None:
        kwargs["max_candidates"] = args.max_candidates
    kwargs["predict_test"] = not args.train_only

    result = run_pipeline(**kwargs)
    print("\nSummary:", result)
