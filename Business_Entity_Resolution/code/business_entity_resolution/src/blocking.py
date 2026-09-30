from typing import Dict
import pandas as pd
import numpy as np

from .config import MAX_CANDIDATES_PER_S1
from .text_utils import canonical_name, canonical_address


class MultiKeyBlocker:

    def __init__(
        self,
        max_candidates: int = MAX_CANDIDATES_PER_S1,
    ):
        self.max_candidates = max_candidates
        self.s1_ids = []
        self.candidate_records = {}

        # Compact indexes:
        # key -> list of entity IDs
        self.name_index = {}
        self.address_index = {}
        self.country_name_prefix_index = {}
        self.country_address_prefix_index = {}

        # Only records that actually become candidates are retained.
        self.records = {}

    @staticmethod
    def _norm_name(x):
        try:
            return canonical_name(str(x or "")).strip().lower()
        except Exception:
            return str(x or "").strip().lower()

    @staticmethod
    def _norm_address(x):
        try:
            return canonical_address(str(x or "")).strip().lower()
        except Exception:
            return str(x or "").strip().lower()

    @staticmethod
    def _prefix(x, n=4):
        x = str(x or "").strip().lower()
        return x[:n]

    def _add(self, index, key, entity_id):
        if not key:
            return

        # Avoid giant duplicate lists for extremely common keys.
        bucket = index.get(key)

        if bucket is None:
            index[key] = [entity_id]
        elif len(bucket) < 200:
            bucket.append(entity_id)

    def fit(self, s2_df, s3_df):

        # Reset everything.
        self.name_index.clear()
        self.address_index.clear()
        self.country_name_prefix_index.clear()
        self.country_address_prefix_index.clear()
        self.records.clear()

        total = len(s2_df) + len(s3_df)
        processed = 0

        print(
            f"[blocker] building compact indexes for "
            f"{total:,} S2/S3 records..."
        )

        # Process source tables in chunks so temporary Python
        # objects don't explode.
        for df in (s2_df, s3_df):

            for start in range(
                0,
                len(df),
                100000,
            ):

                chunk = df.iloc[
                    start:start + 100000
                ]

                for row in chunk.itertuples(
                    index=False
                ):

                    entity_id = str(
                        row.entity_id
                    )

                    name = str(
                        row.business_name or ""
                    )

                    address = str(
                        row.business_address or ""
                    )

                    country = str(
                        row.country or ""
                    ).strip().lower()

                    name_norm = self._norm_name(
                        name
                    )

                    address_norm = self._norm_address(
                        address
                    )

                    self._add(
                        self.name_index,
                        name_norm,
                        entity_id,
                    )

                    self._add(
                        self.address_index,
                        address_norm,
                        entity_id,
                    )

                    if country and name_norm:
                        self._add(
                            self.country_name_prefix_index,
                            country + "|" +
                            self._prefix(name_norm),
                            entity_id,
                        )

                    if country and address_norm:
                        self._add(
                            self.country_address_prefix_index,
                            country + "|" +
                            self._prefix(address_norm),
                            entity_id,
                        )

                    # Store the record once.
                    self.records[entity_id] = {
                        "entity_id": entity_id,
                        "business_name": name,
                        "business_address": address,
                        "country": country,
                    }

                processed += len(chunk)

                if processed % 500000 == 0:
                    print(
                        f"[blocker] indexed "
                        f"{processed:,} / {total:,}"
                    )

        print(
            "[blocker] compact indexes ready: "
            f"name={len(self.name_index):,}, "
            f"address={len(self.address_index):,}, "
            f"name_prefix={len(self.country_name_prefix_index):,}, "
            f"address_prefix={len(self.country_address_prefix_index):,}"
        )

    def transform(self, s1_df):

        candidates = {}
        self.candidate_records = {}

        self.s1_ids = s1_df[
            "entity_id"
        ].tolist()

        total = len(s1_df)

        print(
            f"[blocker] generating candidates for "
            f"{total:,} S1 records..."
        )

        for i, row in enumerate(
            s1_df.itertuples(index=False),
            start=1,
        ):

            s1_id = str(row.entity_id)

            name = str(
                row.business_name or ""
            )

            address = str(
                row.business_address or ""
            )

            country = str(
                row.country or ""
            ).strip().lower()

            name_norm = self._norm_name(
                name
            )

            address_norm = self._norm_address(
                address
            )

            candidate_scores = {}

            # Exact normalized name.
            for rid in self.name_index.get(
                name_norm,
                []
            ):
                if rid != s1_id:
                    candidate_scores[rid] = max(
                        candidate_scores.get(rid, 0),
                        100,
                    )

            # Exact normalized address.
            for rid in self.address_index.get(
                address_norm,
                []
            ):
                if rid != s1_id:
                    candidate_scores[rid] = max(
                        candidate_scores.get(rid, 0),
                        90,
                    )

            # Country + name prefix.
            key = (
                country + "|" +
                self._prefix(name_norm)
            )

            for rid in self.country_name_prefix_index.get(
                key,
                []
            ):
                if rid != s1_id:
                    candidate_scores[rid] = max(
                        candidate_scores.get(rid, 0),
                        60,
                    )

            # Country + address prefix.
            key = (
                country + "|" +
                self._prefix(address_norm)
            )

            for rid in self.country_address_prefix_index.get(
                key,
                []
            ):
                if rid != s1_id:
                    candidate_scores[rid] = max(
                        candidate_scores.get(rid, 0),
                        50,
                    )

            ordered = sorted(
                candidate_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            ordered = ordered[
                :self.max_candidates
            ]

            ids = []

            for rid, score in ordered:

                rec = self.records.get(rid)

                if rec is None:
                    continue

                ids.append(rid)

                self.candidate_records[
                    rid
                ] = rec

            candidates[s1_id] = ids

            if i % 10000 == 0:
                print(
                    f"[blocker] processed S1 "
                    f"{i:,} / {total:,}"
                )

        counts = [
            len(v)
            for v in candidates.values()
        ]

        if counts:
            print(
                "[blocker] candidates per S1: "
                f"min={min(counts)}, "
                f"avg={np.mean(counts):.1f}, "
                f"max={max(counts)}"
            )

        print(
            f"[blocker] total candidate links: "
            f"{sum(counts):,}"
        )

        return candidates