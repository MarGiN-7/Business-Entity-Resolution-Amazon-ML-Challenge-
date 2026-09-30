import os
import re
import csv
import duckdb

DATA = r"D:\dataset_Amazon\student_resource\dataset"
ROOT = os.path.dirname(os.path.abspath(__file__))

OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

S1 = os.path.join(DATA, "test", "test_source1.tsv")
S2 = os.path.join(DATA, "test", "test_source2.tsv")
S3 = os.path.join(DATA, "test", "test_source3.tsv")

MATCHING = os.path.join(OUT, "matching_results.tsv")
CANDIDATES = os.path.join(OUT, "candidate_pairs.tsv")


def norm_sql(col):
    return f"""
        lower(
            trim(
                regexp_replace(
                    coalesce({col}, ''),
                    '[^[:alnum:]]+',
                    ' ',
                    'g'
                )
            )
        )
    """


print("=" * 60)
print("EMERGENCY SUBMISSION GENERATOR")
print("=" * 60)

con = duckdb.connect()

con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='6GB'")

print("[1/5] Reading test data through DuckDB...")

con.execute(f"""
CREATE OR REPLACE VIEW s1 AS
SELECT
    entity_id,
    business_name,
    business_address,
    country,
    {norm_sql("business_name")} AS name_norm,
    {norm_sql("business_address")} AS address_norm,
    lower(trim(coalesce(country,''))) AS country_norm
FROM read_csv(
    '{S1}',
    delim='\\t',
    header=true,
    all_varchar=true
)
""")

con.execute(f"""
CREATE OR REPLACE VIEW s23 AS
SELECT
    entity_id,
    business_name,
    business_address,
    country,
    {norm_sql("business_name")} AS name_norm,
    {norm_sql("business_address")} AS address_norm,
    lower(trim(coalesce(country,''))) AS country_norm
FROM read_csv(
    '{S2}',
    delim='\\t',
    header=true,
    all_varchar=true
)
UNION ALL
SELECT
    entity_id,
    business_name,
    business_address,
    country,
    {norm_sql("business_name")} AS name_norm,
    {norm_sql("business_address")} AS address_norm,
    lower(trim(coalesce(country,''))) AS country_norm
FROM read_csv(
    '{S3}',
    delim='\\t',
    header=true,
    all_varchar=true
)
""")

print("[2/5] Finding exact normalized matches...")

# Strong matches:
#   A = same country + normalized business name
#   B = same country + normalized address
#
# We deliberately do NOT use fuzzy matching here.
# Precision is more important than speculative merges.

con.execute("""
CREATE OR REPLACE TEMP TABLE exact_matches AS

SELECT
    s.entity_id AS source1_entity_id,
    e.entity_id AS matched_entity_id,
    2 AS strength
FROM s1 s
JOIN s23 e
  ON s.country_norm <> ''
 AND s.name_norm <> ''
 AND s.country_norm = e.country_norm
 AND s.name_norm = e.name_norm

UNION

SELECT
    s.entity_id AS source1_entity_id,
    e.entity_id AS matched_entity_id,
    1 AS strength
FROM s1 s
JOIN s23 e
  ON s.country_norm <> ''
 AND s.address_norm <> ''
 AND s.country_norm = e.country_norm
 AND s.address_norm = e.address_norm
""")

# Deduplicate IDs per S1 and only retain strongest evidence.
con.execute("""
CREATE OR REPLACE TEMP TABLE final_matches AS
SELECT
    source1_entity_id,
    matched_entity_id
FROM (
    SELECT
        source1_entity_id,
        matched_entity_id,
        MAX(strength) AS strength
    FROM exact_matches
    GROUP BY source1_entity_id, matched_entity_id
)
""")

match_count = con.execute(
    "SELECT COUNT(*) FROM final_matches"
).fetchone()[0]

matched_s1 = con.execute(
    "SELECT COUNT(DISTINCT source1_entity_id) FROM final_matches"
).fetchone()[0]

total_s1 = con.execute(
    "SELECT COUNT(*) FROM s1"
).fetchone()[0]

print(f"    S1 rows:              {total_s1:,}")
print(f"    S1 with matches:      {matched_s1:,}")
print(f"    match links:          {match_count:,}")
print(f"    S1 without matches:   {total_s1 - matched_s1:,}")

print("[3/5] Writing matching_results.tsv...")

# The challenge expects every S1 entity exactly once.
# Empty list is represented as [].

with open(
    MATCHING,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(
        f,
        delimiter="\t",
        lineterminator="\n"
    )

    writer.writerow([
        "source1_entity_id",
        "matched_entity_ids"
    ])

    cur = con.execute("""
        SELECT
            s.entity_id,
            COALESCE(
                list(e.matched_entity_id ORDER BY e.matched_entity_id),
                []
            ) AS ids
        FROM s1 s
        LEFT JOIN final_matches e
          ON s.entity_id = e.source1_entity_id
        GROUP BY s.entity_id
        ORDER BY s.entity_id
    """)

    for sid, ids in cur.fetchall():

        if ids is None:
            ids = []

        # DuckDB may return [] or Python list.
        ids = list(ids)

        # Extra safety: deduplicate.
        ids = sorted(set(
            str(x) for x in ids
            if x is not None and str(x)
        ))

        writer.writerow([
            sid,
            "[" + ",".join(ids) + "]"
        ])

print(f"    wrote {MATCHING}")

print("[4/5] Writing candidate_pairs.tsv...")

# Candidate set = exact normalized candidates.
# One row per candidate link.
#
# If there are no candidates for an S1, it does not need
# an artificial S2/S3 ID.

with open(
    CANDIDATES,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(
        f,
        delimiter="\t",
        lineterminator="\n"
    )

    writer.writerow([
        "source1_entity_id",
        "candidate_entity_ids"
    ])

    cur = con.execute("""
        SELECT
            s.entity_id,
            COALESCE(
                list(e.matched_entity_id ORDER BY e.matched_entity_id),
                []
            ) AS ids
        FROM s1 s
        LEFT JOIN final_matches e
          ON s.entity_id = e.source1_entity_id
        GROUP BY s.entity_id
        ORDER BY s.entity_id
    """)

    for sid, ids in cur.fetchall():

        ids = sorted(set(
            str(x) for x in (ids or [])
            if x is not None and str(x)
        ))

        writer.writerow([
            sid,
            "[" + ",".join(ids) + "]"
        ])

print(f"    wrote {CANDIDATES}")

print("[5/5] Checking output counts...")

# Basic checks before we waste time packaging.

rows = con.execute(
    f"""
    SELECT COUNT(*)
    FROM read_csv(
        '{MATCHING}',
        delim='\\t',
        header=true,
        all_varchar=true
    )
    """
).fetchone()[0]

cand_rows = con.execute(
    f"""
    SELECT COUNT(*)
    FROM read_csv(
        '{CANDIDATES}',
        delim='\\t',
        header=true,
        all_varchar=true
    )
    """
).fetchone()[0]

print(f"    matching rows:   {rows:,}")
print(f"    candidate rows:  {cand_rows:,}")

if rows != total_s1:
    raise RuntimeError(
        f"BAD OUTPUT: expected {total_s1:,} "
        f"S1 rows, got {rows:,}"
    )

if cand_rows != total_s1:
    raise RuntimeError(
        f"BAD OUTPUT: expected {total_s1:,} "
        f"candidate rows, got {cand_rows:,}"
    )

print()
print("=" * 60)
print("OUTPUT GENERATION COMPLETE")
print("=" * 60)
print(MATCHING)
print(CANDIDATES)