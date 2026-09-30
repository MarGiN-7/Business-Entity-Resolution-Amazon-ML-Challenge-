import duckdb
from pathlib import Path
BASE = r"D:\dataset_Amazon\student_resource\dataset\test"
OUT = Path("output")
OUT.mkdir(exist_ok=True)
con = duckdb.connect()
print("[1/4] Loading test data...")
con.execute(f"""
CREATE OR REPLACE VIEW s1 AS
SELECT
    entity_id,
    lower(trim(regexp_replace(coalesce(business_name,''), '[^[:alnum:]]+', '', 'g'))) AS name_norm,
    lower(trim(regexp_replace(coalesce(business_address,''), '[^[:alnum:]]+', '', 'g'))) AS address_norm,
    lower(trim(coalesce(country,''))) AS country_norm
FROM read_csv_auto('{BASE}/test_source1.tsv', delim='\t', header=true)
""")
con.execute(f"""
CREATE OR REPLACE VIEW s23 AS
SELECT
    entity_id,
    lower(trim(regexp_replace(coalesce(business_name,''), '[^[:alnum:]]+', '', 'g'))) AS name_norm,
    lower(trim(regexp_replace(coalesce(business_address,''), '[^[:alnum:]]+', '', 'g'))) AS address_norm,
    lower(trim(coalesce(country,''))) AS country_norm
FROM (
    SELECT * FROM read_csv_auto('{BASE}/test_source2.tsv', delim='\t', header=true)
    UNION ALL BY NAME
    SELECT * FROM read_csv_auto('{BASE}/test_source3.tsv', delim='\t', header=true)
)
""")
print("[2/4] Finding unique full exact matches...")
con.execute("""
CREATE OR REPLACE VIEW keyed AS
SELECT
    *,
    count(*) OVER (
        PARTITION BY country_norm, name_norm, address_norm
    ) AS full_count
FROM s23
""")
con.execute("""
CREATE OR REPLACE VIEW matches AS
SELECT
    s.entity_id AS source1_entity_id,
    e.entity_id AS matched_entity_id
FROM s1 s
JOIN keyed e
    ON s.country_norm <> ''
   AND s.name_norm <> ''
   AND s.address_norm <> ''
   AND s.country_norm = e.country_norm
   AND s.name_norm = e.name_norm
   AND s.address_norm = e.address_norm
   AND e.full_count = 1
""")
print("[3/4] Writing Submission #2...")
matching = OUT / "matching_results_v2.tsv"
candidate = OUT / "candidate_pairs_v2.tsv"
# Every S1 must appear, including unmatched rows.
con.execute(f"""
COPY (
    SELECT
        s.entity_id AS source1_entity_id,
        COALESCE(
            string_agg(m.matched_entity_id, ',' ORDER BY m.matched_entity_id),
            ''
        ) AS matched_entity_ids
    FROM s1 s
    LEFT JOIN matches m
        ON s.entity_id = m.source1_entity_id
    GROUP BY s.entity_id
    ORDER BY s.entity_id
)
TO '{matching.as_posix()}'
WITH (FORMAT CSV, DELIMITER '\t', HEADER TRUE)
""")
con.execute(f"""
COPY (
    SELECT
        s.entity_id AS source1_entity_id,
        COALESCE(
            string_agg(m.matched_entity_id, ',' ORDER BY m.matched_entity_id),
            ''
        ) AS candidate_entity_ids
    FROM s1 s
    LEFT JOIN matches m
        ON s.entity_id = m.source1_entity_id
    GROUP BY s.entity_id
    ORDER BY s.entity_id
)
TO '{candidate.as_posix()}'
WITH (FORMAT CSV, DELIMITER '\t', HEADER TRUE)
""")
print("[4/4] Checking output...")
print(con.execute("""
SELECT
    (SELECT count(*) FROM s1) AS s1_rows,
    (SELECT count(*) FROM matches) AS match_links,
    (SELECT count(DISTINCT source1_entity_id) FROM matches) AS matched_s1
""").fetchone())
print("Created:")
print(matching)
print(candidate)
