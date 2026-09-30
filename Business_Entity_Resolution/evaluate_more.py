# Append additional rule evaluation to the existing corrected evaluator.
# This reuses the views already created by evaluate_correct.py.
import duckdb
BASE = r"D:\dataset_Amazon\student_resource\dataset\train"
con = duckdb.connect()
con.execute(f"""
CREATE OR REPLACE VIEW s1 AS
SELECT
    entity_id,
    lower(trim(regexp_replace(coalesce(business_name,''), '[^[:alnum:]]+', '', 'g'))) AS name_norm,
    lower(trim(regexp_replace(coalesce(business_address,''), '[^[:alnum:]]+', '', 'g'))) AS address_norm,
    lower(trim(coalesce(country,''))) AS country_norm
FROM read_csv_auto('{BASE}/train_source1.tsv', delim='\t', header=true)
""")
con.execute(f"""
CREATE OR REPLACE VIEW s23 AS
SELECT
    entity_id,
    lower(trim(regexp_replace(coalesce(business_name,''), '[^[:alnum:]]+', '', 'g'))) AS name_norm,
    lower(trim(regexp_replace(coalesce(business_address,''), '[^[:alnum:]]+', '', 'g'))) AS address_norm,
    lower(trim(coalesce(country,''))) AS country_norm
FROM (
    SELECT * FROM read_csv_auto('{BASE}/train_source2.tsv', delim='\t', header=true)
    UNION ALL BY NAME
    SELECT * FROM read_csv_auto('{BASE}/train_source3.tsv', delim='\t', header=true)
)
""")
con.execute(f"""
CREATE OR REPLACE VIEW gt AS
SELECT
    source1_entity_id,
    unnest(string_split(matched_entity_ids, ',')) AS target_id
FROM read_csv_auto(
    '{BASE}/train_ground_truth.tsv',
    delim='\t',
    header=true
)
WHERE trim(matched_entity_ids) <> ''
""")
con.execute("""
CREATE OR REPLACE VIEW gt_counts AS
SELECT source1_entity_id, count(DISTINCT target_id) AS actual
FROM gt GROUP BY source1_entity_id
""")
rules = {
    "V1_ORIGINAL": """
        s.country_norm <> ''
        AND s.country_norm = e.country_norm
        AND (
            (s.name_norm <> '' AND s.name_norm = e.name_norm)
            OR
            (s.address_norm <> '' AND s.address_norm = e.address_norm)
        )
    """,
    "NAME_ONLY": """
        s.country_norm <> '' AND s.name_norm <> ''
        AND s.country_norm = e.country_norm
        AND s.name_norm = e.name_norm
    """,
    "ADDRESS_ONLY": """
        s.country_norm <> '' AND s.address_norm <> ''
        AND s.country_norm = e.country_norm
        AND s.address_norm = e.address_norm
    """,
    "NAME_UNIQUE_OR_ADDRESS_UNIQUE": """
        s.country_norm <> ''
        AND s.country_norm = e.country_norm
        AND (
            (
                s.name_norm <> ''
                AND s.name_norm = e.name_norm
                AND e.name_count = 1
            )
            OR
            (
                s.address_norm <> ''
                AND s.address_norm = e.address_norm
                AND e.address_count = 1
            )
        )
    """
}
con.execute("""
CREATE OR REPLACE VIEW keyed AS
SELECT *,
    count(*) OVER (
        PARTITION BY country_norm, name_norm
    ) AS name_count,
    count(*) OVER (
        PARTITION BY country_norm, address_norm
    ) AS address_count
FROM s23
""")
for rule, condition in rules.items():
    print("\n==============================")
    print(rule)
    print("==============================")
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW pred AS
    SELECT DISTINCT
        s.entity_id AS s1_id,
        e.entity_id AS target_id
    FROM s1 s
    JOIN keyed e ON {condition}
    """)
    r = con.execute("""
    WITH ps AS (
        SELECT s1_id, count(*) AS predicted
        FROM pred GROUP BY s1_id
    ),
    ts AS (
        SELECT p.s1_id, count(DISTINCT p.target_id) AS tp
        FROM pred p
        JOIN gt g
          ON g.source1_entity_id = p.s1_id
         AND g.target_id = p.target_id
        GROUP BY p.s1_id
    ),
    base AS (
        SELECT
            s.entity_id,
            coalesce(ps.predicted,0) AS predicted,
            coalesce(ts.tp,0) AS tp,
            coalesce(gc.actual,0) AS actual
        FROM s1 s
        LEFT JOIN ps ON ps.s1_id=s.entity_id
        LEFT JOIN ts ON ts.s1_id=s.entity_id
        LEFT JOIN gt_counts gc ON gc.source1_entity_id=s.entity_id
    ),
    scored AS (
        SELECT *,
        CASE
            WHEN predicted=0 AND actual=0 THEN 1.0
            WHEN predicted=0 THEN 0.0
            WHEN actual=0 THEN 0.0
            ELSE
                1.25*tp /
                (1.25*tp + 0.25*(predicted-tp) + (actual-tp))
        END AS f05
        FROM base
    )
    SELECT
        avg(f05) AS macro_f05,
        sum(CASE WHEN predicted>0 THEN 1 ELSE 0 END) AS predicted_s1,
        sum(predicted) AS predicted_links,
        sum(tp) AS tp,
        avg(CASE WHEN predicted>0 THEN predicted ELSE 0 END) AS avg_predicted
    FROM scored
    """).fetchone()
    print(r)
