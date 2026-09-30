import duckdb
BASE = r"D:\dataset_Amazon\student_resource\dataset\train"
con = duckdb.connect()
print("Loading training views...")
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
    unnest(string_split(
        CASE WHEN trim(matched_entity_ids) = '' THEN NULL
             ELSE matched_entity_ids END, ','
    )) AS target_id
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
FROM gt
GROUP BY source1_entity_id
""")
con.execute("""
CREATE OR REPLACE VIEW keyed AS
SELECT *,
    count(*) OVER (
        PARTITION BY country_norm, name_norm, address_norm
    ) AS full_count,
    count(*) OVER (
        PARTITION BY country_norm, name_norm
    ) AS name_count,
    count(*) OVER (
        PARTITION BY country_norm, address_norm
    ) AS address_count
FROM s23
""")
rules = {
"FULL_EXACT_UNIQUE": """
    s.country_norm <> '' AND s.name_norm <> '' AND s.address_norm <> ''
    AND s.country_norm = e.country_norm
    AND s.name_norm = e.name_norm
    AND s.address_norm = e.address_norm
    AND e.full_count = 1
""",
"NAME_UNIQUE": """
    s.country_norm <> '' AND s.name_norm <> ''
    AND s.country_norm = e.country_norm
    AND s.name_norm = e.name_norm
    AND e.name_count = 1
""",
"ADDRESS_UNIQUE": """
    s.country_norm <> '' AND s.address_norm <> ''
    AND s.country_norm = e.country_norm
    AND s.address_norm = e.address_norm
    AND e.address_count = 1
""",
"NAME_AND_ADDRESS": """
    s.country_norm <> '' AND s.name_norm <> '' AND s.address_norm <> ''
    AND s.country_norm = e.country_norm
    AND s.name_norm = e.name_norm
    AND s.address_norm = e.address_norm
""",
}
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
    result = con.execute("""
    WITH pred_stats AS (
        SELECT
            s1_id,
            count(DISTINCT target_id) AS predicted
        FROM pred
        GROUP BY s1_id
    ),
    tp_stats AS (
        SELECT
            p.s1_id,
            count(DISTINCT p.target_id) AS tp
        FROM pred p
        JOIN gt g
          ON g.source1_entity_id = p.s1_id
         AND g.target_id = p.target_id
        GROUP BY p.s1_id
    ),
    base AS (
        SELECT
            s.entity_id,
            coalesce(ps.predicted, 0) AS predicted,
            coalesce(ts.tp, 0) AS tp,
            coalesce(gc.actual, 0) AS actual
        FROM s1 s
        LEFT JOIN pred_stats ps ON ps.s1_id = s.entity_id
        LEFT JOIN tp_stats ts ON ts.s1_id = s.entity_id
        LEFT JOIN gt_counts gc ON gc.source1_entity_id = s.entity_id
    ),
    scored AS (
        SELECT *,
            CASE
                WHEN predicted = 0 AND actual = 0 THEN 1.0
                WHEN predicted = 0 THEN 0.0
                WHEN actual = 0 THEN 0.0
                ELSE
                    (1.25 * tp)
                    /
                    (
                        1.25 * tp
                        + 0.25 * (predicted - tp)
                        + (actual - tp)
                    )
            END AS f05
        FROM base
    )
    SELECT
        avg(f05) AS macro_f05,
        sum(CASE WHEN actual > 0 THEN 1 ELSE 0 END) AS s1_with_truth,
        sum(CASE WHEN predicted > 0 THEN 1 ELSE 0 END) AS s1_predicted,
        sum(actual) AS true_links,
        sum(predicted) AS predicted_links,
        sum(tp) AS true_positive_links,
        avg(CASE WHEN predicted > 0 THEN predicted ELSE 0 END) AS avg_predicted_all_s1
    FROM scored
    """).fetchone()
    print(result)
print("\nDONE")
