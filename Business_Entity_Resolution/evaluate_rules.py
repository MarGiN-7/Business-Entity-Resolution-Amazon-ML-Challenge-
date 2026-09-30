import duckdb
BASE = r"D:\dataset_Amazon\student_resource\dataset"
con = duckdb.connect()
con.execute(f"""
CREATE OR REPLACE VIEW s1 AS
SELECT
    entity_id,
    lower(trim(regexp_replace(coalesce(business_name,''), '[^[:alnum:]]+', '', 'g'))) AS name_norm,
    lower(trim(regexp_replace(coalesce(business_address,''), '[^[:alnum:]]+', '', 'g'))) AS address_norm,
    lower(trim(coalesce(country,''))) AS country_norm
FROM read_csv_auto('{BASE}/train/train_source1.tsv', delim='\t', header=true)
""")
con.execute(f"""
CREATE OR REPLACE VIEW s23 AS
SELECT
    entity_id,
    lower(trim(regexp_replace(coalesce(business_name,''), '[^[:alnum:]]+', '', 'g'))) AS name_norm,
    lower(trim(regexp_replace(coalesce(business_address,''), '[^[:alnum:]]+', '', 'g'))) AS address_norm,
    lower(trim(coalesce(country,''))) AS country_norm
FROM (
    SELECT * FROM read_csv_auto('{BASE}/train/train_source2.tsv', delim='\t', header=true)
    UNION ALL BY NAME
    SELECT * FROM read_csv_auto('{BASE}/train/train_source3.tsv', delim='\t', header=true)
)
""")
con.execute(f"""
CREATE OR REPLACE VIEW gt AS
SELECT
    source1_entity_id,
    unnest(string_split(matched_entity_ids, ',')) AS target_id
FROM read_csv_auto(
    '{BASE}/train/train_ground_truth.tsv',
    delim='\t',
    header=true
)
""")
# Target-side uniqueness for each key.
con.execute("""
CREATE OR REPLACE VIEW keyed AS
SELECT
    *,
    count(*) OVER (PARTITION BY country_norm, name_norm) AS name_count,
    count(*) OVER (PARTITION BY country_norm, address_norm) AS address_count,
    count(*) OVER (PARTITION BY country_norm, name_norm, address_norm) AS full_count
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
}
for rule, condition in rules.items():
    print("\n==============================")
    print(rule)
    print("==============================")
    q = f"""
    WITH pred AS (
        SELECT s.entity_id AS s1_id, e.entity_id AS target_id
        FROM s1 s
        JOIN keyed e ON {condition}
    ),
    scored AS (
        SELECT
            s1.entity_id,
            count(DISTINCT p.target_id) AS predicted,
            count(DISTINCT CASE WHEN g.target_id IS NOT NULL THEN p.target_id END) AS tp,
            count(DISTINCT g.target_id) AS actual
        FROM s1
        LEFT JOIN pred p ON p.s1_id = s1.entity_id
        LEFT JOIN gt g ON g.source1_entity_id = s1.entity_id
                         AND g.target_id = p.target_id
        GROUP BY s1.entity_id
    ),
    metrics AS (
        SELECT
            *,
            CASE
                WHEN predicted = 0 AND actual = 0 THEN 1.0
                WHEN predicted = 0 THEN 0.0
                WHEN tp = 0 THEN 0.0
                ELSE
                    (1.25 * tp)
                    / (1.25 * tp + 0.25 * (predicted - tp) + (actual - tp))
            END AS f05
        FROM scored
    )
    SELECT
        AVG(f05) AS macro_f05,
        SUM(CASE WHEN predicted > 0 THEN 1 ELSE 0 END) AS matched_s1,
        AVG(CASE WHEN predicted > 0 THEN predicted ELSE 0 END) AS avg_predictions,
        SUM(predicted) AS total_predictions,
        SUM(tp) AS total_tp
    FROM metrics
    """
    print(con.execute(q).fetchone())
print("\nDONE")
