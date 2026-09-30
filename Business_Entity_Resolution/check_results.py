import duckdb
con = duckdb.connect()
con.execute("""
CREATE OR REPLACE VIEW m AS
SELECT * FROM read_csv_auto(
    'output/matching_results.tsv',
    delim='\t',
    header=true
)
""")
q = """
SELECT
    COUNT(*) AS rows,
    SUM(CASE WHEN matched_entity_ids = '' THEN 1 ELSE 0 END) AS empty,
    AVG(
        CASE
            WHEN matched_entity_ids = '' THEN 0
            ELSE length(matched_entity_ids)
                 - length(replace(matched_entity_ids, ',', '')) + 1
        END
    ) AS avg_matches_nonempty,
    MAX(
        CASE
            WHEN matched_entity_ids = '' THEN 0
            ELSE length(matched_entity_ids)
                 - length(replace(matched_entity_ids, ',', '')) + 1
        END
    ) AS max_matches
FROM m
"""
print(con.execute(q).fetchall())
