from pathlib import Path
for filename in [
    "output/matching_results_v2.tsv",
    "output/candidate_pairs_v2.tsv"
]:
    p = Path(filename)
    data = p.read_text(encoding="utf-8")
    # DuckDB CSV writer produced empty fields as ""
    lines = data.splitlines(True)
    fixed = []
    for line in lines:
        if line.rstrip("\r\n").endswith('\t""'):
            ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            body = line[:-len(ending)] if ending else line
            body = body[:-3] + "\t"
            fixed.append(body + ending)
        else:
            fixed.append(line)
    p.write_text("".join(fixed), encoding="utf-8")
    print("Fixed:", filename)
