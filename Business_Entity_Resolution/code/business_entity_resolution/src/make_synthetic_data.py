import os
import random
import pandas as pd

random.seed(0)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATASET = os.path.join(BASE, "dataset")
TRAIN = os.path.join(DATASET, "train")
TEST = os.path.join(DATASET, "test")
os.makedirs(TRAIN, exist_ok=True)
os.makedirs(TEST, exist_ok=True)

BUSINESSES = [
    ("Acme Corporation", "123 Main Street, New York, NY 10001", "US"),
    ("Globex Industries Ltd", "456 Oak Avenue, Los Angeles, CA 90001", "US"),
    ("Initech Private Limited", "789 Park Road, Bengaluru, Karnataka 560001", "India"),
    ("Umbrella Corp", "321 Elm St, Chicago, IL 60601", "US"),
    ("Stark Industries Inc", "1000 Broadway, San Francisco, CA 94105", "US"),
    ("Wayne Enterprises", "1007 Mountain Drive, Gotham, NJ 07001", "US"),
    ("Pied Piper Inc", "5230 Newell Road, Palo Alto, CA 94301", "US"),
    ("Hooli Corporation", "1 Hooli Way, Mountain View, CA 94043", "US"),
    ("Tata Consultancy Services", "Bandra Kurla Complex, Mumbai, Maharashtra 400051", "India"),
    ("Infosys Limited", "Electronics City, Hosur Road, Bengaluru 560100", "India"),
    ("Wipro Pvt Ltd", "Sarjapur Road, Bengaluru, Karnataka 560035", "India"),
    ("Reliance Industries", "Maker Chambers IV, Nariman Point, Mumbai 400021", "India"),
    ("Soylent Corp", "450 Sansome St, San Francisco, CA 94111", "US"),
    ("Cyberdyne Systems", "18144 El Camino Real, Sunnyvale, CA 94087", "US"),
    ("Tyrell Corporation", "Los Angeles, CA", "US"),
    ("Massive Dynamic", "2425 George Washington Memorial Hwy, Boston, MA 02134", "US"),
    ("Dunder Mifflin Paper Company", "1725 Slough Avenue, Scranton, PA 18503", "US"),
    ("Los Pollos Hermanos", "12000 – 12100 Coors Blvd, Albuquerque, NM 87121", "US"),
    ("Sterling Cooper", "1271 Avenue of the Americas, New York, NY 10020", "US"),
    ("Bluth Company", "1 Lucille Lane, Newport Beach, CA 92663", "US"),
    ("Oscar & Gomez Associates", "40 Rue du Faubourg Saint-Honore, Paris 75008", "France"),
    ("Bistro Chez Pierre SARL", "12 Avenue des Champs Elysees, Paris 75008", "France"),
    ("Fromage Fort SA", "8 Rue de la Paix, Lyon 69002", "France"),
    ("La Boulangerie du Village", "25 Grande Rue, Marseille 13001", "France"),
    ("Eiffel Tech Solutions", "48 Rue de Rivoli, Paris 75004", "France"),
]


def noisify_name(name: str) -> str:
    n = name
    opts = [
        lambda x: x,
        lambda x: x,
        lambda x: x.replace("Corporation", "Corp"),
        lambda x: x.replace("Incorporated", "Inc"),
        lambda x: x.replace("Limited", "Ltd"),
        lambda x: x.replace("Private", "Pvt"),
        lambda x: x.replace("Company", "Co"),
        lambda x: x.replace(" Inc", ""),
        lambda x: x.replace(" Corp", ""),
        lambda x: x.replace(" Ltd", ""),
        lambda x: x.replace(" Pvt", ""),
        lambda x: x.replace(" & ", " and "),
        lambda x: x.replace(" and ", " & "),
    ]
    for fn in random.sample(opts, random.randint(0, 2)):
        n = fn(n)
    return n.strip()


def noisify_addr(addr: str) -> str:
    a = addr
    opts = [
        lambda x: x,
        lambda x: x,
        lambda x: x.replace("Street", "St"),
        lambda x: x.replace("Avenue", "Ave"),
        lambda x: x.replace("Road", "Rd"),
        lambda x: x.replace("Drive", "Dr"),
        lambda x: x.replace("Boulevard", "Blvd"),
        lambda x: x.replace("Lane", "Ln"),
        lambda x: x.split(",")[0] if random.random() < 0.25 else x,
        lambda x: ", ".join(x.split(", ")[1:]) if random.random() < 0.15 else x,
    ]
    for fn in random.sample(opts, random.randint(0, 2)):
        a = fn(a)
    return a.strip()


def make_df(prefix, start, n):
    rows = []
    for i in range(n):
        base_idx = (start + i) % len(BUSINESSES)
        bname, baddr, bcountry = BUSINESSES[base_idx]
        rows.append({
            "entity_id": f"{prefix}-{start + i:05d}",
            "business_name": noisify_name(bname),
            "business_address": noisify_addr(baddr),
            "country": bcountry,
            "_base_idx": base_idx,
        })
    return pd.DataFrame(rows)


print("Creating synthetic dataset...")
train_s1 = make_df("S1", 1, 40)
train_s2 = make_df("S2", 1, 80)
train_s3 = make_df("S3", 1, 60)
test_s1 = make_df("S1", 41, 30)
test_s2 = make_df("S2", 81, 70)
test_s3 = make_df("S3", 61, 50)

print(f"Train S1:{len(train_s1)} S2:{len(train_s2)} S3:{len(train_s3)}")
print(f"Test S1:{len(test_s1)} S2:{len(test_s2)} S3:{len(test_s3)}")

gt_rows = []
for _, r in train_s1.iterrows():
    bi = r["_base_idx"]
    matches = []
    for _, r2 in train_s2.iterrows():
        if r2["_base_idx"] == bi:
            matches.append(r2["entity_id"])
    for _, r3 in train_s3.iterrows():
        if r3["_base_idx"] == bi:
            matches.append(r3["entity_id"])
    gt_rows.append({"source1_entity_id": r["entity_id"], "matched_entity_ids": ",".join(matches)})
gt_df = pd.DataFrame(gt_rows)

for df in (train_s1, train_s2, train_s3, test_s1, test_s2, test_s3):
    df.drop(columns=["_base_idx"], inplace=True)

train_s1.to_csv(os.path.join(TRAIN, "train_source1.tsv"), sep="\t", index=False)
train_s2.to_csv(os.path.join(TRAIN, "train_source2.tsv"), sep="\t", index=False)
train_s3.to_csv(os.path.join(TRAIN, "train_source3.tsv"), sep="\t", index=False)
gt_df.to_csv(os.path.join(TRAIN, "train_ground_truth.tsv"), sep="\t", index=False)
test_s1.to_csv(os.path.join(TEST, "test_source1.tsv"), sep="\t", index=False)
test_s2.to_csv(os.path.join(TEST, "test_source2.tsv"), sep="\t", index=False)
test_s3.to_csv(os.path.join(TEST, "test_source3.tsv"), sep="\t", index=False)

print("Done creating dataset.")
print(f"GT singletons: {(gt_df['matched_entity_ids'] == '').sum()} / {len(gt_df)}")
print(f"GT avg matches per S1: {gt_df['matched_entity_ids'].apply(lambda s: len(s.split(',')) if s else 0).mean():.2f}")
