import json
import os
import time
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&retmode=json&id="


def fetch(ids):
    out = {}
    ids = [str(i) for i in ids]
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        for attempt in range(3):
            try:
                with urllib.request.urlopen(BASE + ",".join(chunk), timeout=30) as fh:
                    data = json.load(fh)["result"]
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        for uid in data["uids"]:
            g = data[uid]
            out[uid] = (g.get("name", ""), g.get("organism", {}).get("scientificname", ""))
        time.sleep(0.4)
    return out


def main():
    cand = pd.read_csv(os.path.join(RESULTS, "candidates.csv"))
    ids = pd.unique(pd.concat([cand["gene1"], cand["gene2"]]).astype(str))
    print(f"annotating {len(ids)} genes for {len(cand)} candidate pairs")
    info = fetch(ids)
    missing = [i for i in map(str, ids) if i not in info]
    for c, side in (("gene1", "1"), ("gene2", "2")):
        cand["symbol" + side] = cand[c].astype(str).map(lambda i: info.get(i, ("", ""))[0])
        cand["species" + side] = cand[c].astype(str).map(lambda i: info.get(i, ("", ""))[1])
    cand["human_pair"] = (cand["species1"] == "Homo sapiens") & (cand["species2"] == "Homo sapiens")
    cols = ["gene1", "symbol1", "species1", "gene2", "symbol2", "species2", "human_pair",
            "class", "n_variants_available", "n_variants_interacting", "mean_confidence_when_interacting"]
    cand = cand[cols]
    out_path = os.path.join(RESULTS, "candidates_annotated.csv")
    cand.to_csv(out_path, index=False)
    print("wrote", out_path)
    print("human pairs:", int(cand["human_pair"].sum()), "of", len(cand))
    if missing:
        print("unresolved ids:", len(missing))
    print("top human:", cand[cand["human_pair"]][["gene1", "symbol1", "gene2", "symbol2", "class", "mean_confidence_when_interacting"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
