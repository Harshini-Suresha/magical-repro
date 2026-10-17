"""Load the official MAGICAL data: PPI edge list, pair feature tables, balanced set."""
import os
import pandas as pd
import igraph as ig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "official")

PAIR_FILES = {
    "biogrid-sl": "biogrid-sl-pairwise-new-netprop",
    "biogrid-sv": "biogrid-sv-pairwise-new-netprop",
    "cgidb-sl": "cgidb-sl-pairwise-new-netprop",
    "cgidb-sv": "cgidb-sv-pairwise-new-netprop",
    "sldb-sl": "sldb-sl-pairwise-new-netprop",
    "sldb-sv": "sldb-sv-pairwise-new-netprop",
    "not": "not0.pairwiseprop.csv",
}

SOURCE_OF = {
    "biogrid-sl": "BioGRID", "biogrid-sv": "BioGRID",
    "cgidb-sl": "CGIdb", "cgidb-sv": "CGIdb",
    "sldb-sl": "SynLethDB", "sldb-sv": "SynLethDB",
    "not": "NOT",
}


def load_pairs():
    frames = []
    for key, fname in PAIR_FILES.items():
        df = pd.read_csv(os.path.join(DATA, fname))
        df["source"] = SOURCE_OF[key]
        df["gi"] = {"sl": "SL", "sv": "SV"}.get(key.split("-")[-1], "NOT")
        frames.append(df)
    pairs = pd.concat(frames, ignore_index=True)
    pairs["gene1"] = pairs["gene1"].astype("int64")
    pairs["gene2"] = pairs["gene2"].astype("int64")
    return pairs


def load_balanced():
    df = pd.read_csv(os.path.join(DATA, "magical-bal-data-with-geneids.csv"))
    df["gene1"] = df["gene1"].astype("int64")
    df["gene2"] = df["gene2"].astype("int64")
    return df


def load_network():
    edges = []
    with open(os.path.join(DATA, "biogrid-entrez-pairs")) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                a, b = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edges.append((a, b))
    edges = list(dict.fromkeys(edges))
    g = ig.Graph.TupleList(edges, directed=False)
    return g


if __name__ == "__main__":
    g = load_network()
    print("nodes", g.vcount(), "edges", g.ecount())
    comps = g.connected_components()
    print("components", len(comps), "largest", max(comps.sizes()))
    pairs = load_pairs()
    print(pairs.groupby(["gi", "source"]).size())
    bal = load_balanced()
    print("balanced", bal.shape, dict(bal["gi"].value_counts()))
