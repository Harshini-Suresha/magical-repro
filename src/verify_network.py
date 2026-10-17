"""Check which shipped pair features can be reproduced from the shipped edge list.

Node-level columns (degree, betweenness, neighbourhood size, triangle count) are
joined to genes by name; pair-level columns (shortest_path, cohesion, adhesion)
were computed in a loop over the pair table. This script compares both kinds.

Run: .venv/bin/python src/verify_network.py
Saves results/network_verification.json
"""
import json
import os
import time
import numpy as np
import pandas as pd

from network_io import load_network, load_pairs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    g = load_network()
    vcount = g.vcount()
    names = {int(nm): i for i, nm in enumerate(g.vs["name"])}
    t0 = time.time()
    deg = np.asarray(g.degree())
    tri = np.zeros(vcount, dtype=np.int64)
    for a, b, c in g.list_triangles():
        tri[a] += 1
        tri[b] += 1
        tri[c] += 1
    nb2 = np.asarray(g.neighborhood_size(order=2))
    bet = np.asarray(g.betweenness(directed=False))
    print(f"node properties in {time.time()-t0:.1f}s")

    pairs = load_pairs()
    pairs = pairs[pairs["gene1"].isin(names) & pairs["gene2"].isin(names)]
    sample = pairs.sample(500, random_state=1)

    stats = {k: 0 for k in ["degree", "neighbor2", "triangle", "betweenness", "shortest_path"]}
    for _, r in sample.iterrows():
        ia, ib = names[int(r["gene1"])], names[int(r["gene2"])]
        stats["degree"] += (int(deg[ia]) + int(deg[ib])) / 2 == r["avgdeg"]
        stats["neighbor2"] += abs((int(nb2[ia]) + int(nb2[ib])) / 2 - r["avgneighbor2"]) <= 0.5
        stats["triangle"] += (int(tri[ia]) + int(tri[ib])) / 2 == r["avgtriangle"]
        stats["betweenness"] += abs((int(bet[ia]) + int(bet[ib])) / 2 - r["avgbet"]) <= max(1.0, 0.01 * r["avgbet"])
        d = g.distances(source=[ia], target=[ib])[0][0]
        d = 26 if not np.isfinite(d) else int(d)
        stats["shortest_path"] += d == int(r["shortest_path"])
    stats = {k: round(v / len(sample), 3) for k, v in stats.items()}
    print("share of 500 shipped rows reproduced:", stats)

    t0 = time.time()
    conn = [g.edge_connectivity(names[int(r["gene1"])], names[int(r["gene2"])])
            for _, r in sample.head(50).iterrows()]
    per_pair_ms = (time.time() - t0) / 50 * 1000
    print(f"edge_connectivity: {per_pair_ms:.1f} ms/pair")
    adm = sample.head(50)
    adm_share = float(np.mean([c == int(s) for c, s in zip(conn, adm["adhesion"])]))
    print("adhesion exact share on 50 rows:", round(adm_share, 3))

    report = {
        "network": {"nodes": g.vcount(), "edges": g.ecount(),
                    "components": len(g.connected_components()),
                    "largest_component": max(g.connected_components().sizes())},
        "reproduced_share": stats,
        "adhesion_exact_share_50rows": adm_share,
        "edge_connectivity_ms_per_pair": round(per_pair_ms, 2),
        "notes": [
            "degree, neighbour2, triangle and betweenness reproduce the shipped values",
            "shipped shortest_path and adhesion do not: the pair-level loop in "
            "network-property-calculation.R queried graph vertices with factor values, "
            "so the two columns cannot be recomputed from the shipped edge list",
            "shipped files contain no pair at distance 1 (adjacency pairs were dropped "
            "by na.omit after cohesion was set NA for them)",
        ],
    }
    with open(os.path.join(RESULTS, "network_verification.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("saved results/network_verification.json")


if __name__ == "__main__":
    main()
