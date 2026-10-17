"""Consistent re-extraction of the five MAGICAL feature types from a PPI network.

Used for the robustness experiments: the shipped shortest_path and adhesion
columns cannot be reproduced from the shipped edge list, so the perturbation
curves are computed with one consistent extraction applied to every network
variant (original, node-removed, edge-removed, noise-added, rewired).

Node-level: degree, betweenness, 2-hop neighbourhood size, triangle count.
Pair-level: shortest path (disconnected = 26, the original convention), adhesion
(edge connectivity, 0 when disconnected).
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

MAX_DIST = 26


def node_properties(g):
    n = g.vcount()
    tri = np.zeros(n, dtype=np.int64)
    for a, b, c in g.list_triangles():
        tri[a] += 1
        tri[b] += 1
        tri[c] += 1
    return {
        "deg": np.asarray(g.degree()),
        "bet": np.asarray(g.betweenness(directed=False)),
        "nb2": np.asarray(g.neighborhood_size(order=2)),
        "tri": tri,
    }


def _distances(g, sources_idx, targets_idx):
    """Shortest path lengths for element-wise pairs (sources_idx, targets_idx)."""
    adj = csr_matrix(g.get_adjacency_sparse())
    out = np.full(len(sources_idx), MAX_DIST, dtype=np.int64)
    uniq = np.unique(sources_idx)
    for s in uniq:
        sel = sources_idx == s
        dist = dijkstra(adj, indices=int(s), unweighted=True, limit=MAX_DIST)
        d = dist[targets_idx[sel]]
        d = np.where(np.isfinite(d), d, MAX_DIST).astype(np.int64)
        out[sel] = d
    return out


def pair_features(g, pairs, props=None):
    """pairs: DataFrame with columns gene1, gene2 (int). Returns feature frame.

    Rows whose genes are missing from g are dropped. props (from node_properties)
    can be passed in to avoid recomputation.
    """
    idx = {int(nm): i for i, nm in enumerate(g.vs["name"])}
    keep = pairs["gene1"].isin(idx) & pairs["gene2"].isin(idx)
    sub = pairs[keep].reset_index(drop=True)
    ia = sub["gene1"].map(idx).to_numpy()
    ib = sub["gene2"].map(idx).to_numpy()

    if props is None:
        props = node_properties(g)
    sp = _distances(g, ia, ib)

    adm = np.zeros(len(sub), dtype=np.int64)
    for k, (a, b) in enumerate(zip(ia, ib)):
        try:
            adm[k] = g.edge_connectivity(int(a), int(b))
        except Exception:
            adm[k] = 0

    out = sub[["gene1", "gene2"]].copy()
    out["avgdeg"] = (props["deg"][ia] + props["deg"][ib]) / 2
    out["avgbet"] = (props["bet"][ia].astype(np.int64) + props["bet"][ib].astype(np.int64)) / 2
    out["avgneighbor2"] = (props["nb2"][ia].astype(np.int64) + props["nb2"][ib].astype(np.int64)) / 2
    out["avgtriangle"] = (props["tri"][ia] + props["tri"][ib]) / 2
    out["shortest_path"] = sp
    out["adhesion"] = adm
    return out


FEATURES = ["avgneighbor2", "avgbet", "avgtriangle", "shortest_path", "adhesion"]
