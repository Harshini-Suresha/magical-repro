"""Degree-preserving null for pair features.

Usage: python src/degree_null.py data/toy_edges.txt ATM BRCA1 [n_rewirings]
Edge file: one interaction per line, two gene names separated by space, tab or comma.
Compares a pair's shared neighbours, Jaccard, Adamic-Adar and shortest path with networks
rewired to keep every gene's degree. A pair that is no closer than the null may just reflect hubs.
"""
import sys
import numpy as np
import networkx as nx


def read_edges(path):
    G = nx.Graph()
    for line in open(path):
        p = line.split("#")[0].replace(",", " ").split()
        if len(p) >= 2 and p[0] != p[1]:
            G.add_edge(p[0], p[1])
    return G


def features(G, a, b):
    na, nb = set(G[a]), set(G[b])
    common = na & nb
    union = na | nb
    try:
        d = nx.shortest_path_length(G, a, b)
    except nx.NetworkXNoPath:
        d = 9
    aa = sum(1 / np.log(G.degree(k)) for k in common if G.degree(k) > 1)
    return np.array([len(common), len(common) / len(union) if union else 0.0, d, aa], float)


def pair_null(G, a, b, n=300, seed=0):
    rng = np.random.default_rng(seed)
    m = G.number_of_edges()
    obs = features(G, a, b)
    null = []
    for _ in range(n):
        H = G.copy()
        nx.double_edge_swap(H, nswap=5 * m, max_tries=100 * m, seed=int(rng.integers(1 << 31)))
        null.append(features(H, a, b))
    null = np.array(null)
    out = []
    for i, name in enumerate(["common neighbours", "jaccard", "shortest path", "adamic-adar"]):
        mu, sd = null[:, i].mean(), null[:, i].std()
        hit = (null[:, i] <= obs[i]) if name == "shortest path" else (null[:, i] >= obs[i])
        out.append((name, obs[i], mu, sd, (obs[i] - mu) / sd if sd else float("nan"), (hit.sum() + 1) / (n + 1)))
    return out


if __name__ == "__main__":
    path, a, b = sys.argv[1:4]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    print(f"{'feature':18s}{'observed':>10s}{'null mean':>11s}{'null sd':>9s}{'z':>7s}{'p':>8s}")
    for name, o, mu, sd, z, p in pair_null(read_edges(path), a, b, n):
        print(f"{name:18s}{o:10.2f}{mu:11.2f}{sd:9.2f}{z:7.1f}{p:8.3f}")
