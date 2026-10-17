"""Pair features from a PPI network. Replace/extend to match the MAGICAL repository."""
import networkx as nx

def pair_features(G, a, b):
    nbr_a, nbr_b = set(G[a]), set(G[b])
    common = nbr_a & nbr_b
    union = nbr_a | nbr_b
    try:
        dist = nx.shortest_path_length(G, a, b)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        dist = -1
    return {
        "deg_a": G.degree(a), "deg_b": G.degree(b),
        "common_neighbours": len(common),
        "jaccard": len(common) / len(union) if union else 0.0,
        "shortest_path": dist,
    }
