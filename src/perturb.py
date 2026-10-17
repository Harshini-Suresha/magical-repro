"""Network perturbations. G is a networkx Graph of the PPI network."""
import random
import networkx as nx

def remove_nodes(G, frac, seed=0):
    rng = random.Random(seed); H = G.copy()
    H.remove_nodes_from(rng.sample(list(H.nodes), int(frac * H.number_of_nodes())))
    return H

def remove_edges(G, frac, seed=0):
    rng = random.Random(seed); H = G.copy()
    H.remove_edges_from(rng.sample(list(H.edges), int(frac * H.number_of_edges())))
    return H

def add_noise_edges(G, frac, seed=0):
    rng = random.Random(seed); H = G.copy(); nodes = list(H.nodes)
    target = H.number_of_edges() + int(frac * H.number_of_edges())
    while H.number_of_edges() < target:
        a, b = rng.sample(nodes, 2); H.add_edge(a, b)
    return H

def rewire_keep_degree(G, swaps_per_edge=10, seed=0):
    H = G.copy()
    nx.double_edge_swap(H, nswap=swaps_per_edge * H.number_of_edges(), max_tries=100 * swaps_per_edge * H.number_of_edges(), seed=seed)
    return H
