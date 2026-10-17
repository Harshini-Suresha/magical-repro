"""Double-knockout screen on a toy pathway graph (networkx).

A cell is alive if a route from 'Nutrient' to 'Biomass' survives. A gene is essential if its single
knockout kills the cell. A pair is synthetic lethal if neither gene is essential alone but losing
both kills the cell. Abstract model: no context dependence, no synthetic viability.

Run: python src/knockout_screen.py
"""
from itertools import combinations
import networkx as nx

PRESETS = {
    "Two parallel routes": [("Nutrient", "A1"), ("A1", "A2"), ("A2", "Biomass"), ("Nutrient", "B1"), ("B1", "B2"), ("B2", "Biomass")],
    "Shared steps, one fork": [("Nutrient", "X"), ("X", "Y1"), ("X", "Y2"), ("Y1", "Z"), ("Y2", "Z"), ("Z", "Biomass")],
    "Three routes, no pairs": [("Nutrient", "R1"), ("Nutrient", "R2"), ("Nutrient", "R3"), ("R1", "Biomass"), ("R2", "Biomass"), ("R3", "Biomass")],
}


def alive(G, knocked):
    H = G.copy()
    H.remove_nodes_from(knocked)
    return nx.has_path(H, "Nutrient", "Biomass")


def screen(G):
    genes = [n for n in G if n not in ("Nutrient", "Biomass")]
    essential = [g for g in genes if not alive(G, [g])]
    rest = [g for g in genes if g not in essential]
    sl = [(a, b) for a, b in combinations(rest, 2) if not alive(G, [a, b])]
    return essential, sl


if __name__ == "__main__":
    for name, edges in PRESETS.items():
        ess, sl = screen(nx.DiGraph(edges))
        print(f"{name}: essential={ess} synthetic_lethal={sl}")
