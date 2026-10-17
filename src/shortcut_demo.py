"""Shortcut lab on synthetic data: can a gene memoriser score well without learning pair-level biology?

Run: python src/shortcut_demo.py
Not MAGICAL data: genes have hidden propensities, labels mix gene-level and pair-level effects.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def trial(strength, noise, rng, G=80, held_n=16):
    h = rng.random(G)
    i, j = np.triu_indices(G, 1)
    u = rng.normal(size=i.size)
    z = strength * 3 * (h[i] + h[j] - 1) + (1 - strength) * u
    x = u + noise * rng.normal(size=i.size)          # observable pair-level signal
    y = (z > np.quantile(z, 0.75)).astype(int)

    def memoriser(tr, te):
        tot, cnt = np.zeros(G), np.zeros(G)
        for idx in (i[tr], j[tr]):
            np.add.at(tot, idx, y[tr]); np.add.at(cnt, idx, 1)
        rate = np.where(cnt > 0, tot / np.maximum(cnt, 1), y[tr].mean())
        return auc(y[te], (rate[i[te]] + rate[j[te]]) / 2)

    perm = rng.permutation(i.size); cut = int(0.8 * i.size)
    tr_r, te_r = perm[:cut], perm[cut:]
    held = np.isin(np.arange(G), rng.choice(G, held_n, replace=False))
    tr_g = ~held[i] & ~held[j]
    te_g = held[i] & held[j]
    return [memoriser(tr_r, te_r), memoriser(tr_g, te_g), auc(y[te_r], x[te_r]), auc(y[te_g], x[te_g])]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("gene-effect  memo(random) memo(held-out) pair(random) pair(held-out)")
    for s in np.linspace(0, 1, 6):
        r = np.nanmean([trial(s, 0.8, rng) for _ in range(10)], axis=0)
        print(f"{s:10.0%}  " + "  ".join(f"{v:12.2f}" for v in r))
