"""Robustness experiments on the real network.

Protocol: the model is trained once on features re-extracted from the original
network for the training pairs. Each perturbation is applied to the network
only; features for the test pairs are recomputed with the same extraction and
the fixed model is scored again. Pairs whose gene was removed are dropped and
their share is reported.

One resumable job per network variant; pair features for the labelled pairs
and the NOT candidate pool are cached in results/perturb_parts/<tag>.csv, so
re-running skips finished variants.

Also produces candidate-pair stability: the NOT pool (random non-interacting
pairs) is scored under the original and four perturbed networks; a pair is a
candidate when it is predicted the same interacting class in most variants.

Run: .venv/bin/python -u src/exp_perturb.py
Saves results/perturbation.json and results/candidates.json
"""
import json
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import igraph as ig
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from network_io import load_network, load_balanced, DATA
from features_extract import pair_features, node_properties, FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PARTS = os.path.join(RESULTS, "perturb_parts")
N_PROC = max(2, min(8, (os.cpu_count() or 4) - 1))

SETTINGS = ([("baseline", "none", 0.0)] +
            [(f"node-{int(f*100)}", "node", f) for f in (0.05, 0.10, 0.20, 0.30)] +
            [(f"edge-{int(f*100)}", "edge", f) for f in (0.05, 0.10, 0.20, 0.30)] +
            [(f"noise-{int(f*100)}", "noise", f) for f in (0.05, 0.10, 0.20, 0.30)] +
            [("rewire-1x", "rewire", 1.0), ("rewire-5x", "rewire", 5.0)])

CAND_SETTINGS = [("baseline", "none", 0.0), ("node-10", "node", 0.10),
                 ("edge-10", "edge", 0.10), ("noise-10", "noise", 0.10),
                 ("rewire-1x", "rewire", 1.0)]


def perturb(g, kind, param, seed=0):
    rng = np.random.default_rng(seed)
    H = g.copy()
    if kind == "none":
        return H
    if kind == "node":
        n = H.vcount()
        rm = np.sort(rng.choice(n, size=int(param * n), replace=False).astype(int))
        H.delete_vertices(rm.tolist())
    elif kind == "edge":
        m = H.ecount()
        rm = np.sort(rng.choice(m, size=int(param * m), replace=False).astype(int))
        H.delete_edges(rm.tolist())
    elif kind == "noise":
        m = H.ecount()
        target = int(param * m)
        added = 0
        while added < target:
            a, b = rng.integers(0, H.vcount(), size=2)
            if a != b and H.get_eid(int(a), int(b), error=False) < 0:
                H.add_edge(int(a), int(b))
                added += 1
    elif kind == "rewire":
        ig.set_random_number_generator(random.Random(seed))
        H.rewire(n=int(param * H.ecount()), mode="simple")
    return H


def extract_setting(job):
    setting, pairs = job
    tag, kind, param = setting
    out_path = os.path.join(PARTS, f"{tag}.csv")
    if os.path.exists(out_path):
        return tag, "cached", 0, 0.0
    try:
        t0 = time.time()
        g = perturb(load_network(), kind, param)
        props = node_properties(g)
        feats = pair_features(g, pairs, props=props)
        tmp = out_path + ".tmp"
        feats.to_csv(tmp, index=False)
        os.replace(tmp, out_path)
        return tag, "done", len(feats), time.time() - t0
    except Exception as exc:
        return tag, f"error: {exc!r}", 0, time.time() - t0


def part_frame(tag):
    return pd.read_csv(os.path.join(PARTS, f"{tag}.csv"))


def aligned(tag, pairs):
    f = part_frame(tag).drop_duplicates(subset=["gene1", "gene2"], keep="first")
    return f.set_index(pd.MultiIndex.from_arrays([f["gene1"], f["gene2"]])).reindex(
        pd.MultiIndex.from_arrays([pairs["gene1"], pairs["gene2"]]))


def main():
    os.makedirs(PARTS, exist_ok=True)
    t_start = time.time()

    df = load_balanced()
    rng = np.random.default_rng(0)
    test_mask = rng.random(len(df)) < 0.30

    def unique_pairs(frame):
        p = frame[["gene1", "gene2"]].copy()
        p["a"] = p[["gene1", "gene2"]].min(axis=1)
        p["b"] = p[["gene1", "gene2"]].max(axis=1)
        return p[["a", "b"]].drop_duplicates().reset_index(drop=True).rename(
            columns={"a": "gene1", "b": "gene2"})

    train_pairs = unique_pairs(df[~test_mask])
    test_pairs = unique_pairs(df[test_mask])
    all_pairs = pd.concat([train_pairs, test_pairs]).drop_duplicates().reset_index(drop=True)

    not_pool = pd.read_csv(os.path.join(DATA, "not0.pairwiseprop.csv"))
    not_pool["g1"] = not_pool["gene1"].astype("int64")
    not_pool["g2"] = not_pool["gene2"].astype("int64")
    not_pool["a"] = not_pool[["g1", "g2"]].min(axis=1)
    not_pool["b"] = not_pool[["g1", "g2"]].max(axis=1)
    not_pool = not_pool[["a", "b"]].drop_duplicates().reset_index(drop=True).rename(
        columns={"a": "gene1", "b": "gene2"})
    train_keys = set(map(tuple, all_pairs.values))
    not_pool = not_pool[~not_pool.apply(lambda r: (r["gene1"], r["gene2"]) in train_keys, axis=1)]
    print(f"unique pairs: train {len(train_pairs)}, test {len(test_pairs)}, total {len(all_pairs)}")
    print("candidate pool (NOT pairs unseen in labels):", len(not_pool), flush=True)

    pairs_all = pd.concat([all_pairs, not_pool], ignore_index=True)
    jobs = [(s, pairs_all) for s in SETTINGS]
    t0 = time.time()
    status = {}
    with Pool(N_PROC) as pool:
        for tag, state, n_rows, secs in pool.imap_unordered(extract_setting, jobs):
            status[tag] = state
            extra = f" ({n_rows} rows, {secs:.0f}s)" if n_rows else ""
            print(f"[{time.strftime('%H:%M:%S')}] {tag}: {state}{extra}", flush=True)
    failed = {t: s for t, s in status.items() if s.startswith("error")}
    if failed or len(status) != len(SETTINGS):
        raise SystemExit(f"variant extraction failed: {failed or sorted(set(s[0] for s in SETTINGS) - set(status))}")
    print(f"pair features for {len(status)} variants in {time.time()-t0:.0f}s", flush=True)

    train_rows = df[~test_mask].copy()
    train_rows["a"] = train_rows[["gene1", "gene2"]].min(axis=1)
    train_rows["b"] = train_rows[["gene1", "gene2"]].max(axis=1)
    base_feats = aligned("baseline", train_pairs)[FEATURES].reset_index()
    train_joined = train_rows.merge(base_feats, left_on=["a", "b"], right_on=["gene1", "gene2"],
                                    how="inner", suffixes=("_shipped", ""))
    train_joined = train_joined[train_joined[FEATURES].notna().all(axis=1)]
    model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0)
    model.fit(train_joined[FEATURES], train_joined["gi"])
    train_acc = accuracy_score(train_joined["gi"], model.predict(train_joined[FEATURES]))
    print("re-extracted baseline train accuracy:", round(train_acc, 4),
          f"({len(train_joined)}/{len(train_rows)} train rows have both genes in the network)", flush=True)

    import pickle
    with open(os.path.join(RESULTS, "model_reextracted.pkl"), "wb") as fh:
        pickle.dump(model, fh)

    test_rows = df[test_mask].copy()
    test_rows["a"] = test_rows[["gene1", "gene2"]].min(axis=1)
    test_rows["b"] = test_rows[["gene1", "gene2"]].max(axis=1)

    curves = {}
    for tag, kind, param in SETTINGS:
        f = aligned(tag, test_pairs)[FEATURES].reset_index()
        joined = test_rows.merge(f, left_on=["a", "b"], right_on=["gene1", "gene2"],
                                 how="inner", suffixes=("_shipped", ""))
        joined = joined[joined[FEATURES].notna().all(axis=1)]
        y_true = joined["gi"]
        y_pred = model.predict(joined[FEATURES])
        curves[tag] = {
            "family": kind, "param": param,
            "n_rows": int(len(joined)),
            "dropped_share": float(1 - len(joined) / len(test_rows)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        }
        print(tag, {k: round(v, 4) if isinstance(v, float) else v for k, v in curves[tag].items()}, flush=True)

    with open(os.path.join(RESULTS, "perturbation.json"), "w") as fh:
        json.dump(curves, fh, indent=1)

    n_variants = len(CAND_SETTINGS)
    preds = np.full((n_variants, len(not_pool)), None, dtype=object)
    confs = np.full((n_variants, len(not_pool)), np.nan)
    for vi, (tag, kind, param) in enumerate(CAND_SETTINGS):
        f = aligned(tag, not_pool)
        mask = f[FEATURES].notna().all(axis=1).to_numpy()
        proba = model.predict_proba(f.loc[mask, FEATURES])
        preds[vi, mask] = model.classes_[proba.argmax(axis=1)]
        confs[vi, mask] = proba.max(axis=1)

    available = np.array([[p is not None for p in col] for col in preds.T])
    interacting = np.array([[(p is not None) and (p != "NOT") for p in col] for col in preds.T])
    same_class = []
    for col in preds.T:
        vals = {p for p in col if p is not None}
        same_class.append(len(vals) == 1 and len(vals) > 0)
    same_class = np.array(same_class)
    n_avail = available.sum(axis=1)
    n_int = interacting.sum(axis=1)
    mean_conf = np.nanmean(np.where(interacting, confs.T, np.nan), axis=1)
    first_class = np.array([next((p for p in col if p is not None), "NOT") for col in preds.T])

    stable = (n_avail >= 4) & (n_int >= 4) & same_class & (first_class != "NOT")
    out = not_pool.copy()
    out["class"] = first_class
    out["n_variants_available"] = n_avail
    out["n_variants_interacting"] = n_int
    out["same_class"] = same_class
    out["mean_confidence_when_interacting"] = mean_conf
    cand = out[stable].copy()
    cand = cand.sort_values("mean_confidence_when_interacting", ascending=False)
    cand["mean_confidence_when_interacting"] = cand["mean_confidence_when_interacting"].round(3)
    summary = {
        "pool": int(len(not_pool)),
        "n_variants": n_variants,
        "predicted_interacting_in_any_variant": int(interacting.any(axis=1).sum()),
        "stable_candidates": int(stable.sum()),
        "stability_histogram": {str(k): int((n_int == k).sum()) for k in range(n_variants + 1)},
        "top_sl": cand[cand["class"] == "SL"].head(15).to_dict(orient="records"),
        "top_sv": cand[cand["class"] == "SV"].head(15).to_dict(orient="records"),
        "candidate_frame_sample": cand.head(200).to_dict(orient="records"),
    }
    with open(os.path.join(RESULTS, "candidates.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("stable candidates:", int(stable.sum()), "of pool", len(not_pool), flush=True)
    print(f"total wall time {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
