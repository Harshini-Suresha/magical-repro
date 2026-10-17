"""Baselines and stricter splits on the published feature table.

Baselines (same published Bernoulli 70/30 split, 10 seeds):
  - random forest with the five published features (rerun reference)
  - logistic regression, standardised
  - histogram gradient boosting
  - single-split decision stump on average degree (a degree-only heuristic)
  - single-split decision stump on shortest path (a distance-only heuristic)
  - random forest using all 20 network columns in the shipped table

Stricter splits (published five features, 10 seeds each):
  - random pairs (the published protocol)
  - gene held out: test pairs use only genes unseen in training pairs
  - source held out: train on two of {CGIdb, BioGRID, SynLethDB}, test on the third

Run: .venv/bin/python src/exp_baselines.py
Saves results/baselines.json and results/splits.json
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from network_io import load_balanced, load_pairs, SOURCE_OF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PAPER_FEATURES = ["avgneighbor2", "avgbet", "avgtriangle", "shortest_path", "adhesion"]
ALL_NET_FEATURES = ["avgdeg", "avgbet", "avgclose", "avgcore", "avgcons", "avgecc",
                    "avgeigen_cen", "avghub_score", "avgneighbor1", "avgneighbor2", "avgneighbor3",
                    "avgneighbor4", "avgneighbor5", "avgneighbor6", "avgtriangle",
                    "avgcommon_neighbor", "comm.detect", "shortest_path", "cohesion", "adhesion"]


def bernoulli_mask(n, seed, frac=0.30):
    return np.random.default_rng(seed).random(n) < frac


def evaluate(model, train, test):
    model.fit(train[0], train[1])
    pred = model.predict(test[0])
    return {"accuracy": float(accuracy_score(test[1], pred)),
            "macro_f1": float(f1_score(test[1], pred, average="macro")),
            "mcc": float(matthews_corrcoef(test[1], pred)),
            "confusion": confusion_matrix(test[1], pred, labels=["NOT", "SL", "SV"]).tolist(),
            "per_class": {k: {m: float(v[m]) for m in ("precision", "recall", "f1-score")}
                          for k, v in classification_report(test[1], pred, output_dict=True,
                                                             zero_division=0).items()
                          if k in ("NOT", "SL", "SV")}}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    df = load_balanced()

    models = {
        "magical_reextract_rf": (lambda seed: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed), PAPER_FEATURES),
        "logistic_regression": (lambda seed: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)), PAPER_FEATURES),
        "gradient_boosting": (lambda seed: HistGradientBoostingClassifier(random_state=seed), PAPER_FEATURES),
        "degree_stump": (lambda seed: DecisionTreeClassifier(max_depth=1, random_state=seed), ["avgdeg"]),
        "path_stump": (lambda seed: DecisionTreeClassifier(max_depth=1, random_state=seed), ["shortest_path"]),
        "rf_all_columns": (lambda seed: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed), ALL_NET_FEATURES),
    }

    baselines = {}
    for name, (factory, feats) in models.items():
        accs, f1s = [], []
        for seed in range(10):
            m = bernoulli_mask(len(df), seed)
            r = evaluate(factory(seed), (df[~m][feats], df[~m]["gi"]), (df[m][feats], df[m]["gi"]))
            accs.append(r["accuracy"])
            f1s.append(r["macro_f1"])
        baselines[name] = {
            "features": feats,
            "accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_sd": float(np.std(f1s)),
        }
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in baselines[name].items()})

    with open(os.path.join(RESULTS, "baselines.json"), "w") as fh:
        json.dump(baselines, fh, indent=1)

    # ---------- splits ----------
    pairs = load_pairs()
    keys = {}  # (g1,g2) sorted -> set of sources
    for _, r in pairs.iterrows():
        k = (min(int(r["gene1"]), int(r["gene2"])), max(int(r["gene1"]), int(r["gene2"])))
        keys.setdefault(k, set()).add(r["source"])

    def source_of(row):
        k = (min(int(row["gene1"]), int(row["gene2"])), max(int(row["gene1"]), int(row["gene2"])))
        return keys.get(k, set())

    df = df.copy()
    df["sources"] = [source_of(r) for _, r in df.iterrows()]
    df["primary_source"] = df["sources"].map(
        lambda s: "CGIdb" if "CGIdb" in s else ("BioGRID" if "BioGRID" in s else ("SynLethDB" if "SynLethDB" in s else "NOT")))

    splits = {}

    # random pairs (published protocol) reference
    accs, f1s, mccs, cms = [], [], [], []
    for seed in range(10):
        m = bernoulli_mask(len(df), seed)
        r = evaluate(RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed),
                     (df[~m][PAPER_FEATURES], df[~m]["gi"]), (df[m][PAPER_FEATURES], df[m]["gi"]))
        accs.append(r["accuracy"]); f1s.append(r["macro_f1"]); mccs.append(r["mcc"])
        if seed == 0:
            cms.append(r["confusion"])
    splits["random_pairs"] = {"accuracy_mean": float(np.mean(accs)), "macro_f1_mean": float(np.mean(f1s)),
                              "accuracy_sd": float(np.std(accs)), "macro_f1_sd": float(np.std(f1s)),
                              "mcc_mean": float(np.mean(mccs)), "n_eval": int(len(df)),
                              "confusion_seed0": cms[0] if cms else None}

    # gene held out
    accs, f1s, mccs, fracs, cms = [], [], [], [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        genes = pd.unique(pd.concat([df["gene1"], df["gene2"]]))
        held = set(rng.choice(genes, size=int(0.20 * len(genes)), replace=False).tolist())
        in_a = df["gene1"].isin(held)
        in_b = df["gene2"].isin(held)
        test = df[in_a & in_b]
        train = df[~in_a & ~in_b]
        fracs.append(len(test) / len(df))
        r = evaluate(RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed),
                     (train[PAPER_FEATURES], train["gi"]), (test[PAPER_FEATURES], test["gi"]))
        accs.append(r["accuracy"]); f1s.append(r["macro_f1"]); mccs.append(r["mcc"])
        if seed == 0:
            cms.append(r["confusion"])
    splits["gene_held_out"] = {"accuracy_mean": float(np.mean(accs)), "macro_f1_mean": float(np.mean(f1s)),
                               "accuracy_sd": float(np.std(accs)), "macro_f1_sd": float(np.std(f1s)),
                               "mcc_mean": float(np.mean(mccs)),
                               "test_share_mean": float(np.mean(fracs)),
                               "confusion_seed0": cms[0] if cms else None}

    # source held out
    src_accs = {}
    for held_src in ["CGIdb", "BioGRID", "SynLethDB"]:
        accs, f1s, mccs, n_tests = [], [], [], []
        cm0 = None
        for seed in range(5):
            test = df[df["primary_source"] == held_src]
            train = df[~df["primary_source"].isin([held_src])]
            if len(test) < 50:
                continue
            accs_ = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)
            try:
                r = evaluate(accs_, (train[PAPER_FEATURES], train["gi"]), (test[PAPER_FEATURES], test["gi"]))
            except ValueError as ex:
                print("skip", held_src, ex); continue
            accs.append(r["accuracy"]); f1s.append(r["macro_f1"]); mccs.append(r["mcc"])
            n_tests.append(len(test))
            if seed == 0:
                cm0 = r["confusion"]
        src_accs[held_src] = {"accuracy_mean": float(np.mean(accs)) if accs else None,
                              "macro_f1_mean": float(np.mean(f1s)) if f1s else None,
                              "mcc_mean": float(np.mean(mccs)) if mccs else None,
                              "n_test_rows": int(np.mean(n_tests)) if n_tests else 0,
                              "confusion_seed0": cm0}
        print("hold out", held_src, src_accs[held_src])
    splits["source_held_out"] = src_accs

    with open(os.path.join(RESULTS, "splits.json"), "w") as fh:
        json.dump(splits, fh, indent=1)
    print("saved results/baselines.json and results/splits.json")


if __name__ == "__main__":
    main()
