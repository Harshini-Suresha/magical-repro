"""Reproduction of the published MAGICAL-core evaluation.

Pipeline mirrored from MAGICAL-CORE/magical-core.R:
  - balanced training table shipped by the authors (SMOTE, perc.under=200)
  - 70/30 split by independent Bernoulli draws via sample(2, n, prob=c(0.7,0.3))
  - random forest, 500 trees (R default), 3 classes SL / SV / NOT
  - features = the five the combined script selects after dropping gene1/gene2:
    avgneighbor2, avgbet, avgtriangle, shortest_path, adhesion

Also reruns with the three reproducible features only (avgneighbor2, avgbet,
avgtriangle) because the shipped shortest_path and adhesion columns cannot be
reproduced from the shipped edge list (see results/network_verification.json).

Run: .venv/bin/python src/exp_reproduction.py
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             roc_curve, precision_recall_curve, auc)

from network_io import load_balanced

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

PAPER_FEATURES = ["avgneighbor2", "avgbet", "avgtriangle", "shortest_path", "adhesion"]
CLEAN_FEATURES = ["avgneighbor2", "avgbet", "avgtriangle"]
LABELS = ["NOT", "SL", "SV"]


def one_run(df, features, seed, mode="bernoulli"):
    rng = np.random.default_rng(seed)
    if mode == "bernoulli":
        test_mask = rng.random(len(df)) < 0.30
    else:
        rng2 = np.random.RandomState(seed)
        test_mask = np.zeros(len(df), dtype=bool)
        for cls in LABELS:
            idx = np.where(df["gi"].values == cls)[0]
            take = rng2.permutation(len(idx))[: int(0.30 * len(idx))]
            test_mask[idx[take]] = True
    train, test = df[~test_mask], df[test_mask]
    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)
    clf.fit(train[features], train["gi"])
    pred = clf.predict(test[features])
    proba = clf.predict_proba(test[features])
    out = {
        "accuracy": accuracy_score(test["gi"], pred),
        "macro_f1": f1_score(test["gi"], pred, average="macro"),
        "mcc": matthews_corrcoef(test["gi"], pred),
        "per_class": classification_report(test["gi"], pred, output_dict=True, zero_division=0),
    }
    return out, test, pred, proba, clf


def main():
    df = load_balanced()
    os.makedirs(RESULTS, exist_ok=True)
    summary = {}

    for name, feats in [("paper_features", PAPER_FEATURES), ("clean_features", CLEAN_FEATURES)]:
        accs, f1s, mccs = [], [], []
        for seed in range(20):
            out, *_ = one_run(df, feats, seed)
            accs.append(out["accuracy"])
            f1s.append(out["macro_f1"])
            mccs.append(out["mcc"])
        summary[name] = {
            "features": feats,
            "accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_sd": float(np.std(f1s)),
            "mcc_mean": float(np.mean(mccs)), "mcc_sd": float(np.std(mccs)),
        }
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in summary[name].items()})

    # representative run saved in full for figures
    out, test, pred, proba, clf = one_run(df, PAPER_FEATURES, 0)
    cm = confusion_matrix(test["gi"], pred, labels=LABELS)
    summary["representative_seed0"] = {
        "confusion": cm.tolist(),
        "per_class": {k: {m: v[m] for m in ("precision", "recall", "f1-score", "support")}
                      for k, v in out["per_class"].items() if k in LABELS},
        "accuracy": out["accuracy"], "macro_f1": out["macro_f1"], "mcc": out["mcc"],
    }

    # stratified split reference
    strat_res = [one_run(df, PAPER_FEATURES, s, mode="stratified")[0] for s in range(10)]
    summary["stratified_split"] = {
        "accuracy_mean": float(np.mean([r["accuracy"] for r in strat_res])),
        "macro_f1_mean": float(np.mean([r["macro_f1"] for r in strat_res])),
        "accuracy_sd": float(np.std([r["accuracy"] for r in strat_res])),
        "macro_f1_sd": float(np.std([r["macro_f1"] for r in strat_res])),
    }

    # curve data for the representative run
    figdata = {"labels": LABELS, "cm": cm.tolist()}
    for i, cls in enumerate(LABELS):
        y = (test["gi"].values == cls).astype(int)
        p = proba[:, i]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        figdata[f"roc_{cls}"] = {"fpr": fpr[:: max(1, len(fpr) // 60)].tolist(),
                                 "tpr": tpr[:: max(1, len(tpr) // 60)].tolist(),
                                 "auc": float(auc(fpr, tpr))}
        figdata[f"pr_{cls}"] = {"recall": rec[:: max(1, len(rec) // 60)].tolist(),
                                "precision": prec[:: max(1, len(prec) // 60)].tolist(),
                                "auc": float(auc(rec, prec))}
    conf = proba.max(axis=1)
    correct = (pred == test["gi"].values).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() > 10:
            cal.append({"conf": float(conf[m].mean()), "acc": float(correct[m].mean()), "n": int(m.sum())})
    figdata["calibration"] = cal

    # learning curve
    lc = []
    rng = np.random.default_rng(0)
    order = rng.permutation(len(df))
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        accs = []
        for seed in range(3):
            r = np.random.default_rng(seed)
            keep = order[: int(frac * len(order))]
            sub = df.iloc[keep]
            m = r.random(len(sub)) < 0.30
            clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
            clf.fit(sub[~m][PAPER_FEATURES], sub[~m]["gi"])
            accs.append(accuracy_score(sub[m]["gi"], clf.predict(sub[m][PAPER_FEATURES])))
        lc.append({"frac": frac, "accuracy": float(np.mean(accs))})
    figdata["learning_curve"] = lc

    # performance by degree bucket
    test = test.copy()
    test["degbin"] = pd.cut(test["avgdeg"], [0, 5, 15, 40, 1000], labels=["<5", "5-15", "15-40", ">40"])
    deg_stats = []
    for b in ["<5", "5-15", "15-40", ">40"]:
        m = test["degbin"] == b
        if m.sum() > 0:
            deg_stats.append({"bucket": b, "n": int(m.sum()),
                              "accuracy": float(accuracy_score(test["gi"][m], pred[m]))})
    figdata["degree_buckets"] = deg_stats

    with open(os.path.join(RESULTS, "reproduction.json"), "w") as fh:
        json.dump({"summary": summary, "figure_data": figdata}, fh, indent=1)
    print("saved results/reproduction.json")
    print("representative confusion matrix (rows NOT,SL,SV):")
    print(cm)


if __name__ == "__main__":
    main()
