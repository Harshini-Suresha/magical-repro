"""Build site/results.js (window.RES) from results/*.json.

Run after the experiments: .venv/bin/python src/build_site.py
Missing files become nulls in the payload; the site falls back to placeholders.
"""
import json
import os
import shutil
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SITE = os.path.join(ROOT, "site")

PAPER_ACCURACY = 0.81


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        print("missing:", name)
        return None
    with open(path) as fh:
        return json.load(fh)


def main():
    repro = load("reproduction.json")
    baselines = load("baselines.json")
    splits = load("splits.json")
    perturb = load("perturbation.json")
    candidates = load("candidates.json")
    verify = load("network_verification.json")

    res = {
        "generated": time.strftime("%Y-%m-%d"),
        "paper": {"accuracy": PAPER_ACCURACY,
                  "note": "published MAGICAL-core reports ~81% accuracy (PLOS Comput Biol 2024)"},
        "repro": None, "baselines": baselines, "splits": splits,
        "perturb": perturb, "candidates": None, "verify": verify,
    }

    if repro:
        s = repro["summary"]
        fd = repro["figure_data"]
        rep = s["representative_seed0"]
        classes = ["SL", "SV", "NOT"]
        res["repro"] = {
            "paper_acc": [s["paper_features"]["accuracy_mean"], s["paper_features"]["accuracy_sd"]],
            "paper_f1": [s["paper_features"]["macro_f1_mean"], s["paper_features"]["macro_f1_sd"]],
            "paper_mcc": [s["paper_features"]["mcc_mean"], s["paper_features"]["mcc_sd"]],
            "clean_acc": [s["clean_features"]["accuracy_mean"], s["clean_features"]["accuracy_sd"]],
            "clean_f1": [s["clean_features"]["macro_f1_mean"], s["clean_features"]["macro_f1_sd"]],
            "clean_mcc": [s["clean_features"]["mcc_mean"], s["clean_features"]["mcc_sd"]],
            "stratified": {"acc": s["stratified_split"]["accuracy_mean"],
                           "f1": s["stratified_split"]["macro_f1_mean"]},
            "cm": fd["cm"],
            "labels": fd["labels"],
            "per_class": {c: {"p": rep["per_class"][c]["precision"],
                              "r": rep["per_class"][c]["recall"],
                              "f": rep["per_class"][c]["f1-score"],
                              "n": rep["per_class"][c]["support"]} for c in rep["per_class"]},
            "roc": {c: fd["roc_" + c] for c in classes},
            "pr": {c: fd["pr_" + c] for c in classes},
            "calibration": fd["calibration"],
            "learning": fd["learning_curve"],
            "degree_buckets": fd["degree_buckets"],
        }

    if candidates:
        res["candidates"] = {
            "pool": candidates["pool"],
            "n_variants": candidates["n_variants"],
            "stable": candidates["stable_candidates"],
            "any_interacting": candidates["predicted_interacting_in_any_variant"],
            "histogram": candidates["stability_histogram"],
            "top_sl": candidates["top_sl"],
            "top_sv": candidates["top_sv"],
        }

    out = "window.RES = " + json.dumps(res, indent=1) + ";\n"
    os.makedirs(SITE, exist_ok=True)
    with open(os.path.join(SITE, "results.js"), "w") as fh:
        fh.write(out)
    print("wrote site/results.js", len(out), "bytes")

    csv_src = os.path.join(RESULTS, "candidates_annotated.csv")
    if os.path.exists(csv_src):
        shutil.copyfile(csv_src, os.path.join(SITE, "candidates.csv"))
        print("copied candidates.csv to site/")
    else:
        print("missing: candidates_annotated.csv (run src/annotate_candidates.py)")


if __name__ == "__main__":
    main()
