"""Generate figures/fig_080.svg .. fig_100.svg from results/*.json.

Run after the experiments: .venv/bin/python src/make_figures.py
Also refreshes the matching rows of figures/INDEX.tsv. Figures whose data is
missing keep their existing row and file untouched.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")

BG = "#fff6f9"
INK = "#3b1a2b"
MUT = "#7a5063"
LINE = "#e3cdd7"
SL = "#b4134f"
SV = "#00796b"
X0, X1, Y0, Y1 = 36.0, 290.0, 172.0, 30.0

INDEX_ROWS = {}


def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size=9, anchor="middle", color=None, weight=None, rotate=None):
    a = f'x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}"'
    if color:
        a += f' fill="{color}"'
    if weight:
        a += f' font-weight="{weight}"'
    if rotate is not None:
        a += f' transform="rotate({rotate} {x:.1f} {y:.1f})"'
    return f'<text {a}>{esc(s)}</text>'


def line(x1, y1, x2, y2, color=LINE, width=1, dash=None):
    a = f'stroke="{color}" stroke-width="{width}"'
    if dash:
        a += f' stroke-dasharray="{dash}"'
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {a}/>'


def rect(x, y, w, h, fill, opacity=None, rx=None, stroke=None, width=None):
    a = f'x="{x:.1f}" y="{y:.1f}" width="{max(w,0):.1f}" height="{max(h,0):.1f}" fill="{fill}"'
    if opacity is not None:
        a += f' fill-opacity="{opacity:.3f}"'
    if rx:
        a += f' rx="{rx}"'
    if stroke:
        a += f' stroke="{stroke}"'
    if width:
        a += f' stroke-width="{width}"'
    return f'<rect {a}/>'


def poly(points, color, width=2.2, dash=None, fill="none"):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    a = f'points="{pts}" fill="{fill}" stroke="{color}" stroke-width="{width}"'
    if dash:
        a += f' stroke-dasharray="{dash}"'
    return f'<polyline {a}/>'


def circ(x, y, r, fill, stroke=None, width=None):
    a = f'cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}"'
    if stroke:
        a += f' stroke="{stroke}" stroke-width="{width or 2}"'
    return f'<circle {a}/>'


def f2(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def frame(title, xlab=None, ylab=None):
    p = [f'<rect width="300" height="210" fill="{BG}"/>',
         txt(150, 14, title, 10.5, weight="bold", color=INK),
         line(X0, Y0, X1, Y0, MUT), line(X0, Y0, X0, Y1, MUT)]
    if xlab:
        p.append(txt(163, 203, xlab, 9, color=MUT))
    if ylab:
        p.append(txt(11, (Y0 + Y1) / 2, ylab, 9, color=MUT, rotate=-90))
    return p


def ymap(lo, hi, v):
    if hi == lo:
        hi = lo + 1
    return Y0 - (v - lo) / (hi - lo) * (Y0 - Y1)


def xmap(lo, hi, v):
    if hi == lo:
        hi = lo + 1
    return X0 + (v - lo) / (hi - lo) * (X1 - X0)


def save(num, parts, caption):
    body = "".join(parts)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 210" '
           f'width="900" height="630" font-family="Arial,sans-serif">{body}</svg>')
    with open(os.path.join(FIGDIR, f"fig_{num:03d}.svg"), "w") as fh:
        fh.write(svg)
    INDEX_ROWS[num] = ("MAGICAL results", caption)


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def cm_figure(num, cm, labels, title, caption, note=None):
    parts = [f'<rect width="300" height="210" fill="{BG}"/>',
             txt(150, 16, title, 10.5, weight="bold", color=INK),
             txt(126, 30, "predicted", 8.5, color=MUT),
             txt(14, 100, "true", 8.5, color=MUT, rotate=-90)]
    total = max(sum(sum(r) for r in cm), 1)
    peak = max(max(r) for r in cm) or 1
    x0, y0, w, h = 62.0, 36.0, 62.0, 40.0
    names = {"NOT": "other", "SL": "lethal", "SV": "viable"}
    for i, lab in enumerate(labels):
        parts.append(txt(x0 + i * w + w / 2, 32, names.get(lab, lab), 8.5, color=MUT))
        parts.append(txt(x0 - 4, y0 + i * h + h / 2 + 3, names.get(lab, lab), 8.5, anchor="end", color=MUT))
        for j in range(len(labels)):
            v = cm[i][j]
            col = SV if i == j else SL
            parts.append(rect(x0 + j * w + 1, y0 + i * h + 1, w - 2, h - 2, col, opacity=0.07 + 0.85 * v / peak, rx=4))
            parts.append(txt(x0 + j * w + w / 2, y0 + i * h + h / 2 + 4, str(v), 11, color="#ffffff" if v > 0.45 * peak else INK))
    parts.append(txt(150, 196, note or f"{total} test pairs", 9, color=MUT))
    save(num, parts, caption)


def curve_figure(num, kind, xs, ys, auc, title, caption, diag=False, prevalence=None):
    parts = frame(title, "false positive rate" if kind == "roc" else "recall",
                  "true positive rate" if kind == "roc" else "precision")
    for v in (0.25, 0.5, 0.75):
        parts.append(line(X1 - (X1 - X0) * v, Y0, X1 - (X1 - X0) * v, Y1, LINE))
        parts.append(line(X0, Y0 - (Y0 - Y1) * v, X1, Y0 - (Y0 - Y1) * v, LINE))
    parts.append(txt(X0 - 3, Y0 + 3, "0", 8, anchor="end", color=MUT))
    parts.append(txt(X0 - 3, Y1 + 6, "1", 8, anchor="end", color=MUT))
    parts.append(txt(X0, Y0 + 10, "0", 8, color=MUT))
    parts.append(txt(X1, Y0 + 10, "1", 8, color=MUT))
    if diag:
        parts.append(line(X0, Y0, X1, Y1, MUT, 1.2, dash="4 3"))
    if prevalence is not None:
        parts.append(line(X0, ymap(0, 1, prevalence), X1, ymap(0, 1, prevalence), MUT, 1.2, dash="4 3"))
    parts.append(poly([(xmap(xs[0], xs[-1], x), ymap(0, 1, y)) for x, y in zip(xs, ys)], SL))
    label = f"AUC {auc:.3f}" if kind == "roc" else f"avg precision {auc:.3f}"
    parts.append(rect(X1 - 104, Y1, 104, 16, BG, opacity=0.85))
    parts.append(txt(X1 - 4, Y1 + 12, label, 9, anchor="end", color=INK))
    save(num, parts, caption)


def bars_figure(num, labels, values, title, caption, colors=None, errs=None, ylab=None):
    parts = frame(title, None, ylab)
    n = len(values)
    w = (X1 - X0) / n
    lo, hi = 0.0, max(values) * 1.15
    parts.append(txt(X0 - 3, Y0 + 3, f2(lo), 8, anchor="end", color=MUT))
    parts.append(txt(X0 - 3, Y1 + 6, f2(hi), 8, anchor="end", color=MUT))
    for k, v in enumerate(values):
        col = (colors[k] if colors else SL)
        y = ymap(lo, hi, v)
        parts.append(rect(X0 + k * w + 4, y, w - 8, Y0 - y, col, rx=3))
        parts.append(txt(X0 + k * w + w / 2, y - 3, f"{v:.2f}" if v < 1.001 else f"{v:.2f}", 8, color=INK))
        parts.append(txt(X0 + k * w + w / 2, Y0 + 10, labels[k], 7.2, color=MUT))
        if errs is not None and errs[k] is not None:
            mid = X0 + k * w + w / 2
            parts.append(line(mid, ymap(lo, hi, v - errs[k]), mid, ymap(lo, hi, v + errs[k]), INK, 1.2))
    save(num, parts, caption)


def line_figure(num, pts, title, caption, xlab, ylab="score", extra=None, series2=None):
    parts = frame(title, xlab, ylab)
    xs = [p[0] for p in pts]
    allv = [p[1] for p in pts]
    if series2:
        allv += [p[1] for p in series2]
    lo, hi = min(allv + [0]) * 0.95, max(allv) * 1.05
    for v in (lo, (lo + hi) / 2, hi):
        parts.append(line(X0, ymap(lo, hi, v), X1, ymap(lo, hi, v), LINE))
        parts.append(txt(X0 - 3, ymap(lo, hi, v) + 3, f2(v), 8, anchor="end", color=MUT))
    parts.append(txt(X0, Y0 + 10, f2(xs[0]), 8, color=MUT))
    parts.append(txt(X1, Y0 + 10, f2(xs[-1]), 8, color=MUT))
    parts.append(poly([(xmap(xs[0], xs[-1], x), ymap(lo, hi, y)) for x, y in pts], SL))
    for x, y in pts:
        parts.append(circ(xmap(xs[0], xs[-1], x), ymap(lo, hi, y), 2.5, SL))
    if series2:
        parts.append(poly([(xmap(xs[0], xs[-1], x), ymap(lo, hi, y)) for x, y in series2], SV, dash="4 3"))
    for x, y, lab in (extra or []):
        parts.append(txt(x, y, lab, 8.5, color=INK))
    save(num, parts, caption)


def main():
    repro = load("reproduction.json")
    baselines = load("baselines.json")
    splits = load("splits.json")
    perturb = load("perturbation.json")
    candidates = load("candidates.json")

    if repro:
        s = repro["summary"]
        fd = repro["figure_data"]
        rep = s["representative_seed0"]
        cls_order = ["SL", "SV", "NOT"]

        cm_figure(80, fd["cm"], fd["labels"], "Confusion matrix, seed 0",
                  "Confusion matrix on the published split (30% test), seed 0 of 10")
        if splits:
            rp = splits.get("random_pairs") or {}
            cm81 = rp.get("confusion_seed1") or rp.get("confusion_seed0")
            if cm81:
                n81 = sum(sum(r) for r in cm81)
                cm_figure(81, cm81, ["NOT", "SL", "SV"], "Confusion matrix, random pairs",
                          f"Confusion matrix, random-pairs split, seed 1 (macro-F1 {rp['macro_f1_mean']:.3f}); seed 0 matches the reproduction figure",
                          note=f"{n81} test rows")
            gh = splits.get("gene_held_out") or {}
            if gh.get("confusion_seed0"):
                cm_figure(82, gh["confusion_seed0"], ["NOT", "SL", "SV"], "Confusion matrix, genes held out",
                          f"Confusion matrix, gene-held-out split, seed 0 (macro-F1 {gh['macro_f1_mean']:.3f})",
                          note=f"{gh['test_share_mean']*100:.1f}% of rows land in the test set")
            sh = splits.get("source_held_out") or {}
            usable = {k: v for k, v in sh.items() if v and v.get("confusion_seed0")}
            if usable:
                src = max(usable, key=lambda k: usable[k]["n_test_rows"])
                v = usable[src]
                cm_figure(83, v["confusion_seed0"], ["NOT", "SL", "SV"], f"Confusion matrix, {src} held out",
                          f"Confusion matrix holding out {src} (macro-F1 {v['macro_f1_mean']:.3f})",
                          note=f"{v['n_test_rows']} test rows from {src}")

        for k, c in enumerate(cls_order):
            r = fd[f"roc_{c}"]
            curve_figure(84 + k, "roc", r["fpr"], r["tpr"], r["auc"],
                         f"ROC, { {'SL':'lethal','SV':'viable','NOT':'other'}[c] } class, seed 0",
                         f"ROC curve for the { {'SL':'lethal','SV':'viable','NOT':'other'}[c] } class (AUC {r['auc']:.3f})",
                         diag=True)
        for k, c in enumerate(cls_order):
            r = fd[f"pr_{c}"]
            curve_figure(87 + k, "pr", r["recall"], r["precision"], r["auc"],
                         f"Precision-recall, { {'SL':'lethal','SV':'viable','NOT':'other'}[c] } class",
                         f"Precision-recall curve for the { {'SL':'lethal','SV':'viable','NOT':'other'}[c] } class (average precision {r['auc']:.3f})",
                         prevalence=None)

        f1 = [rep["per_class"][c]["f1-score"] for c in cls_order]
        bars_figure(90, ["Lethal", "Viable", "Other"], f1, "F1 per class, seed 0",
                    f"F1 by class on the published split (lethal {f1[0]:.3f}, viable {f1[1]:.3f}, other {f1[2]:.3f})",
                    colors=[SL, SV, MUT], ylab="F1")

        learn = [(p["frac"], p["accuracy"]) for p in fd["learning_curve"]]
        line_figure(93, learn, "Learning curve", "Random-forest accuracy as the share of training pairs grows",
                    "share of training pairs", ylab="accuracy")

        cal = [(p["conf"], p["acc"]) for p in fd["calibration"]]
        parts = frame("Calibration, seed 0", "predicted confidence", "observed accuracy")
        parts.append(line(X0, Y0, X1, Y1, MUT, 1.2, dash="4 3"))
        lo, hi = 0.25, 1.0
        for v in (0.4, 0.6, 0.8, 1.0):
            parts.append(line(xmap(lo, hi, v), Y0, xmap(lo, hi, v), Y1, LINE))
            parts.append(txt(xmap(lo, hi, v), Y0 + 10, f2(v), 8, color=MUT))
        parts.append(poly([(xmap(lo, hi, x), ymap(lo, hi, y)) for x, y in cal], SL))
        for x, y in cal:
            parts.append(circ(xmap(lo, hi, x), ymap(lo, hi, y), 2.5, SL))
        save(98, parts, "Calibration on the published split: predicted confidence against observed accuracy (seed 0)")

        buckets = fd["degree_buckets"]
        bars_figure(99, [b["bucket"] for b in buckets], [b["accuracy"] for b in buckets],
                    "Accuracy by degree bucket", "Accuracy by average-degree bucket on the published split (seed 0)",
                    colors=[SL] * len(buckets), ylab="accuracy")

    if baselines:
        order = ["magical_reextract_rf", "rf_all_columns", "logistic_regression",
                 "gradient_boosting", "degree_stump", "path_stump"]
        names = ["RF, 5 paper features (rerun)", "RF, all 20 columns", "Logistic regression",
                 "Gradient boosting", "Degree stump", "Path stump"]
        vals = [0.81] + [baselines[k]["accuracy_mean"] for k in order]
        errs = [None] + [baselines[k]["accuracy_sd"] for k in order]
        labs = ["Paper\n(~81%)", "RF 5 feat", "RF 20 col", "Logistic", "Grad boost", "Deg stump", "Path stump"]
        bars_figure(91, labs, vals, "Accuracy by model",
                    "Accuracy on the published split: paper model (5 features), its rerun and the baselines", 
                    colors=[MUT] + [SL] * 6, errs=errs, ylab="accuracy")

    if splits:
        rp = splits["random_pairs"]
        gh = splits["gene_held_out"]
        sh = [(v["mcc_mean"], k) for k, v in splits["source_held_out"].items() if v and v.get("mcc_mean")]
        src_mean = sum(m for m, _ in sh) / len(sh) if sh else 0.0
        vals = [rp["mcc_mean"], gh["mcc_mean"], src_mean]
        bars_figure(92, ["Random pairs", "Genes held out", "Source held out"], vals, "MCC by split",
                    f"MCC on the published split and the two stricter splits (random {vals[0]:.3f}, gene-held-out {vals[1]:.3f}, source-held-out {vals[2]:.3f})",
                    colors=[SL, SV, MUT], ylab="MCC")

    if perturb:
        xset = sorted({v["param"] for v in perturb.values() if v["family"] == "node"})
        fam = {
            "node": (94, "Macro-F1 against genes removed", "genes removed"),
            "edge": (95, "Macro-F1 against interactions removed", "interactions removed"),
            "noise": (96, "Macro-F1 against noise edges added", "noise edges added"),
        }
        for family, (num, title, xlab) in fam.items():
            pts = sorted([(v["param"], v["macro_f1"]) for v in perturb.values() if v["family"] == family])
            drops = sorted([(v["param"], v["dropped_share"]) for v in perturb.values() if v["family"] in (family, "none")])
            base = perturb["baseline"]["macro_f1"]
            base_drop = perturb["baseline"]["dropped_share"]
            pts = [(0.0, base)] + pts
            n = num
            parts = frame(title, f"share of {xlab}", "macro-F1")
            xs = [p[0] for p in pts]
            hi = max(1.0, max(p[1] for p in pts) * 1.05)
            for v in (0.2, 0.4, 0.6, 0.8, 1.0):
                if v <= hi:
                    parts.append(line(X0, ymap(0, hi, v), X1, ymap(0, hi, v), LINE))
                    parts.append(txt(X0 - 3, ymap(0, hi, v) + 3, f2(v), 8, anchor="end", color=MUT))
            parts.append(txt(X0, Y0 + 10, "0", 8, color=MUT))
            parts.append(txt(X1, Y0 + 10, f"{int(xs[-1]*100)}%", 8, color=MUT))
            parts.append(poly([(xmap(xs[0], xs[-1], x), ymap(0, hi, y)) for x, y in pts], SL))
            for x, y in pts:
                parts.append(circ(xmap(xs[0], xs[-1], x), ymap(0, hi, y), 2.5, SL))
            ds = [d for d in drops if d[0] > 0]
            if ds:
                parts.append(poly([(xmap(xs[0], xs[-1], x), ymap(0, hi, y)) for x, y in ds], SV, dash="4 3"))
                parts.append(circ(xmap(xs[0], xs[-1], xs[-1]), ymap(0, hi, ds[-1][1]), 2.5, SV))
            parts.append(rect(X0 + 4, Y1, 116, 15, BG, opacity=0.85))
            parts.append(txt(X0 + 6, Y1 + 11, "solid: macro-F1", 8, anchor="start", color=SL))
            parts.append(txt(X0 + 116, Y1 + 11, "dashed: pairs dropped", 8, anchor="end", color=SV))
            cap = (f"{title} on the published split: macro-F1 {base:.3f} at 0% and {pts[-1][1]:.3f} at {int(pts[-1][0]*100)}%")
            if ds and ds[-1][1] > base_drop + 0.005:
                cap += f", while the share of test pairs missing a gene rises from {base_drop:.0%} to {ds[-1][1]:.0%}"
            save(n, parts, cap)

        rw = sorted([(v["param"], v["macro_f1"]) for v in perturb.values() if v["family"] == "rewire"])
        base = perturb["baseline"]["macro_f1"]
        vals = [base] + [v for _, v in rw]
        bars_figure(97, ["Original", "Rewired 1x", "Rewired 5x"], vals, "Macro-F1 against rewiring",
                    "Macro-F1 after degree-preserving rewiring (1x and 5x edge count swaps)",
                    colors=[MUT, SL, SL], ylab="macro-F1")

    if candidates:
        hist = candidates["stability_histogram"]
        ks = sorted(hist, key=int)
        vals = [hist[k] for k in ks]
        bars_figure(100, [f"{k} of {candidates['n_variants']}" for k in ks], vals,
                    "Candidate stability",
                    f"How often the {candidates['pool']} NOT-pool pairs are predicted interacting across five network variants; {candidates['stable_candidates']} pairs are stable candidates",
                    colors=[SV if int(k) >= 4 else MUT for k in ks], ylab="pairs")

    index_path = os.path.join(FIGDIR, "INDEX.tsv")
    with open(index_path) as fh:
        rows = fh.read().splitlines()
    out = []
    for r in rows:
        parts = r.split("\t")
        if parts and parts[0].startswith("fig_") and parts[0][4:7].isdigit():
            num = int(parts[0][4:7])
            if num in INDEX_ROWS:
                src, cap = INDEX_ROWS[num]
                out.append(f"{parts[0]}\t{src}\t{cap}")
                continue
        out.append(r)
    with open(index_path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {len(INDEX_ROWS)} figures:", ", ".join(f"{n:03d}" for n in sorted(INDEX_ROWS)))


if __name__ == "__main__":
    main()
