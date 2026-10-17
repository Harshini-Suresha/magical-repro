# MAGICAL reproduction

Reproduction and robustness test of MAGICAL (PPI-network classifier for synthetic lethal / synthetic viable gene pairs).

- Paper: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012336
- Code: https://github.com/Anubhagithub/MAGICAL
- Database: http://sls.uohyd.ac.in/new/magicaldb

## Layout
- `site/index.html` project website (open in any browser, no build step)
- `figures/` 100 SVG figures (fig_001.svg to fig_100.svg) with `INDEX.tsv` listing source and caption; the same figures are in the website gallery
- `src/` starter helpers: splits, metrics, network perturbations, features, and `shortcut_demo.py` (synthetic demo that a gene memoriser scores well on random splits but drops to chance on gene-held-out splits; run `python src/shortcut_demo.py`) `knockout_screen.py` (toy double-knockout screen showing where synthetic lethality comes from: `python src/knockout_screen.py`) and `degree_null.py` (is a gene pair closer than degree alone explains? `python src/degree_null.py data/toy_edges.txt ATM BRCA1`)
- `data/` includes `toy_edges.txt`; put the downloaded pairs and PPI network here
- `results/` outputs of each experiment (keep original-paper numbers separate from yours)

## Start
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Clone the official repo next to this folder and read how its features are built.
4. Make `src/features.py` match the repo's feature extraction, then run on a small subset first.

The helpers in `src/` are generic starting points, not the paper's code.

## Contact
harshinisuresha7@gmail.com

## About the figures
Figures 1-39 are computed and 40-59 simulated on a made-up 14-gene toy network, 60-79 are concept drawings, and 80-100 are empty placeholders for real results. None show MAGICAL results.
