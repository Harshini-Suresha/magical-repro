"""Train/test splits. pairs: DataFrame with columns gene_a, gene_b, label."""
import numpy as np
from sklearn.model_selection import train_test_split

def random_pair_split(pairs, test_size=0.2, seed=0):
    return train_test_split(pairs, test_size=test_size, random_state=seed, stratify=pairs["label"])

def gene_held_out_split(pairs, test_frac=0.2, seed=0):
    """Test pairs contain only genes never seen in training pairs."""
    rng = np.random.default_rng(seed)
    genes = np.unique(pairs[["gene_a", "gene_b"]].values)
    held = set(rng.choice(genes, size=int(len(genes) * test_frac), replace=False))
    in_a, in_b = pairs["gene_a"].isin(held), pairs["gene_b"].isin(held)
    test = pairs[in_a & in_b]
    train = pairs[~in_a & ~in_b]
    return train, test  # pairs touching one held-out gene are dropped to avoid leakage
