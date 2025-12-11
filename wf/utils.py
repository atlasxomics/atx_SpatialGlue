import numpy as np
import pandas as pd
import warnings

from anndata import AnnData
from scipy import sparse
from scipy.spatial.distance import cdist
from typing import Optional


def clean_ids(ix):
    s = pd.Index(ix).astype(str).str.strip()
    s = s.str.replace(r".*#", "", regex=True)  # drop sample prefix like 'D01887#'
    s = s.str.replace(r"-\d+$", "", regex=True)  # drop trailing '-1'
    return s


def merge_small_clusters(
    adata: AnnData,
    cluster_key: str = "sg_leiden",
    embed_key: str = "SpatialGlue",
    min_cells: int = 20,
    new_key: Optional[str] = None,
    verbose: bool = True,
):
    """
    Merge clusters smaller than `min_cells` into the nearest larger cluster
    (nearest by centroid distance in `embed_key` space).

    Writes merged labels to `new_key` (default: f"{cluster_key}_merged").
    Keeps the original labels unchanged.
    """
    if new_key is None:
        new_key = f"{cluster_key}_merged"

    if cluster_key not in adata.obs:
        raise KeyError(f"{cluster_key!r} not found in adata.obs")
    if embed_key not in adata.obsm:
        raise KeyError(f"{embed_key!r} not found in adata.obsm")

    labels = adata.obs[cluster_key].astype(str).copy()
    emb = np.asarray(adata.obsm[embed_key])
    if emb.ndim != 2 or emb.shape[0] != adata.n_obs:
        raise ValueError(f"{embed_key!r} must be 2D with shape (n_obs, n_dim)")

    # sizes
    counts = labels.value_counts().sort_index()
    small = counts[counts < min_cells].index.tolist()
    if len(counts) <= 1:
        if verbose:
            print("Only one cluster found; nothing to merge.")
        adata.obs[new_key] = labels.astype("category")
        return adata

    # centroids
    centroids = {}
    for cl in counts.index:
        mask = (labels == cl).values
        if not np.any(mask):
            continue  # safety (shouldn't happen)
        centroids[cl] = emb[mask].mean(axis=0)

    # merge targets
    large = [
        cl for cl in counts.index
        if (counts[cl] >= min_cells and cl in centroids)
    ]
    if len(large) == 0:
        fallback = counts.sort_values(ascending=False).index[0]
        large = [fallback]

    # build merge map
    merge_map = {}
    for s in small:
        if s not in centroids:
            continue
        candidates = [c for c in large if c != s]
        if len(candidates) == 0:
            candidates = [c for c in counts.index if c != s and c in centroids]
        if len(candidates) == 0:
            continue

        s_cent = centroids[s].reshape(1, -1)
        c_mat = np.vstack([centroids[c] for c in candidates])
        d = cdist(s_cent, c_mat, metric="euclidean").ravel()
        tgt = candidates[int(np.argmin(d))]
        merge_map[s] = tgt

    if len(merge_map) == 0:
        if verbose:
            print(f"All clusters have ≥{min_cells} cells; no merge performed.")
        adata.obs[new_key] = labels.astype("category")
        return adata

    # warn + apply
    msg_lines = [f"Merging clusters with <{min_cells} cells (nearest centroid in '{embed_key}'):"]
    for s, t in merge_map.items():
        msg_lines.append(f"  - {s} (n={counts[s]}) → {t} (n={counts[t]})")
    msg = "\n".join(msg_lines)
    warnings.warn(msg)
    if verbose:
        print(msg)

    new_labels = labels.copy()
    for s, t in merge_map.items():
        new_labels.loc[new_labels == s] = t

    new_labels = pd.Categorical(
        new_labels, categories=sorted(new_labels.unique())
    )
    adata.obs[new_key] = new_labels

    if verbose:
        post_counts = adata.obs[new_key].value_counts().sort_index()
        print("Post-merge cluster sizes:")
        print(post_counts.to_string())

    return adata


def to_dense(X):
    return X.toarray() if sparse.issparse(X) else X
