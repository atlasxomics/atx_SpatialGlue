import logging
import re

import numpy as np
import pandas as pd
import warnings

from anndata import AnnData
from scipy import sparse
from scipy.spatial.distance import cdist
from typing import Optional

from latch.functions.messages import message

_BARCODE_REGEX = re.compile(r"([ATCG]{16})")
_GENE_SYMBOL_REGEX = re.compile(r"^(?!ENS[A-Z]*\d+)[A-Za-z][A-Za-z0-9_.-]{0,30}$")


def clean_ids(ix):
    s = pd.Index(ix).astype(str).str.strip()
    s = s.str.replace(r".*#", "", regex=True)  # drop sample prefix like 'D01887#'
    s = s.str.replace(r"-\d+$", "", regex=True)  # drop trailing '-1'
    return s


def validate_obs_barcodes(obs_names, context: str, min_fraction: float = 1.0):
    """Check obs_names contain a 16bp A/T/C/G barcode.

    The barcode can be embedded in a larger string (e.g., "S1#<barcode>-1").
    Raises a RuntimeError if fewer than `min_fraction` of names contain a
    matching barcode. Returns a pandas Index of the extracted barcodes.
    """

    names = pd.Index(obs_names).astype(str)
    barcodes = names.str.upper().str.extract(_BARCODE_REGEX, expand=False)
    valid = barcodes.notna()
    n = len(names)
    frac_valid = valid.sum() / n if n else 0.0

    if frac_valid < min_fraction:
        missing_examples = names[~valid][:5].tolist()
        raise RuntimeError(
            f"{context}: only {frac_valid * 100:.1f}% of obs_names contain a 16bp barcode; "
            f"examples missing barcode: {missing_examples}"
        )

    logging.info(
        f"{context}: {valid.sum()}/{n} obs_names contain 16bp barcodes; "
        f"examples: {barcodes[:3].tolist()}"
    )
    return barcodes


def ensure_obs_barcodes(
        adata: AnnData, context: str, min_fraction: float = 1.0
):
    """Validate obs_names, or fall back to obs['barcode'/'barcodes'] if
    present.

    If obs_names fail validation, we search for obs columns named 'barcode' or
    'barcodes'. If one validates, we overwrite obs_names with that column.
    Raises a RuntimeError if neither are valid.
    """

    try:
        return validate_obs_barcodes(
            adata.obs_names, f"{context} obs_names", min_fraction
        )
    except RuntimeError as err:
        for col in ["barcode", "barcodes"]:
            if col in adata.obs.columns:
                barcodes = validate_obs_barcodes(
                    adata.obs[col], f"{context} obs['{col}']", min_fraction
                )
                adata.obs_names = pd.Index(barcodes)
                logging.warning(
                    f"{context}: obs_names invalid; replaced with obs['{col}'] barcodes"
                )
                return adata.obs_names
        message(
            typ="error",
            data={
                "title": "Missing barcodes",
                "body": """Cannot find cell barcodes in obs.  Please use an
                    AnnData object with obs set to cells with cell barcodes
                    contained in obs_names.  If using outputs from the RNAQC
                    Workflow, please select the .h5ad file in the optimize_outs
                    directory, NOT the top directory."""
            }
        )
        raise RuntimeError(
            f"{context}: obs_names invalid and no valid 'barcode'/'barcodes' column found"
        ) from err


def validate_var_gene_symbols(
        var_names, context: str, min_fraction: float = 0.8
):
    """Check var_names look like gene symbols (not Ensembl IDs).

    Heuristic: must start with a letter, allow letters/digits/._-, and cannot
    start with an Ensembl-like prefix (e.g., ENSG, ENSMUSG).
    """

    names = pd.Index(var_names).astype(str)
    valid = names.str.match(_GENE_SYMBOL_REGEX)
    n = len(names)
    frac_valid = valid.sum() / n if n else 0.0

    if frac_valid < min_fraction:
        bad_examples = names[~valid][:5].tolist()
        raise RuntimeError(
            f"{context}: only {frac_valid * 100:.1f}% of var_names look like gene symbols; "
            f"examples failing heuristic: {bad_examples}"
        )

    logging.info(
        f"{context}: {valid.sum()}/{n} var_names look like gene symbols; "
        f"examples: {names[:3].tolist()}"
    )
    return names


def ensure_var_gene_symbols(
        adata: AnnData, context: str, min_fraction: float = 0.8
):
    """Validate var_names or fall back to gene symbol columns if available."""

    try:
        return validate_var_gene_symbols(
            adata.var_names, f"{context} var_names", min_fraction
        )
    except RuntimeError as err:
        for col in ["geneName", "gene_name", "gene_symbols"]:
            if col in adata.var.columns:
                names = validate_var_gene_symbols(
                    adata.var[col], f"{context} var['{col}']", min_fraction
                )
                adata.var_names = pd.Index(names)
                logging.warning(
                    f"{context}: var_names invalid; replaced with var['{col}'] gene symbols"
                )
                return adata.var_names
        message(
            typ="error",
            data={
                "title": "Missing gene names",
                "body": """Cannot find gene names in var.  Please use an
                    AnnData object with var set to genes with gene symbols
                    contained in var_names.  If using outputs from the RNAQC
                    Workflow, please select the .h5ad file in the optimize_outs
                    directory, NOT the top directory."""
            }
        )
        raise RuntimeError(
            f"{context}: var_names invalid and no valid gene symbol column found"
        ) from err


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
