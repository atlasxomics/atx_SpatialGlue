from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import sys

from numpy import ndarray
from scipy import sparse
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from anndata import AnnData


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def get_corr_df(
    array1: ndarray,
    array2: ndarray,
    genes: List[str],
    array1_name: str = "RNA",
    array2_name: str = "GA",
    chunk_size: int = 250,
) -> pd.DataFrame:
    from scipy.stats import rankdata, t
    from statsmodels.stats.multitest import multipletests

    """Compute column-wise Spearman correlations with p-values/FDR.

    The implementation ranks and correlates genes in chunks. This keeps the
    output schema from the original scipy loop while avoiding one spearmanr
    call per gene.
    """

    gene_names = pd.Index(genes).astype(str)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")
    if array1.shape != array2.shape:
        raise ValueError(
            f"array1 and array2 must have the same shape; got "
            f"{array1.shape} and {array2.shape}."
        )

    n_obs, n_genes = array1.shape
    if n_genes != len(gene_names):
        raise ValueError(
            f"Number of genes ({len(gene_names)}) does not match matrix columns "
            f"({n_genes})."
        )

    rhos = np.empty(n_genes, dtype=np.float32)
    pvals = np.ones(n_genes, dtype=np.float64)
    dof = n_obs - 2

    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        x = _dense_float32(array1[:, start:end])
        y = _dense_float32(array2[:, start:end])

        x_rank = np.apply_along_axis(rankdata, 0, x).astype(np.float32)
        y_rank = np.apply_along_axis(rankdata, 0, y).astype(np.float32)

        x_centered = x_rank - x_rank.mean(axis=0)
        y_centered = y_rank - y_rank.mean(axis=0)
        numerator = (x_centered * y_centered).sum(axis=0)
        denominator = np.sqrt(
            (x_centered ** 2).sum(axis=0) * (y_centered ** 2).sum(axis=0)
        )

        rho = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=denominator > 0,
        )
        rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
        rhos[start:end] = rho.astype(np.float32)

        if dof > 0:
            clipped = np.clip(rho.astype(np.float64), -1 + 1e-15, 1 - 1e-15)
            t_stat = clipped * np.sqrt(dof / ((1.0 - clipped) * (1.0 + clipped)))
            p = 2.0 * t.sf(np.abs(t_stat), dof)
            pvals[start:end] = np.where(denominator > 0, p, 1.0)

    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")

    res = pd.DataFrame({
        "gene": gene_names.values,
        "spearman_rho": rhos,
        "pval": pvals,
        "qval_bh": qvals,
        f"mean_{array1_name}": _colmean(array1),
        f"mean_{array2_name}": _colmean(array2),
    })

    res["abs_rho"] = res["spearman_rho"].abs()
    res = res.sort_values("abs_rho", ascending=False)

    return res


def _dense_float32(X) -> ndarray:
    if sparse.issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def _colmean(X) -> ndarray:
    return np.asarray(X.mean(axis=0)).ravel()


def log_norm(array: ndarray, scaleto: int) -> ndarray:
    lib = array.sum(axis=1, keepdims=True)
    lib[lib < 1] = 1
    array_norm = np.log1p((array / lib) * scaleto).astype(np.float32)

    return array_norm


def synch_adata(adata1: AnnData, adata2: AnnData) -> Tuple[AnnData, AnnData]:

    # Align cells
    common = adata1.obs_names.intersection(adata2.obs_names)

    if len(common) == 0:
        raise RuntimeError(
            "Could not find common cells between transcriptome and gene accessibility data; please ensure the input files are from the same experiment."
        )

    adata1 = adata1[common, :]
    adata2 = adata2[common, :]
    adata2 = adata2[adata2.obs_names.get_indexer(adata1.obs_names), :]
    assert (adata1.obs_names == adata2.obs_names).all()

    # Align genes
    adata1_feats_up = pd.Index(adata1.var_names.astype(str)).str.upper()
    adata2_feats_up = pd.Index(adata2.var_names.astype(str)).str.upper()
    logging.info(f"Object 1 feats preview: {adata1_feats_up[:5]}")
    logging.info(f"Object 2 feats preview: {adata2_feats_up[:5]}")
    feats_common = adata1_feats_up.intersection(adata2_feats_up)
    if len(feats_common) == 0:
        raise RuntimeError("No features overlap. Check gene naming (symbols vs IDs).")
    if len(feats_common) < 500:
        logging.warning(
            f"Only {len(feats_common)} features overlap. Check gene naming (symbols vs IDs)."
        )

    # indexers
    logging.info("Reindexing...")
    adata1_idx = adata1_feats_up.get_indexer(feats_common)
    adata2_idx = adata2_feats_up.get_indexer(feats_common)

    logging.info("Filtering AnnData by common genes...")
    adata1_sub = adata1[:, adata1_idx].copy()
    adata2_sub = adata2[:,  adata2_idx].copy()

    return adata1_sub, adata2_sub
