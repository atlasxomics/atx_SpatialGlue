from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import sys

from numpy import ndarray
from scipy import sparse
from typing import TYPE_CHECKING, List, NamedTuple, Tuple

if TYPE_CHECKING:
    from anndata import AnnData


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


class AlignmentIndexers(NamedTuple):
    obs_names: pd.Index
    genes: pd.Index
    rna_obs: np.ndarray
    ge_obs: np.ndarray
    rna_var: np.ndarray
    ge_var: np.ndarray


class IndexedMatrix:
    """A row/column-indexed matrix that materializes only requested chunks."""

    def __init__(self, matrix, row_indices, col_indices):
        self.matrix = matrix
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.col_indices = np.asarray(col_indices, dtype=np.int64)
        self.shape = (len(self.row_indices), len(self.col_indices))
        self.dtype = getattr(matrix, "dtype", None)

    def select_columns(self, selector):
        return IndexedMatrix(
            self.matrix,
            self.row_indices,
            self.col_indices[selector],
        )

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("IndexedMatrix requires matrix[row, column] indexing.")
        row_selector, col_selector = key
        rows = np.atleast_1d(self.row_indices[row_selector])
        cols = np.atleast_1d(self.col_indices[col_selector])
        if sparse.issparse(self.matrix):
            # Select the small gene chunk first. Row-first CSR indexing would
            # temporarily copy nearly the entire source matrix for every chunk.
            return self.matrix[:, cols][rows, :]
        return np.asarray(self.matrix)[np.ix_(rows, cols)]


def optimize_indexed_matrix_columns(indexed: IndexedMatrix, cache=None) -> IndexedMatrix:
    """Convert a sparse source to CSC once for repeated gene-column access."""
    source = indexed.matrix
    if not sparse.issparse(source) or sparse.isspmatrix_csc(source):
        return indexed

    if cache is None:
        cache = {}
    cache_key = id(source)
    optimized = cache.get(cache_key)
    if optimized is None:
        logging.info(
            "Converting %s correlation source %s to CSC for column access...",
            source.shape,
            type(source).__name__,
        )
        optimized = source.tocsc()
        optimized.sum_duplicates()
        optimized.eliminate_zeros()
        cache[cache_key] = optimized

    return IndexedMatrix(
        optimized,
        indexed.row_indices,
        indexed.col_indices,
    )


def alignment_indexers(adata1: AnnData, adata2: AnnData) -> AlignmentIndexers:
    """Find common cells/genes without creating full AnnData views or copies."""
    keep_obs = adata1.obs_names.isin(adata2.obs_names)
    common_obs = adata1.obs_names[keep_obs]
    if len(common_obs) == 0:
        raise RuntimeError(
            "Could not find common cells between transcriptome and gene "
            "accessibility data; please ensure the inputs are from the same experiment."
        )

    rna_obs = adata1.obs_names.get_indexer(common_obs)
    ge_obs = adata2.obs_names.get_indexer(common_obs)

    rna_upper = pd.Index(adata1.var_names.astype(str)).str.upper()
    ge_upper = pd.Index(adata2.var_names.astype(str)).str.upper()
    logging.info(f"Object 1 feats preview: {rna_upper[:5]}")
    logging.info(f"Object 2 feats preview: {ge_upper[:5]}")
    keep_gene = rna_upper.isin(ge_upper) & ~rna_upper.duplicated()
    common_upper = rna_upper[keep_gene]
    if len(common_upper) == 0:
        raise RuntimeError("No features overlap. Check gene naming (symbols vs IDs).")
    if len(common_upper) < 500:
        logging.warning(
            f"Only {len(common_upper)} features overlap. Check gene naming (symbols vs IDs)."
        )

    logging.info("Building correlation row/column index maps...")
    rna_var = np.flatnonzero(keep_gene)
    ge_first = ~ge_upper.duplicated()
    ge_lookup = pd.Series(
        np.flatnonzero(ge_first),
        index=ge_upper[ge_first],
    )
    ge_var = ge_lookup.reindex(common_upper).to_numpy(dtype=np.int64)
    if (rna_obs < 0).any() or (ge_obs < 0).any() or (ge_var < 0).any():
        raise RuntimeError("Internal error while indexing correlation modalities.")

    genes = pd.Index(adata1.var_names[rna_var].astype(str))
    return AlignmentIndexers(
        common_obs,
        genes,
        rna_obs,
        ge_obs,
        rna_var,
        ge_var,
    )


def aligned_metadata(adata: AnnData, obs_indices, var_indices):
    """Build a matrix-free AnnData containing metadata needed by reports."""
    from anndata import AnnData

    out = AnnData(
        obs=adata.obs.iloc[obs_indices].copy(),
        var=adata.var.iloc[var_indices].copy(),
    )
    if "spatial" in adata.obsm:
        out.obsm["spatial"] = np.asarray(adata.obsm["spatial"])[obs_indices].copy()
    return out


def indexed_rna_counts(adata: AnnData, alignment: AlignmentIndexers):
    """Select the RNA count source while retaining only an indexed reference."""
    if "counts" in adata.layers:
        return (
            IndexedMatrix(adata.layers["counts"], alignment.rna_obs, alignment.rna_var),
            "counts",
        )
    raw = getattr(adata, "raw", None)
    if raw is not None and raw.X is not None and raw.X.shape[0] == adata.n_obs:
        raw_names = pd.Index(raw.var_names).astype(str)
        raw_first = ~raw_names.duplicated()
        raw_lookup = pd.Series(
            np.flatnonzero(raw_first),
            index=raw_names[raw_first],
        )
        raw_idx = raw_lookup.reindex(alignment.genes).fillna(-1).to_numpy(dtype=np.int64)
        if (raw_idx >= 0).all():
            return IndexedMatrix(raw.X, alignment.rna_obs, raw_idx), "raw"
        logging.warning(
            "RNA .raw is missing synchronized genes; using current X for count stats."
        )
    logging.warning(
        "RNA counts/.raw missing; using current X for count stats (may be normalized)."
    )
    return IndexedMatrix(adata.X, alignment.rna_obs, alignment.rna_var), "X"


def get_corr_df(
    array1: ndarray,
    array2: ndarray,
    genes: List[str],
    array1_name: str = "RNA",
    array2_name: str = "GA",
    chunk_size: int = 64,
    n_jobs: int = 1,
) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor
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
    if n_jobs <= 0:
        raise ValueError("n_jobs must be > 0.")
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
    means1 = np.empty(n_genes, dtype=np.float64)
    means2 = np.empty(n_genes, dtype=np.float64)
    dof = n_obs - 2

    executor = ThreadPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
    try:
        for start in range(0, n_genes, chunk_size):
            end = min(start + chunk_size, n_genes)
            x = _dense_float32(array1[:, start:end])
            y = _dense_float32(array2[:, start:end])
            means1[start:end] = x.mean(axis=0, dtype=np.float64)
            means2[start:end] = y.mean(axis=0, dtype=np.float64)

            columns = [x[:, i] for i in range(x.shape[1])]
            columns.extend(y[:, i] for i in range(y.shape[1]))
            if executor is None:
                ranked = [rankdata(column) for column in columns]
            else:
                ranked = list(executor.map(rankdata, columns))
            split = x.shape[1]
            x_rank = np.column_stack(ranked[:split]).astype(np.float32)
            y_rank = np.column_stack(ranked[split:]).astype(np.float32)

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
                clipped = np.clip(
                    rho.astype(np.float64), -1 + 1e-15, 1 - 1e-15
                )
                t_stat = clipped * np.sqrt(
                    dof / ((1.0 - clipped) * (1.0 + clipped))
                )
                p = 2.0 * t.sf(np.abs(t_stat), dof)
                pvals[start:end] = np.where(denominator > 0, p, 1.0)
    finally:
        if executor is not None:
            executor.shutdown()

    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")

    res = pd.DataFrame({
        "gene": gene_names.values,
        "spearman_rho": rhos,
        "pval": pvals,
        "qval_bh": qvals,
        f"mean_{array1_name}": means1,
        f"mean_{array2_name}": means2,
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
