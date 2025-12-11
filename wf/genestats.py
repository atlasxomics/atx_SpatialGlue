import numpy as np
import pandas as pd

from scipy import sparse
from typing import Dict, Optional


def _colsum(X):
    return np.asarray(X.sum(axis=0)).ravel()


def _colnnz(X):
    if sparse.issparse(X):
        return np.diff(X.tocsc().indptr)
    else:
        return (X > 0).sum(axis=0)


def _colmean(X):
    n = X.shape[0]
    return _colsum(X) / max(n, 1)


def _colvar(X):
    # population variance across cells (includes zeros)
    n = X.shape[0]
    if sparse.issparse(X):
        X2_sum = np.asarray(X.power(2).sum(axis=0)).ravel()
    else:
        X2_sum = (X**2).sum(axis=0)
    mu = _colmean(X)
    return (X2_sum / max(n, 1)) - mu**2


def _minmax_nonzero(X):
    """Return per-gene min_nonzero and max (among nonzero entries).
       For genes with all zeros: min_nonzero=0, max=0."""
    if sparse.issparse(X):
        C = X.tocsc()
        mins = np.zeros(C.shape[1], dtype=float)
        maxs = np.zeros(C.shape[1], dtype=float)
        for j in range(C.shape[1]):
            start, end = C.indptr[j], C.indptr[j+1]
            col = C.data[start:end]
            if col.size > 0:
                mins[j] = col.min()
                maxs[j] = col.max()
        return mins, maxs
    else:
        M = np.ma.masked_less_equal(X, 0.0)
        mins = np.array(M.min(axis=0)).ravel()
        mins = np.nan_to_num(mins, nan=0.0)
        M2 = np.ma.masked_equal(X, 0.0)
        maxs = np.array(M2.max(axis=0)).ravel()
        maxs = np.nan_to_num(maxs, nan=0.0)
        return mins, maxs


def get_rna_counts_matrix(adata_view):
    """Prefer true UMI counts if present; otherwise fall back (with warning).
    """
    if "counts" in adata_view.layers:
        return adata_view.layers["counts"], "counts"
    if getattr(adata_view, "raw", None) is not None and adata_view.raw.X is not None:
        return adata_view.raw.X, "raw"
    print(
        "[warn] RNA 'counts' or .raw missing; using current X (may be log-normalized)."
    )
    return adata_view.X, "X"


def compute_gene_stats_matrix(
    X, gene_names, prefix: str, include_minmax_nonzero: bool = True
) -> pd.DataFrame:
    """Compute gene-level stats from a (cells x genes) matrix X."""
    n_cells = X.shape[0]
    s = _colsum(X)
    mu = _colmean(X)
    var = _colvar(X)
    nnz = _colnnz(X)
    det_rate = nnz / max(n_cells, 1)
    nz_mean = np.divide(
        s, nnz, out=np.zeros_like(s, dtype=float), where=nnz > 0
    )
    cv = np.sqrt(np.clip(var, 0, None)) / np.maximum(mu, 1e-12)

    d = {
        "gene": gene_names.astype(str).values,
        f"{prefix}_sum": s,
        f"{prefix}_mean": mu,
        f"{prefix}_var": var,
        f"{prefix}_cv": cv,
        f"{prefix}_nnz": nnz,
        f"{prefix}_detect_rate": det_rate,
        f"{prefix}_nz_mean": nz_mean,
    }

    if include_minmax_nonzero:
        mn, mx = _minmax_nonzero(X)
        d[f"{prefix}_min_nz"] = mn
        d[f"{prefix}_max_nz"] = mx

    return pd.DataFrame(d)


def make_gene_filter(
    stats_df: pd.DataFrame,
    min_rna_total_umi: Optional[float] = 200.0,
    min_rna_detect_rate: Optional[float] = 0.05,  # >=5% of spots with UMI>0
    min_ge_total_signal: Optional[float] = 500.0,
    min_ge_detect_rate: Optional[float] = 0.05,  # >=5% nonzero cells
) -> pd.Series:
    """Return boolean mask over genes meeting both RNA and ATAC-GE thresholds.
    """
    m = pd.Series(True, index=stats_df.index)
    if min_rna_total_umi is not None:
        m &= stats_df["rna_umi_sum"] >= float(min_rna_total_umi)
    if min_rna_detect_rate is not None:
        m &= stats_df["rna_umi_detect_rate"] >= float(min_rna_detect_rate)
    if min_ge_total_signal is not None:
        m &= stats_df["ge_raw_sum"] >= float(min_ge_total_signal)
    if min_ge_detect_rate is not None:
        m &= stats_df["ge_raw_detect_rate"] >= float(min_ge_detect_rate)
    return m


def correlation_yield(
    stats_df: pd.DataFrame,
    mask: pd.Series,
    rho_cutoff: float = 0.30,
    use_abs: bool = True,
    qval_cutoff: Optional[float] = None
) -> Dict[str, float]:
    """Compute % of (filtered) genes with RNA⇄ATAC correlation above cutoff."""
    sub = stats_df.loc[mask].copy()
    if use_abs:
        meets_rho = sub["spearman_rho"].abs() >= rho_cutoff
    else:
        meets_rho = sub["spearman_rho"] >= rho_cutoff
    if qval_cutoff is not None:
        meets_rho &= (sub["qval_bh"] <= qval_cutoff)

    n_all = len(stats_df)
    n_kept = int(mask.sum())
    n_hit = int(meets_rho.sum())
    pct_hit_of_kept = 100.0 * n_hit / max(n_kept, 1)
    pct_hit_of_all = 100.0 * n_hit / max(n_all, 1)

    # optional sign breakdown
    pos = int((sub["spearman_rho"] >= rho_cutoff).sum()) if not use_abs else int((sub["spearman_rho"] >= rho_cutoff).sum())
    neg = int((sub["spearman_rho"] <= -rho_cutoff).sum()) if not use_abs else int((sub["spearman_rho"] <= -rho_cutoff).sum())

    return dict(
        n_all=n_all, n_kept=n_kept, n_hit=n_hit,
        pct_hit_of_kept=pct_hit_of_kept, pct_hit_of_all=pct_hit_of_all,
        n_pos_at_least_cutoff=pos, n_neg_at_most_minus_cutoff=neg
    )
