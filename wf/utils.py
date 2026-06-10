from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import warnings

from scipy import sparse

from latch.functions.messages import message

if TYPE_CHECKING:
    from anndata import AnnData

_BARCODE_REGEX = re.compile(r"([ATCG]{16})")
_DONOR_PREFIX_REGEX = re.compile(r"(D\d{5})")
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


def ensure_obs_run_barcodes(
        adata: AnnData, context: str, min_fraction: float = 1.0
):
    """Return barcode IDs, preserving a Dxxxxx run prefix when available.

    Examples:
      D02735_NG09414#ACGT... -> D02735#ACGT...
      D02735_NG09420#ACGT... -> D02735#ACGT...
      ACGT...-1              -> ACGT...
    """

    original_names = pd.Index(adata.obs_names).astype(str)
    barcodes = ensure_obs_barcodes(adata, context, min_fraction=min_fraction)
    barcodes = pd.Index(barcodes).astype(str)

    donor = original_names.str.extract(_DONOR_PREFIX_REGEX, expand=False)
    for col in ["sample_name", "sample", "sample_id", "library_id", "batch"]:
        if col not in adata.obs.columns:
            continue
        from_col = pd.Index(adata.obs[col]).astype(str).str.extract(
            _DONOR_PREFIX_REGEX, expand=False
        )
        donor = pd.Index([
            fallback if pd.isna(current) else current
            for current, fallback in zip(donor, from_col)
        ])

    if donor.notna().any():
        ids = pd.Index([
            f"{d}#{bc}" if pd.notna(d) else bc
            for d, bc in zip(donor, barcodes)
        ])
    else:
        ids = barcodes

    duplicated = ids[ids.duplicated()].unique()
    if len(duplicated) > 0:
        raise RuntimeError(
            f"{context}: normalized obs IDs are not unique; examples: "
            f"{duplicated[:5].tolist()}"
        )

    logging.info(
        f"{context}: normalized obs IDs examples: {ids[:3].tolist()}"
    )
    return ids


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
                try:
                    names = validate_var_gene_symbols(
                        adata.var[col], f"{context} var['{col}']", min_fraction
                    )
                    adata.var_names = pd.Index(names)
                    message(
                        typ="warning",
                        data={
                            "title": "Reassigned var_names",
                            "body": f"""var_names were not found to contain a
                            majority of gene symbols.  Reassigned var_names to
                            {col}."""
                        }
                    )
                    logging.warning(
                        f"{context}: var_names invalid; replaced with var['{col}'] gene symbols"
                    )
                    return adata.var_names
                except RuntimeError:
                    message(
                        typ="warning",
                        data={
                            "title": "Attempted reassign",
                            "body": f"""var_names were not found to contain a
                            majority of gene symbols. Attempted to reassign
                            barcodes with to {col}, but this did not have a
                            sufficient proportion of gene symbols."""
                        }
                    )
                    logging.warning(
                        f"{context}:var_names invalid; attempted to replace with var['{col}']."
                    )
        raise RuntimeError(
            f"{context}: obs_names invalid and no valid 'barcode'/'barcodes' column found"
        ) from err


def merge_small_clusters(
    adata: AnnData,
    cluster_key: str = "sg_leiden",
    embed_key: str = "SpatialGlue",
    min_cells: int = 20,
    new_key: Optional[str] = None,
    verbose: bool = True,
):
    from scipy.spatial.distance import cdist

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


def to_numpy_array(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


DEFAULT_RESOLUTIONS = "0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.2"
N_COMPONENTS = 50
SEED = 42
SCANPY_CLUSTER_POINT_SIZE = 0.5
SPATIAL_SCATTER_POINT_SIZE = 2.5
PLOTTING_EMBEDDING_KEYS = (
    "SpatialGlue",
    "X_pca",
    "X_stagate",
    "adj_feature",
    "alpha",
    "alpha_omics1",
    "alpha_omics2",
    "feat",
)


def figures_dir(out_dir: str) -> str:
    path = os.path.join(out_dir, "figures")
    os.makedirs(path, exist_ok=True)
    return path


def tables_dir(out_dir: str) -> str:
    path = os.path.join(out_dir, "tables")
    os.makedirs(path, exist_ok=True)
    return path


def fig_path(out_dir: str, filename: str) -> str:
    stem, ext = os.path.splitext(filename)
    if ext.lower() == ".pdf":
        filename = f"{stem}.png"
    path = os.path.join(figures_dir(out_dir), filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def table_path(out_dir: str, filename: str) -> str:
    return os.path.join(tables_dir(out_dir), filename)


def safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return safe or "labels"


def strip_plotting_embeddings(adata) -> None:
    """Remove large intermediate embeddings that are not needed for plotting."""
    for key in PLOTTING_EMBEDDING_KEYS:
        if key in adata.obsm:
            del adata.obsm[key]
        if key in adata.obsp:
            del adata.obsp[key]
        if key in adata.uns:
            del adata.uns[key]


def make_plotting_anndata(adata, matrix_dtype=np.float32, force_dense: bool = False):
    """Return a reduced AnnData for notebook/plotting use."""
    out = adata.copy()

    obs_cols = [
        "barcode",
        "n_genes_by_counts",
        "log1p_n_genes_by_counts",
        "total_counts",
        "log1p_total_counts",
        "pct_counts_in_top_50_genes",
        "pct_counts_in_top_100_genes",
        "pct_counts_in_top_200_genes",
        "pct_counts_in_top_500_genes",
        "total_counts_mt",
        "log1p_total_counts_mt",
        "pct_counts_mt",
    ]
    var_cols = [
        "mt",
        "n_counts",
        "n_cells",
        "n_cells_by_counts",
        "mean_counts",
        "log1p_mean_counts",
        "pct_dropout_by_counts",
        "total_counts",
        "log1p_total_counts",
        "means",
        "dispersions",
        "dispersions_norm",
    ]

    out.obs.drop([c for c in obs_cols if c in out.obs.columns], axis=1, inplace=True)
    out.var.drop([c for c in var_cols if c in out.var.columns], axis=1, inplace=True)
    out.varm.clear()
    out.layers.clear()
    out.raw = None

    for key in ["pca", "log1p"]:
        out.uns.pop(key, None)

    keep_obsm = {"spatial", "X_umap"}
    for key in list(out.obsm.keys()):
        if key not in keep_obsm:
            del out.obsm[key]
    for key in PLOTTING_EMBEDDING_KEYS:
        if key in out.obsp:
            del out.obsp[key]
        if key in out.uns:
            del out.uns[key]

    if sparse.issparse(out.X):
        out.X = out.X.toarray().astype(matrix_dtype) if force_dense else out.X.astype(matrix_dtype)
    else:
        out.X = np.asarray(out.X, dtype=matrix_dtype)

    return out


def save_fig_page(fig, out_dir: str, filename: str, page_idx: int) -> None:
    base = fig_path(out_dir, filename)
    stem, ext = os.path.splitext(base)
    ext = ext or ".png"
    fig.savefig(f"{stem}_{page_idx:03d}{ext}", dpi=200, bbox_inches="tight")


def save_fig_suffix(fig, out_dir: str, filename: str, suffix: str) -> None:
    base = fig_path(out_dir, filename)
    stem, ext = os.path.splitext(base)
    ext = ext or ".png"
    fig.savefig(f"{stem}_{safe_name(suffix)}{ext}", dpi=200, bbox_inches="tight")


def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_gene_list(genes: Optional[str]) -> list[str]:
    if genes is None:
        return []
    return [g.strip() for g in genes.split(",") if g.strip()]


def cluster_sort_key(label):
    label = str(label)
    return (0, int(label)) if label.isdigit() else (1, label)


def parse_resolutions(resolutions: str) -> list[float]:
    vals = []
    for item in resolutions.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise ValueError("At least one clustering resolution is required.")
    return vals


def resolution_suffix(resolution: float) -> str:
    return f"{resolution:g}".replace(".", "p")


def morans_i(connectivities, labels) -> float:
    """Compute Moran's I for numeric labels over a sparse connectivity graph."""
    W = connectivities.astype(np.float64)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    W = W.multiply(1.0 / row_sums[:, None])

    x = np.asarray(labels, dtype=np.float64)
    z = x - x.mean()
    denom = float(z @ z)
    if denom <= 1e-12:
        return 0.0
    return float((z @ W.dot(z)) / denom)


def spatial_connectivities(adata, n_neighbors: int):
    from sklearn.neighbors import kneighbors_graph

    if "spatial" not in adata.obsm:
        logging.warning(
            "No adata.obsm['spatial'] found; using embedding neighbors for Moran's I."
        )
        return adata.obsp["connectivities"]

    coords = np.asarray(adata.obsm["spatial"])
    rows = []
    cols = []
    data = []

    if "sample" in adata.obs.columns:
        groups = adata.obs.groupby("sample", sort=False).indices.values()
    else:
        groups = [np.arange(adata.n_obs)]

    for idx in groups:
        idx = np.asarray(idx, dtype=int)
        if len(idx) < 2:
            continue
        k = min(n_neighbors, len(idx) - 1)
        graph = kneighbors_graph(
            coords[idx],
            n_neighbors=k,
            mode="connectivity",
            include_self=False,
        ).tocoo()
        rows.append(idx[graph.row])
        cols.append(idx[graph.col])
        data.append(graph.data)

    if not rows:
        logging.warning(
            "Could not build spatial neighbor graph; using embedding neighbors "
            "for Moran's I."
        )
        return adata.obsp["connectivities"]

    conn = sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(adata.n_obs, adata.n_obs),
    )
    return conn.maximum(conn.T)


def choose_n_components(n_obs: int, n_vars: int, requested: int) -> int:
    n_components = min(requested, n_obs - 1, n_vars - 1)
    if n_components < 1:
        raise ValueError(
            f"Cannot compute embedding with n_obs={n_obs}, n_vars={n_vars}."
        )
    return n_components


def as_float32_csr(X):
    """Convert AnnData matrix-like inputs to numeric CSR without dtype ambiguity."""
    if sparse.issparse(X):
        out = X.tocsr().astype(np.float32)
    elif hasattr(X, "to_memory"):
        X_mem = X.to_memory()
        if sparse.issparse(X_mem):
            out = X_mem.tocsr().astype(np.float32)
        else:
            out = sparse.csr_matrix(np.asarray(X_mem, dtype=np.float32))
    else:
        out = sparse.csr_matrix(np.asarray(X, dtype=np.float32))

    if out.data.size:
        out.data = np.nan_to_num(out.data, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def compute_lsi(X, n_components: int = N_COMPONENTS, seed: int = SEED) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD

    """Compute TF-IDF + log1p + SVD LSI, dropping the first depth component."""
    X_raw = as_float32_csr(X)
    X_tfidf = X_raw.copy()

    row_sums = np.asarray(X_tfidf.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1
    X_tfidf = X_tfidf.multiply(1.0 / row_sums[:, None])

    col_nnz = np.diff(X_raw.tocsc().indptr)
    idf = np.log1p(X_raw.shape[0] / (col_nnz + 1))
    X_tfidf = X_tfidf.multiply(idf)
    X_tfidf = X_tfidf.multiply(1e4)
    X_tfidf.data = np.log1p(X_tfidf.data)

    n_svd = choose_n_components(
        X_tfidf.shape[0], X_tfidf.shape[1], n_components + 1
    )
    if n_svd < 2:
        raise ValueError("LSI requires at least two SVD components.")

    lsi = TruncatedSVD(n_components=n_svd, random_state=seed).fit_transform(X_tfidf)
    lsi = lsi[:, 1:]
    lsi = (lsi - lsi.mean(axis=0)) / (lsi.std(axis=0) + 1e-9)
    return lsi.astype(np.float32)


def add_rna_features(rna, n_components: int = N_COMPONENTS) -> None:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler

    if "feat" in rna.obsm:
        logging.info("RNA feat already present; reusing it.")
        return

    if "highly_variable" in rna.var.columns:
        hvgs = rna.var["highly_variable"].fillna(False).astype(bool).values
        if hvgs.sum() == 0:
            logging.warning("No highly_variable genes found; using all RNA genes.")
            hvgs = np.ones(rna.n_vars, dtype=bool)
    else:
        logging.warning("RNA highly_variable column not found; using all RNA genes.")
        hvgs = np.ones(rna.n_vars, dtype=bool)

    rna_hvg = rna[:, hvgs].copy()
    for layer in ["lognorm", "normalized", "log1p"]:
        if layer in rna_hvg.layers:
            logging.info(f"Using RNA layer '{layer}' for SpatialGlue features.")
            X = rna_hvg.layers[layer]
            break
    else:
        if "counts" in rna_hvg.layers:
            from wf import correlation as corr

            logging.info(
                "RNA log-normalized layer not found; computing lognorm from counts."
            )
            X = corr.log_norm(to_dense(rna_hvg.layers["counts"]), scaleto=10000)
        else:
            logging.warning(
                "RNA log-normalized/counts layers not found; using RNA .X."
            )
            X = rna_hvg.X

    X = to_dense(X).astype(np.float32)
    X_scaled = StandardScaler().fit_transform(X)
    n_svd = choose_n_components(X_scaled.shape[0], X_scaled.shape[1], n_components)
    rna.obsm["feat"] = TruncatedSVD(
        n_components=n_svd, random_state=SEED
    ).fit_transform(X_scaled).astype(np.float32)
    logging.info(f"RNA SpatialGlue features: {rna.obsm['feat'].shape}")


def align_modalities(rna, ge, atac=None):
    common = rna.obs_names.intersection(ge.obs_names)
    if atac is not None:
        common = common.intersection(atac.obs_names)
    if len(common) == 0:
        if atac is None:
            raise RuntimeError(
                "Could not find common barcodes across transcriptome and gene "
                "accessibility data."
            )
        raise RuntimeError(
            "Could not find common barcodes across transcriptome, gene "
            "accessibility, and ATAC tile data."
        )

    rna_matched = rna[common, :].copy()
    ge_matched = ge[common, :].copy()
    atac_matched = atac[common, :].copy() if atac is not None else None

    ge_matched = ge_matched[
        ge_matched.obs_names.get_indexer(rna_matched.obs_names), :
    ].copy()
    if atac_matched is not None:
        atac_matched = atac_matched[
            atac_matched.obs_names.get_indexer(rna_matched.obs_names), :
        ].copy()

    assert (rna_matched.obs_names == ge_matched.obs_names).all()
    if atac_matched is not None:
        assert (rna_matched.obs_names == atac_matched.obs_names).all()
    return rna_matched, ge_matched, atac_matched
