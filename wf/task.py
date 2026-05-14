import glob
import logging
import os
import pandas as pd
import pickle
import random
import subprocess
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import snapatac2 as snap
import torch

from matplotlib.backends.backend_pdf import PdfPages
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from SpatialGlue.preprocess import construct_neighbor_graph
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue

from latch.functions.messages import message
from latch.resources.tasks import custom_task
from latch.types import LatchDir, LatchFile

import wf.correlation as corr
import wf.genestats as gs
import wf.plotting as pl
import wf.utils as utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


DEFAULT_RESOLUTIONS = "0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.2"
N_COMPONENTS = 50
SEED = 42


def _parse_resolutions(resolutions: str) -> list[float]:
    vals = []
    for item in resolutions.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise ValueError("At least one clustering resolution is required.")
    return vals


def _parse_gene_list(genes: Optional[str]) -> list[str]:
    if genes is None:
        return []
    return [g.strip() for g in genes.split(",") if g.strip()]


def _resolution_suffix(resolution: float) -> str:
    return f"{resolution:g}".replace(".", "p")


def _choose_n_components(n_obs: int, n_vars: int, requested: int) -> int:
    n_components = min(requested, n_obs - 1, n_vars - 1)
    if n_components < 1:
        raise ValueError(
            f"Cannot compute embedding with n_obs={n_obs}, n_vars={n_vars}."
        )
    return n_components


def _morans_i(connectivities, labels) -> float:
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


def _spatial_connectivities(adata, n_neighbors: int):
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


def _compute_lsi(X, n_components: int = N_COMPONENTS, seed: int = SEED) -> np.ndarray:
    """Compute TF-IDF + log1p + SVD LSI, dropping the first depth component."""
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    X_raw = X.copy().astype(np.float32)
    X_tfidf = X_raw.copy()

    row_sums = np.asarray(X_tfidf.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1
    X_tfidf = X_tfidf.multiply(1.0 / row_sums[:, None])

    col_nnz = np.diff(X_raw.tocsc().indptr)
    idf = np.log1p(X_raw.shape[0] / (col_nnz + 1))
    X_tfidf = X_tfidf.multiply(idf)
    X_tfidf = X_tfidf.multiply(1e4)
    X_tfidf.data = np.log1p(X_tfidf.data)

    n_svd = _choose_n_components(
        X_tfidf.shape[0], X_tfidf.shape[1], n_components + 1
    )
    if n_svd < 2:
        raise ValueError("LSI requires at least two SVD components.")

    lsi = TruncatedSVD(n_components=n_svd, random_state=seed).fit_transform(X_tfidf)
    lsi = lsi[:, 1:]
    lsi = (lsi - lsi.mean(axis=0)) / (lsi.std(axis=0) + 1e-9)
    return lsi.astype(np.float32)


def _add_rna_features(rna, n_components: int = N_COMPONENTS) -> None:
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
            logging.info(
                "RNA log-normalized layer not found; computing lognorm from counts."
            )
            X = corr.log_norm(utils.to_dense(rna_hvg.layers["counts"]), scaleto=10000)
        else:
            logging.warning(
                "RNA log-normalized/counts layers not found; using RNA .X."
            )
            X = rna_hvg.X

    X = utils.to_dense(X).astype(np.float32)
    X_scaled = StandardScaler().fit_transform(X)
    n_svd = _choose_n_components(X_scaled.shape[0], X_scaled.shape[1], n_components)
    rna.obsm["feat"] = TruncatedSVD(
        n_components=n_svd, random_state=SEED
    ).fit_transform(X_scaled).astype(np.float32)
    logging.info(f"RNA SpatialGlue features: {rna.obsm['feat'].shape}")


def _align_modalities(rna, ge, atac):
    common = rna.obs_names.intersection(ge.obs_names).intersection(atac.obs_names)
    if len(common) == 0:
        raise RuntimeError(
            "Could not find common barcodes across transcriptome, gene "
            "accessibility, and ATAC tile data."
        )

    rna_matched = rna[common, :].copy()
    ge_matched = ge[common, :].copy()
    atac_matched = atac[common, :].copy()

    ge_matched = ge_matched[
        ge_matched.obs_names.get_indexer(rna_matched.obs_names), :
    ].copy()
    atac_matched = atac_matched[
        atac_matched.obs_names.get_indexer(rna_matched.obs_names), :
    ].copy()

    assert (rna_matched.obs_names == ge_matched.obs_names).all()
    assert (rna_matched.obs_names == atac_matched.obs_names).all()
    return rna_matched, ge_matched, atac_matched


def _cluster_sort_key(label):
    label = str(label)
    return (0, int(label)) if label.isdigit() else (1, label)


def _sample_labels(adata):
    for col in ["sample", "sample_name", "sample_id", "library_id", "batch"]:
        if col in adata.obs.columns:
            return adata.obs[col].astype(str).values
    names = pd.Index(adata.obs_names).astype(str)
    if names.str.contains("#").any():
        return names.str.split("#").str[0].values
    return np.repeat("all", adata.n_obs)


def _selected_report_genes(
    genes_of_interest: Optional[str],
    gene_names,
    corr_df: Optional[pd.DataFrame] = None,
    n_fallback: int = 10,
) -> list[str]:
    gene_set = set(pd.Index(gene_names).astype(str))
    requested = _parse_gene_list(genes_of_interest)
    if requested:
        selected = [g for g in requested if g in gene_set]
        missing = sorted(set(requested) - set(selected))
        if missing:
            logging.warning(f"Requested genes not found and skipped: {missing}")
        return selected
    if corr_df is not None and not corr_df.empty:
        return [
            g for g in corr_df.sort_values("abs_rho", ascending=False)["gene"].astype(str)
            if g in gene_set
        ][:n_fallback]
    return list(pd.Index(gene_names).astype(str)[:n_fallback])


def _scatter_spatial(ax, coords, values, title, cmap="Spectral_r", categorical=False):
    if categorical:
        labels = pd.Categorical(values.astype(str))
        codes = labels.codes
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=codes, s=6, cmap="tab20")
        handles = []
        for code, label in enumerate(labels.categories):
            handles.append(plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=scatter.cmap(scatter.norm(code)),
                markersize=5,
                label=str(label),
            ))
        ax.legend(handles=handles, title="cluster", fontsize=6, loc="best")
    else:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=6, cmap=cmap)
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])


def _write_spatial_expression_reports(
    out_dir: str,
    rna,
    ge,
    genes,
    X_rna,
    X_ge,
    report_genes: list[str],
) -> None:
    if "spatial" not in rna.obsm:
        logging.warning("No RNA spatial coordinates found; skipping spatial expression maps.")
        return
    if not report_genes:
        logging.warning("No report genes available; skipping spatial expression maps.")
        return

    gene_names = pd.Index(genes).astype(str)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    ge.obsm["spatial"] = rna.obsm["spatial"].copy()
    samples = _sample_labels(rna)
    ge.obs["sample_for_plot"] = samples
    rna.obs["sample_for_plot"] = samples

    selected = [g for g in report_genes if g in gene_to_idx][:12]
    if not selected:
        return

    for modality, matrix, adata, fname, cmap in [
        ("RNA", X_rna, rna, "rna_spatial_expression.pdf", "Spectral_r"),
        ("ATAC GE", X_ge, ge, "atac_ge_spatial_expression.pdf", "Spectral_r"),
    ]:
        with PdfPages(os.path.join(out_dir, fname)) as pdf:
            for sample in sorted(set(samples)):
                mask = samples == sample
                ncols = min(3, len(selected))
                nrows = int(np.ceil(len(selected) / ncols))
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(4 * ncols, 4 * nrows),
                    squeeze=False,
                )
                axes = axes.ravel()
                for i, gene in enumerate(selected):
                    idx = gene_to_idx[gene]
                    _scatter_spatial(
                        axes[i],
                        adata.obsm["spatial"][mask],
                        matrix[mask, idx],
                        f"{sample} {modality}: {gene}",
                        cmap=cmap,
                    )
                for ax in axes[len(selected):]:
                    ax.axis("off")
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    with PdfPages(os.path.join(out_dir, "rna_vs_atac_ge_spatial_expression.pdf")) as pdf:
        for sample in sorted(set(samples)):
            mask = samples == sample
            for gene in selected:
                idx = gene_to_idx[gene]
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                _scatter_spatial(
                    axes[0],
                    rna.obsm["spatial"][mask],
                    X_rna[mask, idx],
                    f"{sample} RNA: {gene}",
                )
                _scatter_spatial(
                    axes[1],
                    ge.obsm["spatial"][mask],
                    X_ge[mask, idx],
                    f"{sample} ATAC GE: {gene}",
                )
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def _write_spatial_cluster_reports(out_dir: str, rna, ge) -> None:
    if "spatial" not in rna.obsm:
        logging.warning("No spatial coordinates found; skipping spatial cluster maps.")
        return
    cluster_key = None
    for key in ["sg_leiden_merged", "sg_leiden"]:
        if key in rna.obs.columns:
            cluster_key = key
            break
    if cluster_key is None:
        logging.warning("No cluster labels found; skipping spatial cluster maps.")
        return

    samples = _sample_labels(rna)
    ge.obsm["spatial"] = rna.obsm["spatial"].copy()
    ge.obs[cluster_key] = rna.obs[cluster_key].values

    with PdfPages(os.path.join(out_dir, "spatial_cluster_maps.pdf")) as pdf:
        for sample in sorted(set(samples)):
            mask = samples == sample
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            _scatter_spatial(
                axes[0],
                rna.obsm["spatial"][mask],
                rna.obs[cluster_key].astype(str).values[mask],
                f"{sample} RNA clusters",
                categorical=True,
            )
            _scatter_spatial(
                axes[1],
                ge.obsm["spatial"][mask],
                ge.obs[cluster_key].astype(str).values[mask],
                f"{sample} ATAC GE clusters",
                categorical=True,
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _write_attention_reports(out_dir: str, adata) -> None:
    if "spatial" not in adata.obsm:
        logging.warning("No spatial coordinates found; skipping attention maps.")
        return
    if "alpha_omics1" not in adata.obsm or "alpha_omics2" not in adata.obsm:
        logging.warning("SpatialGlue attention weights missing; skipping attention maps.")
        return

    adata.obs["alpha_RNA_mean"] = np.asarray(adata.obsm["alpha_omics1"]).mean(axis=1)
    adata.obs["alpha_ATAC_mean"] = np.asarray(adata.obsm["alpha_omics2"]).mean(axis=1)
    samples = _sample_labels(adata)

    with PdfPages(os.path.join(out_dir, "spatialglue_attention_maps.pdf")) as pdf:
        for sample in sorted(set(samples)):
            mask = samples == sample
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            _scatter_spatial(
                axes[0],
                adata.obsm["spatial"][mask],
                adata.obs["alpha_RNA_mean"].values[mask],
                f"{sample} RNA attention",
                cmap="Blues",
            )
            _scatter_spatial(
                axes[1],
                adata.obsm["spatial"][mask],
                adata.obs["alpha_ATAC_mean"].values[mask],
                f"{sample} ATAC attention",
                cmap="Reds",
            )
            cluster_key = "sg_leiden_merged" if "sg_leiden_merged" in adata.obs else "sg_leiden"
            _scatter_spatial(
                axes[2],
                adata.obsm["spatial"][mask],
                adata.obs[cluster_key].astype(str).values[mask],
                f"{sample} SpatialGlue clusters",
                categorical=True,
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _write_umi_reports(
    out_dir: str,
    rna,
    genes,
    X_counts,
    X_expr,
    report_genes: list[str],
) -> None:
    cluster_key = None
    for key in ["sg_leiden_merged", "sg_leiden"]:
        if key in rna.obs.columns:
            cluster_key = key
            break
    if cluster_key is None:
        logging.warning("No cluster labels found; skipping UMI reports.")
        return
    if not report_genes:
        logging.warning("No report genes available; skipping UMI reports.")
        return

    gene_names = pd.Index(genes).astype(str)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    selected = [g for g in report_genes if g in gene_to_idx]
    if not selected:
        logging.warning("No report genes found in RNA object; skipping UMI reports.")
        return

    labels = rna.obs[cluster_key].astype(str).values
    clusters = sorted(set(labels), key=_cluster_sort_key)
    rows = []
    for cluster in clusters:
        mask = labels == cluster
        n_spots = int(mask.sum())
        for gene in selected:
            vals = X_counts[mask, gene_to_idx[gene]]
            rows.append({
                "cluster": cluster,
                "n_spots_in_cluster": n_spots,
                "gene": gene,
                "total_umi": float(vals.sum()),
                "mean_umi_per_spot": float(vals.mean()),
                "pct_spots_expressing": float((vals > 0).mean() * 100),
            })

    umi = pd.DataFrame(rows)
    umi_path = os.path.join(out_dir, "umi_per_cluster_genes_of_interest.csv")
    umi.to_csv(umi_path, index=False)
    logging.info(f"Saved per-cluster UMI table: {umi_path}")

    plot_genes = selected[:12]
    with PdfPages(os.path.join(out_dir, "umi_violin_per_cluster.pdf")) as pdf:
        for gene in plot_genes:
            idx = gene_to_idx[gene]
            fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(clusters)), 4))
            data = [X_expr[labels == cluster, idx] for cluster in clusters]
            ax.violinplot(data, showmeans=True, showextrema=False)
            ax.set_xticks(np.arange(1, len(clusters) + 1))
            ax.set_xticklabels(clusters, rotation=45)
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Expression")
            ax.set_title(f"{gene} expression by cluster")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    mean_pivot = umi.pivot_table(
        index="cluster", columns="gene", values="mean_umi_per_spot"
    ).reindex(index=clusters)
    fig, ax = plt.subplots(
        figsize=(max(6, 0.5 * len(selected)), max(4, 0.35 * len(clusters)))
    )
    im = ax.imshow(mean_pivot.fillna(0).values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(mean_pivot.columns)))
    ax.set_xticklabels(mean_pivot.columns, rotation=90)
    ax.set_yticks(np.arange(len(mean_pivot.index)))
    ax.set_yticklabels(mean_pivot.index)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Cluster")
    ax.set_title("Mean raw UMI per spot")
    fig.colorbar(im, ax=ax, label="mean UMI")
    fig.tight_layout()
    dot_path = os.path.join(out_dir, "umi_dotplot_per_cluster.pdf")
    fig.savefig(dot_path, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation_overview(corr_df: pd.DataFrame, outpath: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    hb = ax.hexbin(
        np.log10(corr_df["corr_mean_umi"] + 0.01),
        corr_df["spearman_rho"],
        gridsize=60,
        cmap="YlOrRd",
        bins="log",
        mincnt=1,
    )
    plt.colorbar(hb, ax=ax, label="log10(count)")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("log10(mean UMI + 0.01)")
    ax.set_ylabel("Spearman rho (ATAC GE vs RNA)")
    ax.set_title("Gene abundance vs ATAC-RNA correlation")

    ax = axes[1]
    ax.hist(corr_df["spearman_rho"], bins=120, color="steelblue", edgecolor="none")
    median_rho = corr_df["spearman_rho"].median()
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.axvline(
        median_rho,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"median = {median_rho:.3f}",
    )
    ax.set_xlabel("Spearman rho")
    ax.set_ylabel("N genes")
    ax.set_title("Distribution of ATAC-RNA correlations")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _write_cluster_correlation_outputs(
    out_dir: str,
    rna,
    genes,
    X_rna,
    X_ge,
    corr_df: pd.DataFrame,
    genes_of_interest: Optional[str],
    n_top_heatmap: int = 20,
) -> None:
    cluster_key = None
    for key in ["sg_leiden_merged", "sg_leiden"]:
        if key in rna.obs.columns:
            cluster_key = key
            break
    if cluster_key is None:
        logging.warning("No cluster labels found; skipping per-cluster correlation outputs.")
        return

    gene_names = pd.Index(genes).astype(str)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    requested_genes = _parse_gene_list(genes_of_interest)
    corr_genes = corr_df["gene"].astype(str).tolist()
    if requested_genes:
        selected_genes = [g for g in requested_genes if g in corr_genes and g in gene_to_idx]
        missing = sorted(set(requested_genes) - set(selected_genes))
        if missing:
            logging.warning(
                f"Skipping genes not found in filtered correlation table: {missing}"
            )
    else:
        selected_genes = corr_df.sort_values(
            "abs_rho", ascending=False
        )["gene"].astype(str).head(n_top_heatmap).tolist()

    if not selected_genes:
        logging.warning("No genes available for per-cluster correlation outputs.")
        return

    cluster_labels = rna.obs[cluster_key].astype(str).values
    clusters = sorted(set(cluster_labels), key=_cluster_sort_key)
    rows = []
    for cluster in clusters:
        mask = cluster_labels == cluster
        n_spots = int(mask.sum())
        for gene in selected_genes:
            idx = gene_to_idx[gene]
            rows.append({
                "cluster": cluster,
                "n_spots": n_spots,
                "gene": gene,
                "mean_rna_lognorm": float(X_rna[mask, idx].mean()),
                "mean_atac_ge": float(X_ge[mask, idx].mean()),
            })

    per_cluster = pd.DataFrame(rows)
    per_cluster_path = os.path.join(out_dir, "per_cluster_rna_atac_ge.csv")
    per_cluster.to_csv(per_cluster_path, index=False)
    logging.info(f"Saved per-cluster RNA/ATAC GE table: {per_cluster_path}")

    plot_genes = selected_genes[:10]
    fig, axes = plt.subplots(
        2,
        len(plot_genes),
        figsize=(max(5, 3.2 * len(plot_genes)), 7),
        squeeze=False,
    )
    for j, gene in enumerate(plot_genes):
        sub = per_cluster[per_cluster["gene"] == gene]
        labels = sub["cluster"].astype(str).tolist()
        axes[0, j].bar(labels, sub["mean_rna_lognorm"], color="steelblue")
        axes[0, j].set_title(f"{gene}\nRNA")
        axes[0, j].tick_params(axis="x", rotation=45)
        axes[1, j].bar(labels, sub["mean_atac_ge"], color="coral")
        axes[1, j].set_title("ATAC GE")
        axes[1, j].tick_params(axis="x", rotation=45)
    axes[0, 0].set_ylabel("Mean")
    axes[1, 0].set_ylabel("Mean")
    fig.tight_layout()
    bar_path = os.path.join(out_dir, "per_cluster_rna_atac_ge.pdf")
    fig.savefig(bar_path, bbox_inches="tight")
    plt.close(fig)

    heatmap_genes = corr_df.sort_values(
        "abs_rho", ascending=False
    )["gene"].astype(str).head(n_top_heatmap).tolist()
    heatmap_genes = [g for g in heatmap_genes if g in gene_to_idx]
    if requested_genes:
        heatmap_genes = sorted(set(heatmap_genes + selected_genes))
    if not heatmap_genes:
        return

    heatmap_rows = []
    for cluster in clusters:
        mask = cluster_labels == cluster
        heatmap_rows.append({
            gene: float(X_rna[mask, gene_to_idx[gene]].mean())
            for gene in heatmap_genes
        })
    heatmap = pd.DataFrame(heatmap_rows, index=clusters)
    heatmap_scaled = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9)

    fig, ax = plt.subplots(
        figsize=(max(8, 0.45 * len(heatmap_genes)), max(4, 0.35 * len(clusters)))
    )
    im = ax.imshow(heatmap_scaled.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(heatmap_genes)))
    ax.set_xticklabels(heatmap_genes, rotation=90)
    ax.set_yticks(np.arange(len(clusters)))
    ax.set_yticklabels(clusters)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Cluster")
    ax.set_title("Scaled RNA expression by SpatialGlue cluster")
    fig.colorbar(im, ax=ax, label="scaled mean")
    fig.tight_layout()
    heatmap_path = os.path.join(out_dir, "cluster_gene_heatmap.pdf")
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)


@custom_task(cpu=4, memory=200, storage_gib=1000)
def glue_preprocess_task(
    project_name: str,
    atac_anndata: LatchFile,
    wt_anndata: LatchFile,
    ge_anndata: LatchFile,
) -> LatchDir:

    # ------------------ Initialize ---------------------
    logging.info("Starting glue preprocessing task...")
    out_dir = f"/root/{project_name}_preprocess"
    os.makedirs(out_dir, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    logging.info("Reading WT AnnData...")
    rna = sc.read_h5ad(wt_anndata.local_path)

    logging.info("Reading ATAC tile AnnData...")
    atac = sc.read_h5ad(atac_anndata.local_path)

    logging.info("Reading gene accessibility AnnData...")
    ge = sc.read_h5ad(ge_anndata.local_path)

    logging.info(
        f"n_obs RNA: {rna.n_obs} n_obs ATAC tiles: {atac.n_obs} "
        f"n_obs GE: {ge.n_obs}"
    )
    logging.info(f"\nRNA obs_names examples: {list(map(str, rna.obs_names[:5]))}")
    logging.info(f"ATAC obs_names examples: {list(map(str, atac.obs_names[:5]))}")
    logging.info(f"GE obs_names examples: {list(map(str, ge.obs_names[:5]))}")

    rna.obs_names = utils.ensure_obs_run_barcodes(rna, "RNA")
    atac.obs_names = utils.ensure_obs_run_barcodes(atac, "ATAC")
    ge.obs_names = utils.ensure_obs_run_barcodes(ge, "Gene accessibility")

    rna.var_names = utils.ensure_var_gene_symbols(rna, "RNA", min_fraction=0.5)
    ge.var_names = utils.ensure_var_gene_symbols(ge, "GE", min_fraction=0.5)

    rna.var_names_make_unique()
    ge.var_names_make_unique()

    rna_matched, ge_matched, atac_tiles_matched = _align_modalities(rna, ge, atac)
    logging.info(
        "Matched barcodes across RNA/GE/ATAC tiles: "
        f"{rna_matched.n_obs} spots"
    )

    # -------------------- GE cleanup + LSI --------------------
    logging.info("Cleaning gene accessibility AnnData...")
    if hasattr(ge_matched.X, "tocsr"):
        ge_matched.X = ge_matched.X.tocsr()

    ge_sums = np.array(ge_matched.X.sum(axis=0)).ravel()

    keep_var = ge_sums > 0
    if (~keep_var).sum():
        print(f"Dropping {(~keep_var).sum()} zero-signal GE features")
        ge_matched = ge_matched[:, keep_var].copy()

    logging.info("LSI on gene accessibility AnnData...")
    ge_matched.obsm["X_lsi"] = _compute_lsi(ge_matched.X)
    ge_matched.obsm["feat"] = ge_matched.obsm["X_lsi"].astype("float32")
    logging.info(f"GE SpatialGlue features: {ge_matched.obsm['feat'].shape}")

    # -------------------- RNA features with HVG guard ------------------
    logging.info("Adding feat to WT AnnData...")
    _add_rna_features(rna_matched)

    logging.info("Writing prepared SpatialGlue inputs...")
    rna_matched.write(f"{out_dir}/rna_prepared.h5ad")
    ge_matched.write(f"{out_dir}/ge_prepared.h5ad")
    atac_tiles_matched.write(f"{out_dir}/atac_tiles_prepared.h5ad")

    return LatchDir(out_dir, f"latch:///glue_outs/{project_name}/preprocess")


@custom_task(cpu=126, memory=100, storage_gib=1000)
def glue_train_task(
    project_name: str,
    rna_prepared_h5ad: LatchFile,
    ge_prepared_h5ad: LatchFile,
    atac_tiles_prepared_h5ad: LatchFile,
    n_neighbors: int = 15,
    min_cluster_size: int = 200,
    resolutions: str = DEFAULT_RESOLUTIONS,
    chosen_resolution: float = 0.0,
) -> LatchDir:

    # ------------------ Initialize ---------------------
    logging.info("Starting glue training task...")
    out_dir = f"/root/{project_name}"
    os.makedirs(out_dir, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    logging.info("Reading prepared SpatialGlue inputs...")
    rna_matched = sc.read_h5ad(rna_prepared_h5ad.local_path)
    ge_matched = sc.read_h5ad(ge_prepared_h5ad.local_path)
    atac_tiles_matched = sc.read_h5ad(atac_tiles_prepared_h5ad.local_path)
    logging.info(
        f"Prepared inputs loaded: RNA {rna_matched.shape}, "
        f"GE {ge_matched.shape}, ATAC tiles {atac_tiles_matched.shape}"
    )

    # -------------------- Train SpatialGlue --------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info("construct_neighbor_graph...")
    data = construct_neighbor_graph(
        rna_matched, ge_matched, datatype="Spatial-epigenome-transcriptome"
    )

    logging.info("Train_SpatialGlue...")
    model = Train_SpatialGlue(
        data,
        datatype="Spatial-epigenome-transcriptome",
        random_seed=SEED,
        device=device,
    )
    logging.info("training...")
    out = model.train()

    rna_result = data["adata_omics1"]
    ge_result = data["adata_omics2"]

    # Collect embeddings/weights on an AnnData for downstream analysis
    logging.info("copy data...")
    adata = rna_result.copy()
    adata.obsm["SpatialGlue"] = out["SpatialGlue"]
    adata.obsm["alpha"] = out.get("alpha")
    adata.obsm["alpha_omics1"] = out.get("alpha_omics1")
    adata.obsm["alpha_omics2"] = out.get("alpha_omics2")

    # -------------------- Neighbors/UMAP/Leiden --------------------
    logging.info("clustering on spatialglue dims...")
    sc.pp.neighbors(
        adata, use_rep="SpatialGlue", n_neighbors=n_neighbors, random_state=SEED
    )
    sc.tl.umap(adata, random_state=SEED)

    sweep_rows = []
    final_raw_key = None
    final_merged_key = None
    best_moran = None
    best_raw_key = None
    best_merged_key = None
    best_resolution = None
    connectivities = _spatial_connectivities(adata, n_neighbors=n_neighbors)
    for resolution in _parse_resolutions(resolutions):
        key = f"sg_leiden_{_resolution_suffix(resolution)}"
        merged_key = f"{key}_merged"
        sc.tl.leiden(
            adata, resolution=resolution, random_state=SEED, key_added=key
        )
        adata.obs[key] = adata.obs[key].astype("category")

        utils.merge_small_clusters(
            adata,
            cluster_key=key,
            embed_key="SpatialGlue",
            min_cells=min_cluster_size,
            new_key=merged_key,
            verbose=True,
        )

        morans_i = _morans_i(
            connectivities,
            adata.obs[merged_key].cat.codes.values,
        )
        sweep_rows.append({
            "resolution": resolution,
            "raw_key": key,
            "merged_key": merged_key,
            "n_clusters_raw": int(adata.obs[key].nunique()),
            "n_clusters_merged": int(adata.obs[merged_key].nunique()),
            "morans_i": morans_i,
            "min_cluster_size": int(min_cluster_size),
            "n_neighbors": int(n_neighbors),
        })
        logging.info(
            f"Resolution {resolution:g}: {adata.obs[key].nunique()} raw "
            f"clusters, {adata.obs[merged_key].nunique()} merged clusters, "
            f"Moran's I={morans_i:.4f}"
        )

        if best_moran is None or morans_i > best_moran:
            best_moran = morans_i
            best_raw_key = key
            best_merged_key = merged_key
            best_resolution = resolution

        if chosen_resolution != 0 and np.isclose(resolution, chosen_resolution):
            final_raw_key = key
            final_merged_key = merged_key

    if chosen_resolution == 0:
        final_raw_key = best_raw_key
        final_merged_key = best_merged_key
        chosen_resolution = best_resolution
        logging.info(
            f"Auto-selected resolution {chosen_resolution:g} with "
            f"Moran's I={best_moran:.4f}"
        )

    if final_raw_key is None or final_merged_key is None:
        raise ValueError(
            f"chosen_resolution={chosen_resolution} was not found in "
            f"resolutions={resolutions}."
        )

    adata.obs["sg_leiden"] = adata.obs[final_raw_key].astype("category")
    adata.obs["sg_leiden_merged"] = adata.obs[final_merged_key].astype("category")

    sweep_path = os.path.join(out_dir, "spatialglue_cluster_sweep.csv")
    pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)
    logging.info(f"Saved SpatialGlue cluster sweep: {sweep_path}")

    # -------------------- Plots (merged and raw) --------------------
    logging.info("Plotting figures...")
    sc.pl.umap(adata, color=["sg_leiden_merged"], save="umap_merged.pdf")
    sc.pl.embedding(
        adata,
        basis="spatial",
        color=["sg_leiden_merged"],
        save="spatial_umap_merged.pdf"
    )

    sc.pl.umap(adata, color=["sg_leiden"], save="umap_unmerged.pdf")
    sc.pl.embedding(
        adata,
        basis="spatial",
        color=["sg_leiden"],
        save="spatial_umap_unmerged.pdf"
    )

    # Export coverage for newly assigned clusters

    # -------------------- Save data --------------------
    # Copy new clustering results
    for obj in [rna_result, ge_result, atac_tiles_matched]:
        obj.obs["sg_leiden"] = adata.obs.loc[obj.obs_names, "sg_leiden"].values
        obj.obs["sg_leiden_merged"] = adata.obs.loc[
            obj.obs_names, "sg_leiden_merged"
        ].values

    rna_result.obsm["SpatialGlue"] = adata.obsm["SpatialGlue"]
    for key in ["alpha", "alpha_omics1", "alpha_omics2"]:
        if key in adata.obsm and adata.obsm[key] is not None:
            rna_result.obsm[key] = adata.obsm[key]

    _write_attention_reports(out_dir, rna_result)

    logging.info("Creating coverages for new clusters...")
    coverage_dir = f"{out_dir}/glue_cluster_coverages"
    os.makedirs(coverage_dir, exist_ok=True)
    snap.ex.export_coverage(
        atac_tiles_matched,
        groupby="sg_leiden_merged",
        suffix="_cluster.bw",
        bin_size=10,
        output_format="bigwig",
    )
    bws = glob.glob("*.bw")
    if bws:
        subprocess.run(["mv"] + bws + [coverage_dir], check=True)
    else:
        logging.warning("No bigWig files were created by export_coverage.")
    logging.info("Finished coverages for new clusters...")

    logging.info("Writing data...")
    atac_tiles_matched.write(f"{out_dir}/atac_glue.h5ad")
    ge_result.write(f"{out_dir}/ge_glue.h5ad")
    rna_result.write(f"{out_dir}/rna_glue.h5ad")

    with open(f"{out_dir}/SpatialGlue_model.pickle", "wb") as f:
        pickle.dump(out, f)

    subprocess.run([f"mv /root/figures/* {out_dir}"], shell=True)

    return LatchDir(out_dir, f"latch:///glue_outs/{project_name}")


@custom_task(cpu=8, memory=200, storage_gib=1000)
def corr_task(
    project_name: str,
    results_dir: LatchDir,
    ge_anndata: LatchFile,
    min_umi_threshold: float = 0.5,
    min_frac_expressing: float = 0.05,
    genes_of_interest: Optional[str] = None,
) -> LatchDir:

    out_dir = f"/root/{project_name}"
    os.makedirs(out_dir, exist_ok=True)

    # Download and read data -----------------------------------------------
    logging.info("Downloading RNA data...")
    # Add check if exists on latch
    rna_path = LatchFile(f"{results_dir.remote_path}/rna_glue.h5ad").local_path

    logging.info("Downloading Gene Accessibility data...")
    ge_path = ge_anndata.local_path

    logging.info("Reading RNA data...")
    rna = sc.read_h5ad(rna_path)
    logging.info("Reading Gene Accessibility data...")
    ge = sc.read_h5ad(ge_path)

    logging.info("Preparing data for correlation...")
    rna.obs_names = utils.ensure_obs_run_barcodes(rna, "RNA")
    ge.obs_names = utils.ensure_obs_run_barcodes(ge, "Gene accessibility")

    # Make sure at least half are gene symbols
    rna.var_names = utils.ensure_var_gene_symbols(rna, "RNA", min_fraction=0.5)
    ge.var_names = utils.ensure_var_gene_symbols(ge, "GE", min_fraction=0.5)

    # Get Spearman correlation table -----------------------------------------
    rna.var_names_make_unique()
    ge.var_names_make_unique()

    # Reduce both to common genes/cells and align
    rna_sub, ge_sub = corr.synch_adata(rna, ge)

    genes = rna_sub.var_names

    logging.info("Ensuring dense matrix...")
    # Prefer monotonic transform (normalized/log1p/counts layers)
    preferred_layers = ["normalized", "log1p", "counts"]
    X_rna = None
    for layer in preferred_layers:
        if layer in rna_sub.layers:
            logging.info(f"Using RNA layer '{layer}' for correlation")
            message(
                typ="info",
                data={
                    "title": "RNA Layers", "body":
                    f"Using RNA layer '{layer}' for correlation"
                }
            )
            X_rna = utils.to_dense(rna_sub.layers[layer]).astype(np.float32)
            break
    if X_rna is None:  # Fall back to .X with warning
        logging.warning(
            "RNA layers normalized/log1p/counts not found; using rna_sub.X"
        )
        message(
            typ="warning",
            data={
                "title": "RNA Layers",
                "body": """Using RNA .X for correlation analysis.  If
                Transcriptome AnnData is from an RNAQC or optimize_wt Workflow,
                this represents scaled data which is suboptimal for
                correlations."""
            }
        )
        X_rna = utils.to_dense(rna_sub.X).astype(np.float32)
    X_ge = utils.to_dense(ge_sub.X).astype(np.float32)

    # Gene stats ------------------------------------------------------------
    X_rna_counts, rna_source = gs.get_rna_counts_matrix(rna_sub)
    X_rna_counts_dense = utils.to_dense(X_rna_counts).astype(np.float32)
    mean_umi = X_rna_counts_dense.mean(axis=0)
    frac_expressing = (X_rna_counts_dense > 0).mean(axis=0)
    keep = (
        (mean_umi >= float(min_umi_threshold))
        & (frac_expressing >= float(min_frac_expressing))
    )
    n_keep = int(keep.sum())
    rna_stats = gs.compute_gene_stats_matrix(
        X_rna_counts, genes, prefix="rna_umi", include_minmax_nonzero=True
    )
    filter_stats = pd.DataFrame({
        "gene": genes.astype(str).values,
        "corr_mean_umi": mean_umi,
        "corr_frac_expressing": frac_expressing,
        "passes_corr_filter": keep,
        "corr_min_umi_threshold": float(min_umi_threshold),
        "corr_min_frac_expressing": float(min_frac_expressing),
    })

    ge_stats = gs.compute_gene_stats_matrix(
        sparse.csr_matrix(X_ge),
        genes,
        prefix="ge_norm",
        include_minmax_nonzero=False
    )

    # Merge all stats before correlation so the workflow still produces a
    # useful QC table if all genes fail the correlation filters.
    stats = (
        rna_stats.merge(ge_stats, on="gene", how="inner")
        .merge(filter_stats, on="gene", how="inner")
    )

    res_path = os.path.join(out_dir, "atac-ge_vs_rna_spearman.csv")
    if n_keep == 0:
        msg = (
            "No genes passed correlation filters: "
            f"mean RNA UMI >= {min_umi_threshold}, "
            f"fraction expressing >= {min_frac_expressing}. "
            "Skipping correlation table generation and correlation plots."
        )
        logging.warning(msg)
        message(
            typ="warning",
            data={
                "title": "No correlation genes",
                "body": msg,
            },
        )
        pd.DataFrame(columns=[
            "gene",
            "spearman_rho",
            "pval",
            "qval_bh",
            "mean_RNA",
            "mean_GA",
            "abs_rho",
        ]).to_csv(res_path, index=False)
        pd.DataFrame(columns=[
            "gene",
            "spearman_r",
            "mean_umi",
            "frac_expressing",
            "pval",
            "qval_bh",
        ]).to_csv(
            os.path.join(out_dir, "atac_rna_spearman_all_genes.csv"),
            index=False,
        )
        logging.info(f"Saved empty Spearman results: {res_path}")

        stats_path = os.path.join(out_dir, "gene_stats.csv")
        stats.to_csv(stats_path, index=False)
        logging.info(f"Saved: {stats_path} (RNA counts source: {rna_source})")
        report_genes = _selected_report_genes(genes_of_interest, genes)
        _write_spatial_expression_reports(
            out_dir=out_dir,
            rna=rna_sub,
            ge=ge_sub,
            genes=genes,
            X_rna=X_rna,
            X_ge=X_ge,
            report_genes=report_genes,
        )
        _write_spatial_cluster_reports(out_dir, rna_sub, ge_sub)
        _write_umi_reports(
            out_dir=out_dir,
            rna=rna_sub,
            genes=genes,
            X_counts=X_rna_counts_dense,
            X_expr=X_rna,
            report_genes=report_genes,
        )
        return LatchDir(out_dir, f"latch:///glue_outs/{project_name}")

    logging.info(
        f"Correlation filters retained {n_keep}/{len(genes)} genes "
        f"(mean RNA UMI >= {min_umi_threshold}, "
        f"fraction expressing >= {min_frac_expressing})."
    )

    logging.info("Computing correlations...")
    res = corr.get_corr_df(X_rna[:, keep], X_ge[:, keep], genes[keep])
    res.to_csv(res_path, index=False)
    logging.info(f"Saved Spearman results: {res_path}")

    stats = stats.merge(res, on="gene", how="left")
    corr_with_filter = res.merge(filter_stats, on="gene", how="left")
    notebook_corr = corr_with_filter.rename(columns={
        "spearman_rho": "spearman_r",
        "corr_mean_umi": "mean_umi",
        "corr_frac_expressing": "frac_expressing",
    })
    notebook_corr_path = os.path.join(out_dir, "atac_rna_spearman_all_genes.csv")
    notebook_corr[[
        "gene",
        "spearman_r",
        "mean_umi",
        "frac_expressing",
        "pval",
        "qval_bh",
        "abs_rho",
    ]].to_csv(notebook_corr_path, index=False)
    logging.info(f"Saved notebook-style Spearman table: {notebook_corr_path}")

    stats_path = os.path.join(out_dir, "gene_stats.csv")
    stats.to_csv(stats_path, index=False)
    logging.info(f"Saved: {stats_path} (RNA counts source: {rna_source})")

    overview_path = os.path.join(out_dir, "atac_rna_correlation_overview.pdf")
    logging.info(f"Saving correlation overview to {overview_path}")
    _plot_correlation_overview(corr_with_filter, overview_path)

    report_genes = _selected_report_genes(genes_of_interest, genes, res)
    _write_spatial_expression_reports(
        out_dir=out_dir,
        rna=rna_sub,
        ge=ge_sub,
        genes=genes,
        X_rna=X_rna,
        X_ge=X_ge,
        report_genes=report_genes,
    )
    _write_spatial_cluster_reports(out_dir, rna_sub, ge_sub)
    _write_umi_reports(
        out_dir=out_dir,
        rna=rna_sub,
        genes=genes,
        X_counts=X_rna_counts_dense,
        X_expr=X_rna,
        report_genes=report_genes,
    )

    _write_cluster_correlation_outputs(
        out_dir=out_dir,
        rna=rna_sub,
        genes=genes,
        X_rna=X_rna,
        X_ge=X_ge,
        corr_df=res,
        genes_of_interest=genes_of_interest,
    )

    bar_path = os.path.join(out_dir, "top_genes_bar.pdf")
    logging.info(f"Saving top correlated genes figure to {bar_path}")
    pl.plot_top_genes_bar(res, n=20, fdr_thresh=0.05, outpath=bar_path)

    volcano_path = os.path.join(out_dir, "corr_volcano.pdf")
    logging.info(f"Saving volcano to {volcano_path}")
    pl.plot_corr_volcano_broken(
        res,
        outpath=volcano_path,
        q_thresh=0.05,
        rho_thresh=0.1,
        y_low=(0, 300),       # bottom panel exactly 0–300
        y_high=(300, 311),   # top panel starts at 308; upper bound auto
        jitter_y=0.3,         # keep points from crossing the break
        top_pos_labels=10,
        top_neg_labels=10,
        title="Correlation volcano"
    )

    return LatchDir(out_dir, f"latch:///glue_outs/{project_name}")
