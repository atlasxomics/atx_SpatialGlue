import math
import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import gridspec
from scipy import sparse as sp
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import pdist

import wf.utils as utils


def plot_corr_volcano_broken(
    corr,
    outpath="corr_volcano.png",
    gene_col="gene",
    r_col="spearman_rho",
    q_col="qval_bh",
    q_thresh=0.05,
    rho_thresh=0.30,
    top_pos_labels=8,
    top_neg_labels=8,
    highlight_genes=None,
    # y-axis ranges in -log10 units (set any to None for auto)
    y_low=(0, None),          # bottom panel
    y_high=(None, None),      # top panel
    background_size=10,
    sig_size=22,
    hl_size=30,
    jitter_y=0.0,
    title="Correlation volcano"
):

    df = corr.copy()

    for c in (gene_col, r_col, q_col):
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'")

    # ---- y = -log10(q) with safe handling for q==0 ----
    q = df[q_col].astype(float).to_numpy()
    nz = q > 0
    if nz.any():
        cap_y = -np.log10(q[nz].min()) + 1.0
    else:
        cap_y = 10.0
    y = np.empty_like(q, dtype=float)
    y[nz] = -np.log10(q[nz])
    y[~nz] = cap_y

    # optional jitter to separate identical FDRs
    if jitter_y and jitter_y > 0:
        rng = np.random.default_rng(42)
        y += rng.normal(0, jitter_y, size=y.shape)

    df["_y"] = y
    df["_abs_rho"] = df[r_col].abs()

    # ---- masks ----
    sig = (df[q_col] <= q_thresh) & (df["_abs_rho"] >= rho_thresh)
    sig_pos = sig & (df[r_col] >= 0)
    sig_neg = sig & (df[r_col] < 0)

    # ---- x range (symmetric) ----
    xpad = 0.05
    xlim = max(abs(df[r_col].min()), abs(df[r_col].max())) + xpad

    # ---- choose y ranges (auto if None) ----
    y_low_min, y_low_max = y_low
    y_high_min, y_high_max = y_high

    # sensible defaults if any are None
    if y_low_min is None:
        y_low_min = 0
    if y_low_max is None:
        y_low_max = max(6.0, -math.log10(q_thresh) + 2.0)  # detailed band
    if y_high_min is None:
        y_high_min = max(y_low_max + 5.0, 200.0)           # gap between bands
    if y_high_max is None:
        y_high_max = max(np.quantile(df["_y"], 0.995), cap_y)

    # if user-specified ranges overlap, fix ordering
    if y_low_max >= y_high_min:
        y_high_min = y_low_max + 5.0

    # ---- figure (avoid tight_layout with broken axes) ----
    fig = plt.figure(figsize=(7.2, 7.4))

    # with this (60% top / 40% bottom):
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.60, 0.40], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    def _scatter(ax):
        ax.scatter(
            df[r_col],
            df["_y"],
            s=background_size,
            c="#C7C7C7",
            alpha=0.5,
            linewidths=0,
            label="All"
        )
        if sig_neg.any():
            neg = df[sig_neg]
            ax.scatter(
                neg[r_col],
                neg["_y"],
                s=sig_size,
                c="#3B82F6",
                alpha=0.9,
                linewidths=0,
                label=f"Sig (ρ ≤ -{rho_thresh:g})"
            )
        if sig_pos.any():
            pos = df[sig_pos]
            ax.scatter(
                pos[r_col],
                pos["_y"],
                s=sig_size,
                c="#EF4444",
                alpha=0.9,
                linewidths=0,
                label=f"Sig (ρ ≥ {rho_thresh:g})"
            )
        ax.axhline(
            -math.log10(q_thresh), color="k", linestyle="--", linewidth=1
        )
        ax.axvline(+rho_thresh, color="k", linestyle="--", linewidth=1)
        ax.axvline(-rho_thresh, color="k", linestyle="--", linewidth=1)

    for ax in (ax_top, ax_bot):
        _scatter(ax)
        ax.set_xlim(-xlim, xlim)

    ax_bot.set_ylim(y_low_min,  y_low_max)
    ax_top.set_ylim(y_high_min, y_high_max)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # ---- break marks ----
    d = .008
    kw = dict(color='k', clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_top.plot((1-d, 1+d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_bot.plot((-d, +d), (1-d, 1+d), transform=ax_bot.transAxes, **kw)
    ax_bot.plot((1-d, 1+d), (1-d, 1+d), transform=ax_bot.transAxes, **kw)

    # ---- labels ----
    def _label(ax, rows, ha):
        for _, r in rows.iterrows():
            ax.text(
                r[r_col],
                r["_y"],
                str(r[gene_col]),
                fontsize=8,
                ha=ha,
                va="bottom"
            )

    if top_pos_labels and sig_pos.any():
        lab = df[sig_pos].nlargest(top_pos_labels, "_abs_rho")
        _label(ax_top, lab[lab["_y"] >= y_high_min], "left")
        _label(ax_bot, lab[lab["_y"] < y_high_min], "left")

    if top_neg_labels and sig_neg.any():
        lab = df[sig_neg].nlargest(top_neg_labels, "_abs_rho")
        _label(ax_top, lab[lab["_y"] >= y_high_min], "right")
        _label(ax_bot, lab[lab["_y"] < y_high_min], "right")

    if highlight_genes:
        sel = df[df[gene_col].astype(str).isin(set(highlight_genes))]
        for ax in (ax_top, ax_bot):
            if not sel.empty:
                ax.scatter(sel[r_col], sel["_y"], s=hl_size, facecolors="none", edgecolors="black", linewidths=1.2, label="Highlighted")
                for _, r in sel.iterrows():
                    ax.text(r[r_col], r["_y"], str(r[gene_col]), fontsize=9, weight="bold", ha="center", va="bottom")

    # ---- cosmetics ----
    ax_top.set_title(title)
    ax_bot.set_xlabel("Spearman correlation (ρ)")
    ax_bot.set_ylabel(r"$-\,\log_{10}(\mathrm{FDR})$")
    h, l = ax_top.get_legend_handles_labels()
    if h:
        ax_top.legend(h[:3], l[:3], frameon=False, loc="upper left")

    # use subplots_adjust instead of tight_layout for broken axes
    fig.subplots_adjust(
        hspace=0.05, left=0.12, right=0.98, top=0.92, bottom=0.10
    )
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_top_genes_bar(
    corr: pd.DataFrame,
    n: int = 20,
    fdr_thresh: float = 0.05,
    outpath="top_genes_bar.png",
    r_col="spearman_rho",
    gene_col="gene",
    fdr_col="qval_bh",
    use_abs=False
):
    """
    use_abs: if True, rank by |r| (uses 'abs_rho' if present, else abs of
        r_col).
    """
    df = corr.copy()

    # Basic checks
    if gene_col not in df.columns:
        raise ValueError(f"Missing gene column '{gene_col}' in CSV.")
    if r_col not in df.columns:
        # fallbacks if someone renamed columns
        for c in ["spearman_rho", "r_spearman", "spearman_r", "rho", "r", "correlation"]:
            if c in df.columns:
                r_col = c
                break
        else:
            raise ValueError("Could not find a correlation column.")
    if fdr_col and fdr_col not in df.columns:
        # allow common FDR names
        for c in ["qval_bh", "fdr", "qval", "q_value", "padj", "p_adj", "FDR", "q"]:
            if c in df.columns:
                fdr_col = c
                break
        else:
            fdr_col = None  # proceed without FDR filter

    # Filter by FDR if available
    if fdr_col:
        df = df[df[fdr_col] <= fdr_thresh]

    # Decide ranking column
    if use_abs:
        rank_col = "abs_rho" if "abs_rho" in df.columns else None
        if rank_col is None:
            df["_abs_tmp"] = df[r_col].abs()
            rank_col = "_abs_tmp"
    else:
        rank_col = r_col

    # Rank & select
    df = df[[gene_col, r_col] + ([rank_col] if rank_col not in (r_col, gene_col) else [])].dropna()
    df = df.sort_values(rank_col, ascending=False).head(n).iloc[::-1]  # reverse so largest at top

    # Plot
    plt.figure(figsize=(6, 8))
    plt.barh(df[gene_col].astype(str), df[r_col].astype(float))
    plt.xlabel("Correlation (Spearman ρ)")
    plt.title(f"Top {len(df)} genes by {'|ρ|' if use_abs else 'ρ'}")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def plot_marker_heatmap(
    adata,
    top_genes_per_cluster: Dict[str, List[str]],
    output_path: str,
    groupby: str = "cluster",
    marker_top_n: int = 50,
) -> pd.DataFrame:
    """Plot mean marker expression by cluster as a z-scored heatmap."""

    clusters = list(top_genes_per_cluster.keys())
    seen = set()
    all_top_genes: List[str] = []
    for cluster in clusters:
        for gene in top_genes_per_cluster[cluster]:
            if gene not in seen:
                all_top_genes.append(gene)
                seen.add(gene)

    if len(clusters) == 0 or len(all_top_genes) == 0:
        raise ValueError("No marker genes available for heatmap.")

    gene_idx = adata.var_names.get_indexer(all_top_genes)
    valid = gene_idx >= 0
    ordered_input_genes = [
        gene for gene, keep in zip(all_top_genes, valid) if keep
    ]
    gene_idx = gene_idx[valid]
    if len(ordered_input_genes) == 0:
        raise ValueError("None of the marker genes were present in AnnData.")

    X = adata.X
    mean_expr = pd.DataFrame(
        index=clusters,
        columns=ordered_input_genes,
        dtype=float,
    )
    for cluster in clusters:
        mask = adata.obs[groupby].astype(str) == cluster
        sub = X[mask.to_numpy(), :][:, gene_idx]
        if sp.issparse(sub):
            sub = sub.toarray()
        mean_expr.loc[cluster] = np.asarray(sub).mean(axis=0)

    if len(clusters) > 1:
        values = mean_expr.to_numpy(dtype=float)
        scaled = (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-9)
        Z = linkage(pdist(scaled, metric="euclidean"), method="ward")
        cluster_order = [clusters[i] for i in leaves_list(Z)]
    else:
        Z = None
        cluster_order = clusters

    ordered_gene_set = set(ordered_input_genes)
    seen_ordered = set()
    ordered_genes: List[str] = []
    for cluster in cluster_order:
        for gene in top_genes_per_cluster[cluster]:
            if gene in ordered_gene_set and gene not in seen_ordered:
                ordered_genes.append(gene)
                seen_ordered.add(gene)

    heatmap_data = mean_expr.loc[cluster_order, ordered_genes].astype(float)
    heatmap_zscore = heatmap_data.apply(
        lambda col: (col - col.mean()) / (col.std() + 1e-9),
        axis=0,
    ).clip(-3, 3)

    n_clusters = len(cluster_order)
    n_genes = len(ordered_genes)
    label_every_n = 5
    gene_labels = [
        gene if i % label_every_n == 0 else ""
        for i, gene in enumerate(ordered_genes)
    ]

    fig_w = max(14, n_genes * 0.07)
    fig_h = max(8, n_clusters * 0.50)
    fig = plt.figure(figsize=(fig_w, fig_h))
    if Z is not None:
        gs = gridspec.GridSpec(1, 2, width_ratios=[0.03, 0.97], wspace=0.008)
        ax_dend = fig.add_subplot(gs[0])
        ax_heat = fig.add_subplot(gs[1])
        dendrogram(
            Z,
            orientation="left",
            ax=ax_dend,
            no_labels=True,
            link_color_func=lambda _: "#444444",
        )
        ax_dend.set_axis_off()
    else:
        ax_heat = fig.add_subplot(111)

    im = ax_heat.imshow(
        heatmap_zscore.values,
        aspect="auto",
        cmap="RdYlBu_r",
        vmin=-3,
        vmax=3,
        interpolation="nearest",
    )
    ax_heat.set_yticks(range(n_clusters))
    ax_heat.set_yticklabels(cluster_order, fontsize=10)
    ax_heat.yaxis.set_tick_params(length=0)
    ax_heat.set_ylabel("Cluster", fontsize=11)

    ax_heat.set_xticks(range(n_genes))
    ax_heat.set_xticklabels(gene_labels, rotation=90, fontsize=6.5, ha="center")
    ax_heat.xaxis.set_tick_params(length=0)

    pos = 0
    for cluster in cluster_order:
        pos += len([
            gene for gene in top_genes_per_cluster[cluster]
            if gene in seen_ordered
        ])
        if pos < n_genes:
            ax_heat.axvline(
                x=pos - 0.5,
                color="white",
                linewidth=0.8,
                alpha=0.8,
            )

    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.4, pad=0.01, aspect=25)
    cbar.set_label("Z-score", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    ax_heat.set_title(
        f"Top {marker_top_n} DEGs per Cluster (Wilcoxon, ordered by similarity)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return heatmap_zscore


def sample_labels(adata):
    for col in ["sample", "sample_name", "sample_id", "library_id", "batch"]:
        if col in adata.obs.columns:
            return adata.obs[col].astype(str).values
    names = pd.Index(adata.obs_names).astype(str)
    if names.str.contains("#").any():
        return names.str.split("#").str[0].values
    return np.repeat("all", adata.n_obs)


def _cluster_color_map(labels) -> dict[str, object]:
    categories = sorted(set(pd.Index(labels).astype(str)), key=utils.cluster_sort_key)
    cmap = plt.get_cmap("tab20")
    n_colors = max(len(categories), 1)
    return {
        label: cmap(i % cmap.N / max(min(n_colors, cmap.N) - 1, 1))
        for i, label in enumerate(categories)
    }


def _scatter_clusters(
    ax,
    coords,
    labels,
    title: str,
    color_map: dict[str, object],
    point_size: float,
    invert_y: bool = False,
    show_legend: bool = True,
) -> None:
    labels = pd.Index(labels).astype(str).to_numpy()
    colors = [color_map[label] for label in labels]
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=point_size,
        linewidths=0,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if invert_y:
        ax.set_aspect("equal")
        ax.invert_yaxis()

    if show_legend:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=5,
                label=str(label),
            )
            for label, color in color_map.items()
        ]
        ax.legend(
            handles=handles,
            title="cluster",
            fontsize=6,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )


def write_cluster_resolution_plots(out_dir: str, adata, cluster_keys: list[str]) -> None:
    """Write UMAP and spatial cluster plots for each clustering label column."""
    keys = []
    for key in cluster_keys:
        if key in adata.obs.columns and key not in keys:
            keys.append(key)
    if not keys:
        logging.warning("No cluster label columns found; skipping resolution plots.")
        return

    if "X_umap" not in adata.obsm:
        logging.warning("No UMAP embedding found; skipping resolution UMAP plots.")
    else:
        umap = np.asarray(adata.obsm["X_umap"])
        for key in keys:
            labels = adata.obs[key].astype(str).values
            color_map = _cluster_color_map(labels)
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            _scatter_clusters(
                ax,
                umap,
                labels,
                f"SpatialGlue UMAP: {key}",
                color_map,
                utils.SCANPY_CLUSTER_POINT_SIZE,
                show_legend=True,
            )
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            fig.tight_layout()
            fig.savefig(
                utils.fig_path(out_dir, f"clustering/umap_{utils.safe_name(key)}.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

    if "spatial" not in adata.obsm:
        logging.warning("No spatial coordinates found; skipping resolution spatial plots.")
        return

    coords = np.asarray(adata.obsm["spatial"])
    sample_values = sample_labels(adata)
    samples = sorted(set(sample_values))
    ncols = min(4, len(samples))
    nrows = int(np.ceil(len(samples) / ncols))

    for key in keys:
        labels = adata.obs[key].astype(str).values
        color_map = _cluster_color_map(labels)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.5 * ncols, 4.5 * nrows),
            squeeze=False,
        )
        axes = axes.ravel()
        for i, sample in enumerate(samples):
            mask = sample_values == sample
            _scatter_clusters(
                axes[i],
                coords[mask],
                labels[mask],
                f"{sample}: {key}",
                color_map,
                utils.SPATIAL_SCATTER_POINT_SIZE,
                invert_y=True,
                show_legend=(i == 0),
            )
        for ax in axes[len(samples):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(
            utils.fig_path(out_dir, f"clustering/spatial_{utils.safe_name(key)}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def selected_report_genes(
    genes_of_interest: Optional[str],
    gene_names,
    corr_df: Optional[pd.DataFrame] = None,
    n_fallback: int = 10,
) -> list[str]:
    gene_set = set(pd.Index(gene_names).astype(str))
    requested = utils.parse_gene_list(genes_of_interest)
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


def scatter_spatial(
    ax,
    coords,
    values,
    title,
    cmap="Spectral_r",
    categorical=False,
    show_legend=True,
):
    if categorical:
        labels = pd.Categorical(values.astype(str))
        codes = labels.codes
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=codes,
            s=utils.SPATIAL_SCATTER_POINT_SIZE,
            cmap="tab20",
        )
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
        if show_legend:
            ax.legend(handles=handles, title="cluster", fontsize=6, loc="best")
    else:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            s=utils.SPATIAL_SCATTER_POINT_SIZE,
            cmap=cmap,
        )
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])


def write_spatial_expression_reports(
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
    samples = sample_labels(rna)
    ge.obs["sample_for_plot"] = samples
    rna.obs["sample_for_plot"] = samples

    selected = [g for g in report_genes if g in gene_to_idx][:12]
    if not selected:
        return

    for modality, matrix, adata, fname, cmap in [
        ("RNA", X_rna, rna, "genes_of_interest/rna_spatial_expression.png", "Spectral_r"),
        ("ATAC GE", X_ge, ge, "genes_of_interest/atac_ge_spatial_expression.png", "Spectral_r"),
    ]:
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
                scatter_spatial(
                    axes[i],
                    adata.obsm["spatial"][mask],
                    matrix[mask, idx],
                    f"{sample} {modality}: {gene}",
                    cmap=cmap,
                )
            for ax in axes[len(selected):]:
                ax.axis("off")
            fig.tight_layout()
            utils.save_fig_suffix(fig, out_dir, fname, sample)
            plt.close(fig)

    for sample in sorted(set(samples)):
        mask = samples == sample
        for gene in selected:
            idx = gene_to_idx[gene]
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            scatter_spatial(
                axes[0],
                rna.obsm["spatial"][mask],
                X_rna[mask, idx],
                f"{sample} RNA: {gene}",
            )
            scatter_spatial(
                axes[1],
                ge.obsm["spatial"][mask],
                X_ge[mask, idx],
                f"{sample} ATAC GE: {gene}",
            )
            fig.tight_layout()
            utils.save_fig_suffix(
                fig,
                out_dir,
                "genes_of_interest/rna_vs_atac_ge_spatial_expression.png",
                f"{sample}_{gene}",
            )
            plt.close(fig)


def write_spatial_cluster_reports(out_dir: str, rna, ge) -> None:
    if "spatial" not in rna.obsm:
        logging.warning("No spatial coordinates found; skipping spatial cluster maps.")
        return
    cluster_key = None
    for key in ["sg_clusters", "sg_leiden_merged", "sg_leiden"]:
        if key in rna.obs.columns:
            cluster_key = key
            break
    if cluster_key is None:
        logging.warning("No cluster labels found; skipping spatial cluster maps.")
        return

    samples = sorted(set(sample_labels(rna)))
    ncols = min(4, len(samples))
    nrows = int(np.ceil(len(samples) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4.5 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()
    sample_labels_array = sample_labels(rna)
    for i, sample in enumerate(samples):
        ax = axes[i]
        mask = sample_labels_array == sample
        scatter_spatial(
            ax,
            rna.obsm["spatial"][mask],
            rna.obs[cluster_key].astype(str).values[mask],
            f"{sample} SpatialGlue clusters",
            categorical=True,
            show_legend=(i == 0),
        )
    for ax in axes[len(samples):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(
        utils.fig_path(out_dir, "spatial_clusters.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_umi_reports(
    out_dir: str,
    rna,
    genes,
    X_counts,
    X_expr,
    report_genes: list[str],
) -> None:
    cluster_key = None
    for key in ["sg_clusters", "sg_leiden_merged", "sg_leiden"]:
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
    clusters = sorted(set(labels), key=utils.cluster_sort_key)
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
    umi_path = utils.table_path(out_dir, "umi_per_cluster_genes_of_interest.csv")
    umi.to_csv(umi_path, index=False)
    logging.info(f"Saved per-cluster UMI table: {umi_path}")

    plot_genes = selected[:12]
    for page_idx, gene in enumerate(plot_genes, start=1):
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
        utils.save_fig_suffix(
            fig,
            out_dir,
            "genes_of_interest/umi_violin_per_cluster.png",
            gene,
        )
        plt.close(fig)


def plot_correlation_overview(corr_df: pd.DataFrame, outpath: str) -> None:
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


def write_cluster_correlation_outputs(
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
    for key in ["sg_clusters", "sg_leiden_merged", "sg_leiden"]:
        if key in rna.obs.columns:
            cluster_key = key
            break
    if cluster_key is None:
        logging.warning("No cluster labels found; skipping per-cluster correlation outputs.")
        return

    gene_names = pd.Index(genes).astype(str)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    requested_genes = utils.parse_gene_list(genes_of_interest)
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
    clusters = sorted(set(cluster_labels), key=utils.cluster_sort_key)
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
    per_cluster_path = utils.table_path(out_dir, "per_cluster_rna_atac_ge.csv")
    per_cluster.to_csv(per_cluster_path, index=False)
    logging.info(f"Saved per-cluster RNA/ATAC GE table: {per_cluster_path}")
