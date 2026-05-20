import logging

import numpy as np
import pandas as pd
import scanpy as sc

import wf.plotting as pl
import wf.utils as utils


def marker_expression_matrix(adata, modality_name: str):
    for layer in ["log1p", "lognorm", "normalized"]:
        if layer in adata.layers:
            logging.info(
                f"Using {modality_name} layer '{layer}' for cluster marker genes."
            )
            return layer, adata.layers[layer]

    logging.warning(
        f"{modality_name} log-normalized layers not found; "
        f"using {modality_name} .X for cluster marker genes."
    )
    return "X", adata.X


def ensure_scanpy_log1p_metadata(adata) -> None:
    """Scanpy expects uns['log1p']['base'] when uns['log1p'] exists."""
    if "log1p" not in adata.uns:
        return
    if not isinstance(adata.uns["log1p"], dict):
        adata.uns["log1p"] = {"base": None}
        return
    adata.uns["log1p"].setdefault("base", None)


def write_cluster_marker_outputs(
    adata,
    out_dir: str,
    cluster_key: str = "sg_leiden_merged",
    marker_top_n: int = 50,
    modality_name: str = "RNA",
    output_prefix: str = "",
) -> None:
    if marker_top_n < 1:
        raise ValueError("marker_top_n must be at least 1.")
    if cluster_key not in adata.obs.columns:
        logging.warning(
            f"Skipping cluster marker genes because obs['{cluster_key}'] is missing."
        )
        return

    labels = adata.obs[cluster_key].astype(str)
    n_clusters = int(labels.nunique())
    if n_clusters < 2:
        logging.warning(
            "Skipping cluster marker genes because only %d cluster is present.",
            n_clusters,
        )
        return

    genes = pd.Index(adata.var_names).astype(str)
    genes_upper = genes.str.upper()
    keep_genes = ~(
        genes_upper.str.startswith("MT-")
        | genes_upper.str.startswith("RPS")
        | genes_upper.str.startswith("RPL")
        | genes_upper.str.startswith("MTRNR")
    )
    keep_genes = np.asarray(keep_genes, dtype=bool)
    if int(keep_genes.sum()) == 0:
        logging.warning("Skipping cluster marker genes because no genes remain.")
        return

    expression_layer, X = marker_expression_matrix(adata, modality_name)
    marker_adata = adata[:, keep_genes].copy()
    marker_adata.X = X[:, keep_genes].copy()
    marker_adata.obs["cluster"] = labels.loc[marker_adata.obs_names].values
    marker_adata.obs["cluster"] = marker_adata.obs["cluster"].astype(str)
    ensure_scanpy_log1p_metadata(marker_adata)

    clusters = sorted(marker_adata.obs["cluster"].unique(), key=utils.cluster_sort_key)
    logging.info(
        "Ranking %s marker genes for %d clusters across %d genes.",
        modality_name,
        len(clusters),
        marker_adata.n_vars,
    )
    sc.tl.rank_genes_groups(
        marker_adata,
        groupby="cluster",
        method="wilcoxon",
        use_raw=False,
        pts=True,
        key_added="cluster_markers",
    )

    deg_frames = []
    top_frames = []
    top_genes_per_cluster: dict[str, list[str]] = {}
    for cluster in clusters:
        deg_df = sc.get.rank_genes_groups_df(
            marker_adata,
            group=cluster,
            key="cluster_markers",
            pval_cutoff=0.05,
            log2fc_min=0.25,
        )
        deg_df.insert(0, "cluster", cluster)
        deg_frames.append(deg_df)

        top_df = sc.get.rank_genes_groups_df(
            marker_adata,
            group=cluster,
            key="cluster_markers",
            pval_cutoff=0.05,
        ).head(marker_top_n)
        top_df.insert(0, "cluster", cluster)
        top_frames.append(top_df)
        top_genes_per_cluster[cluster] = top_df["names"].astype(str).tolist()

    markers_df = pd.concat(deg_frames, ignore_index=True)
    top_markers_df = pd.concat(top_frames, ignore_index=True)
    prefix = output_prefix
    markers_path = utils.table_path(out_dir, f"{prefix}deg_clusters.csv")
    top_markers_path = utils.table_path(
        out_dir, f"{prefix}deg_clusters_top{marker_top_n}.csv"
    )
    markers_df.to_csv(markers_path, index=False)
    top_markers_df.to_csv(top_markers_path, index=False)
    logging.info(f"Saved cluster marker genes: {markers_path}")
    logging.info(f"Saved top cluster marker genes: {top_markers_path}")

    adata.uns["stagate_cluster_marker_degs"] = markers_df
    adata.uns["stagate_cluster_marker_degs_params"] = {
        "cluster_source": "stagate",
        "cluster_key": cluster_key,
        "modality": modality_name,
        "groupby": cluster_key,
        "method": "wilcoxon",
        "expression_layer": expression_layer,
        "pval_cutoff": 0.05,
        "log2fc_min": 0.25,
        "excluded_prefixes": ["MT-", "RPS", "RPL", "MTRNR"],
        "included_gene_count": int(keep_genes.sum()),
        "excluded_gene_count": int((~keep_genes).sum()),
    }

    if top_markers_df.empty:
        logging.warning("No significant marker genes found; skipping marker heatmap.")
        return

    heatmap_path = utils.fig_path(
        out_dir, f"{prefix}cluster_marker_heatmap_top{marker_top_n}.png"
    )
    marker_heatmap = pl.plot_marker_heatmap(
        marker_adata,
        top_genes_per_cluster,
        heatmap_path,
        marker_top_n=marker_top_n,
    )
    heatmap_table = utils.table_path(
        out_dir, f"{prefix}cluster_marker_heatmap_top{marker_top_n}.csv"
    )
    marker_heatmap.to_csv(heatmap_table)
    adata.uns["stagate_cluster_marker_heatmap"] = marker_heatmap
    adata.uns["stagate_cluster_marker_heatmap_params"] = {
        "cluster_source": "stagate",
        "cluster_key": cluster_key,
        "modality": modality_name,
        "included_gene_count": int(keep_genes.sum()),
        "excluded_gene_count": int((~keep_genes).sum()),
        "excluded_prefixes": ["MT-", "RPS", "RPL", "MTRNR"],
        "expression_layer": expression_layer,
        "pval_cutoff": 0.05,
        "log2fc_min": 0.25,
        "marker_top_n": marker_top_n,
        "values": "column-wise z-score of mean expression, clipped to [-3, 3]",
    }
    logging.info(f"Saved cluster marker heatmap: {heatmap_path}")
