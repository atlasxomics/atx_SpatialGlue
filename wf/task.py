import glob
import logging
import os
import pickle
import random
import subprocess
import sys

import leidenalg
import numpy as np
import scanpy as sc
import snapatac2 as snap
import torch

from scipy import sparse

from SpatialGlue.preprocess import lsi, construct_neighbor_graph
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue

from latch.resources.tasks import custom_task, medium_task
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


@medium_task
def glue_task(
    project_name: str,
    atac_anndata: LatchFile,
    wt_anndata: LatchFile
) -> LatchDir:

    # ------------------ Initialize ---------------------
    logging.info("Starting glue task...")
    out_dir = f"/root/{project_name}"
    os.makedirs(out_dir, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    logging.info("Reading WT AnnData...")
    rna = sc.read_h5ad(wt_anndata.local_path)

    logging.info("Reading ATAC AnnData...")
    atac = sc.read_h5ad(atac_anndata.local_path)

    logging.info(f"n_obs RNA: {rna.n_obs} n_obs ATAC: {atac.n_obs}")
    logging.info(f"\nRNA obs_names examples: {list(map(str, rna.obs_names[:5]))}")
    logging.info(f"ATAC obs_names examples: {list(map(str, atac.obs_names[:5]))}")

    rna.obs_names = utils.ensure_obs_barcodes(rna, "RNA")
    atac.obs_names = utils.ensure_obs_barcodes(atac, "ATAC")di

    rna.var_names = utils.ensure_var_gene_symbols(rna, "RNA", min_fraction=0.5)

    rna.obs_names = utils.clean_ids(rna.obs_names)
    atac.obs_names = utils.clean_ids(atac.obs_names)

    common = rna.obs_names.intersection(atac.obs_names)

    if len(common) == 0:
        raise RuntimeError(
            "Could not find common cells between transcriptome and gene"
            "accessibility data; please ensure the input files are from the"
            "same experiment."
        )

    rna_matched = rna[common, :].copy()
    atac_matched = atac[common, :].copy()
    atac_matched = atac_matched[atac_matched.obs_names.get_indexer(rna_matched.obs_names), :].copy()

    assert (rna_matched.obs_names == atac_matched.obs_names).all()

    # -------------------- ATAC cleanup + LSI --------------------
    logging.info("Cleaning ATAC AnnData...")
    # Ensure CSR for speed (if sparse)
    if hasattr(atac_matched.X, "tocsr"):
        atac_matched.X = atac_matched.X.tocsr()

    # Drop zero-count peaks/tiles
    peak_sums = np.array(atac_matched.X.sum(axis=0)).ravel()

    keep_var = peak_sums > 0
    if (~keep_var).sum():
        print(f"Dropping {(~keep_var).sum()} zero-count peaks")
        atac_matched = atac_matched[:, keep_var].copy()

    logging.info("LSI on ATAC AnnData...")
    # LSI (stores in .obsm["X_lsi"])
    lsi(atac_matched, use_highly_variable=False, n_components=51)
    atac_matched.obsm["feat"] = atac_matched.obsm["X_lsi"].astype("float32")

    # -------------------- RNA features (PCA) with HVG guard ------------------
    logging.info("Adding feat to WT AnnData...")
    if "feat" not in rna_matched.obsm:  # Should we plan for reusing wf outputs

        logging.info("HVG to WT AnnData...")
        # Run PCA on top 50 components to match atac
        hvgs = rna_matched.var["highly_variable"]
        tmp = rna_matched[:, hvgs].copy()
        tmp.X = rna_matched.layers["counts"][:, hvgs]

        logging.info("PCA on WT AnnData...")
        sc.tl.pca(tmp, n_comps=50)
        rna_matched.obsm["feat"] = tmp.obsm["X_pca"].astype("float32")

    # -------------------- Train SpatialGlue --------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info("construct_neighbor_graph...")
    data = construct_neighbor_graph(
        rna_matched, atac_matched, datatype="Spatial-epigenome-transcriptome"
    )

    logging.info("Train_SpatialGlue...")
    model = Train_SpatialGlue(
        data, datatype="Spatial-epigenome-transcriptome", device=device
    )
    logging.info("training...")
    out = model.train()

    rna_result = data["adata_omics1"]
    atac_result = data["adata_omics2"]

    # Collect embeddings/weights on an AnnData for downstream analysis
    logging.info("copy data...")
    adata = rna_result.copy()
    adata.obsm["SpatialGlue"] = out["SpatialGlue"]
    adata.obsm["alpha"] = out.get("alpha")
    adata.obsm["alpha_omics1"] = out.get("alpha_omics1")
    adata.obsm["alpha_omics2"] = out.get("alpha_omics2")

    # -------------------- Neighbors/UMAP/Leiden --------------------
    logging.info("clustering on spatialglue dims...")
    sc.pp.neighbors(adata, use_rep="SpatialGlue", n_neighbors=50)
    sc.tl.umap(adata)

    sc.tl.leiden(adata, resolution=2, key_added="sg_leiden")
    adata.obs["sg_leiden"] = adata.obs["sg_leiden"].astype("category")

    # -------------------- Merge small clusters --------------------
    # Make this optional
    utils.merge_small_clusters(
        adata,
        cluster_key="sg_leiden",
        embed_key="SpatialGlue",
        min_cells=20,
        verbose=True,
    )

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
    rna_result.obs["glue_clusters"] = adata.obs.loc[rna_result.obs_names, "sg_leiden"].values
    atac_result.obs["glue_clusters"] = adata.obs.loc[atac_result.obs_names, "sg_leiden"].values

    logging.info("Creating coverages for new clusters...")
    coverage_dir = f"{out_dir}/glue_cluster_coverages"
    os.makedirs(coverage_dir, exist_ok=True)
    snap.ex.export_coverage(
        atac_result,
        groupby="glue_clusters",
        suffix="_cluster.bw",
        bin_size=10,
        output_format="bigwig",
    )
    bws = glob.glob("*.bw")
    subprocess.run(["mv"] + bws + [coverage_dir])
    logging.info("Finished coverages for new clusters...")

    logging.info("Writing data...")
    atac_result.write(f"{out_dir}/atac_glue.h5ad")
    rna_result.write(f"{out_dir}/rna_glue.h5ad")

    with open(f"{out_dir}/SpatialGlue_model.pickle", "wb") as f:
        pickle.dump(out, f)

    subprocess.run([f"mv /root/figures/* {out_dir}"], shell=True)

    return LatchDir(out_dir, f"latch:///glue_outs/{project_name}")


@custom_task(cpu=8, memory=200, storage_gib=1000)
def corr_task(
    project_name: str,
    results_dir: LatchDir,
    ge_anndata: LatchFile
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
    rna.obs_names = utils.ensure_obs_barcodes(rna, "RNA")
    ge.obs_names = utils.ensure_obs_barcodes(ge, "Gene accessibility")

    # Make sure at least half are gene symbols
    rna.var_names = utils.ensure_var_gene_symbols(rna, "RNA", min_fraction=0.5)
    ge.var_names = utils.ensure_var_gene_symbols(ge, "GE", min_fraction=0.5)

    # Get Spearman correlation table -----------------------------------------
    rna.obs_names = utils.clean_ids(rna.obs_names)
    ge.obs_names = utils.clean_ids(ge.obs_names)

    rna.var_names_make_unique()
    ge.var_names_make_unique()

    # Reduce both to common genes/cells and align
    rna_sub, ge_sub = corr.synch_adata(rna, ge)

    genes = rna_sub.var_names

    logging.info("Ensuring dense matrix...")
    # Ensure both matrices to dense
    X_rna = utils.to_dense(rna_sub.X).astype(np.float32)  # Need to select log1p or make if not available
    X_ge = utils.to_dense(ge_sub.X).astype(np.float32)

    logging.info("Transforming gene accessibility to log1p...")
    X_ge_norm = corr.log_norm(X_ge, 1e4)  # This should only log

    logging.info("Computing correlations...")
    res = corr.get_corr_df(X_rna, X_ge_norm, genes)

    res_path = os.path.join(out_dir, "atac-ge_vs_rna_spearman.csv")
    res.to_csv(res_path, index=False)
    logging.info(f"Saved Spearman results: {res_path}")

    # Gene stats ------------------------------------------------------------
    # Ensure we use 'counts'
    X_rna_counts, rna_source = gs.get_rna_counts_matrix(rna_sub)
    rna_stats = gs.compute_gene_stats_matrix(
        X_rna_counts, genes, prefix="rna_umi", include_minmax_nonzero=True
    )

    # Pre-log1p, but still normalized from ArchR
    X_ge_raw = ge_sub.X
    ge_raw_stats = gs.compute_gene_stats_matrix(
        X_ge_raw, genes, prefix="ge_raw", include_minmax_nonzero=True
    )

    ge_norm_stats = gs.compute_gene_stats_matrix(
        sparse.csr_matrix(X_ge_norm),
        genes,
        prefix="ge_norm",
        include_minmax_nonzero=False
    )

    # Merge all stats + your correlation results
    stats = (
        rna_stats.merge(ge_raw_stats, on="gene", how="inner")
        .merge(ge_norm_stats, on="gene", how="inner")
        .merge(res, on="gene", how="inner") 
    )

    stats_path = os.path.join(out_dir, "gene_stats.csv")
    stats.to_csv(stats_path, index=False)
    logging.info(f"Saved: {stats_path} (RNA counts source: {rna_source})")

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
