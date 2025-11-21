import logging
import os
import pickle
import random
import subprocess

import leidenalg
import numpy as np
import scanpy as sc
import torch

from SpatialGlue.preprocess import lsi, construct_neighbor_graph
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue

from latch.resources.tasks import small_gpu_task
from latch.types import LatchDir, LatchFile

import wf.utils as utils


@small_gpu_task(retries=0)
def glue_task(
    project_name: str,
    atac_anndata: LatchFile,
    wt_anndata: LatchFile
) -> LatchDir:

    # ------------------ Initialize ---------------------
    logging.info("Starting Workflow...")
    out_dir = f"/root/{project_name}"
    os.makedirs(out_dir, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    logging.info("Reading AnnData...")
    rna = sc.read_h5ad(wt_anndata.local_path)
    atac = sc.read_h5ad(atac_anndata.local_path)

    print("n_obs RNA:", rna.n_obs, " n_obs ATAC:", atac.n_obs)
    print("\nRNA obs_names examples:", list(map(str, rna.obs_names[:5])))
    print("ATAC obs_names examples:", list(map(str, atac.obs_names[:5])))

    # -------------------- Align by XY --------------------

    rna_matched, atac_matched = utils.align_by_xy_exact(rna, atac)
    logging.info("Matched by exact coords: {rna_matched.n_obs}")

    # -------------------- ATAC cleanup + LSI --------------------
    # Ensure CSR for speed (if sparse)
    if hasattr(atac_matched.X, "tocsr"):
        atac_matched.X = atac_matched.X.tocsr()

    # Drop zero-count peaks/spots
    peak_sums = np.array(atac_matched.X.sum(axis=0)).ravel()

    keep_var = peak_sums > 0
    if (~keep_var).sum():
        print(f"Dropping {(~keep_var).sum()} zero-count peaks")
        atac_matched = atac_matched[:, keep_var].copy()

    # LSI (stores in .obsm["X_lsi"])
    lsi(atac_matched, use_highly_variable=False, n_components=51)
    atac_matched.obsm["feat"] = atac_matched.obsm["X_lsi"].astype("float32")

    # -------------------- RNA features (PCA) with HVG guard ------------------
    if "feat" not in rna_matched.obsm:  # Should we plan for reusing wf outputs

        # Run PCA on top 50 components to match atac
        hvgs = rna_matched.var["highly_variable"]
        tmp = rna_matched[:, hvgs].copy()
        tmp.X = rna_matched.layers["counts"][:, hvgs]

        sc.tl.pca(tmp, n_comps=50)
        rna_matched.obsm["feat"] = tmp.obsm["X_pca"].astype("float32")

    # -------------------- Train SpatialGlue --------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = construct_neighbor_graph(
        rna_matched, atac_matched, datatype="Spatial-epigenome-transcriptome"
    )
    model = Train_SpatialGlue(
        data, datatype="Spatial-epigenome-transcriptome", device=device
    )
    out = model.train()

    rna_result = data["adata_omics1"]
    atac_result = data["adata_omics2"]

    # Collect embeddings/weights on an AnnData for downstream analysis
    adata = rna_result.copy()
    adata.obsm["SpatialGlue"] = out["SpatialGlue"]
    adata.obsm["alpha"] = out.get("alpha")
    adata.obsm["alpha_omics1"] = out.get("alpha_omics1")
    adata.obsm["alpha_omics2"] = out.get("alpha_omics2")

    # -------------------- Neighbors/UMAP/Leiden --------------------
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

    # -------------------- Save data --------------------
    atac_result.write(f"{out_dir}/atac.h5ad")
    rna_result.write(f"{out_dir}/rna.h5ad")

    with open(f"{out_dir}/SpatialGlue_model.pickle", "wb") as f:
        pickle.dump(out, f)

    subprocess.run([f"mv /root/figures/* {out_dir}"], shell=True)

    return LatchDir(out_dir, f"latch:///glue_out/{project_name}")
