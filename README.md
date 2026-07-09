# atx_SpatialGlue

SpatialGlue workflow for co-embedding whole-transcriptome RNA and gene accessibility data, clustering the joint embedding, generating RNA/ATAC-GE correlation summaries, and optionally exporting epigenomic coverage tracks and ArchR Peak2Gene links.

## Workflow Output Location

The workflow returns a single output directory:

```text
latch:///glue_outs/{project_name}
```

`project_name` is the value supplied to the workflow. A temporary preprocessing checkpoint is written during the run under `glue_outs/{project_name}/preprocess`, but the finalization task attempts to delete it after successful completion.

## Output Directory Overview

```text
glue_outs/{project_name}/
|-- rna_copro.h5ad
|-- rna_copro_sm.h5ad
|-- atac_gs_copro.h5ad
|-- atac_gs_copro_sm.h5ad
|-- atac_tiles_copro.h5ad
|-- SpatialGlue_model.pickle
|-- coverage_manifest.csv
|-- tables/
|-- figures/
|-- Launch_Plots/
|-- coverages/
`-- peak2gene/
```

Some files and subdirectories are conditional. For example, `atac_tiles_copro.h5ad` is only produced when an epigenomic tile AnnData input is provided, coverage tracks require either ATAC tile data or an ArchRProject, and Peak2Gene links require a Peak2Gene ArchRProject.

## Core AnnData and Model Outputs

| Output | Description |
| --- | --- |
| `rna_copro.h5ad` | Full RNA AnnData object after barcode alignment, SpatialGlue embedding, UMAP, selected/merged Leiden cluster labels, and downstream annotations. Contains `obs["sg_clusters"]`, SpatialGlue embedding data, and cluster metadata used by downstream tasks. |
| `rna_copro_sm.h5ad` | Smaller plotting-oriented RNA AnnData object with dense float16 expression values and selected observation columns. This is optimized for interactive plotting rather than full downstream analysis. |
| `atac_gs_copro.h5ad` | Full gene accessibility AnnData object with matched cells, SpatialGlue-derived cluster labels, embedding metadata, and downstream annotations. |
| `atac_gs_copro_sm.h5ad` | Smaller plotting-oriented gene accessibility AnnData object with dense float16 values and selected observation columns. |
| `atac_tiles_copro.h5ad` | Clustered ATAC tile AnnData object. Produced only when `atac_anndata` is supplied. |
| `SpatialGlue_model.pickle` | Pickled SpatialGlue training output containing the learned embedding and attention weights. This can be supplied to `spatialglue_model_pickle` in a later run to reuse the model output and skip training. |
| `coverage_manifest.csv` | Manifest recording which core H5AD outputs were written and whether ATAC tile data was available for coverage export. |

The main cluster label used by downstream outputs is `sg_clusters`. The workflow also stores raw and merged Leiden labels for each resolution in the sweep, using names such as `sg_leiden_0p4` and `sg_leiden_0p4_merged`.

## Tables

All primary tabular analysis outputs are written under:

```text
glue_outs/{project_name}/tables/
```

| Output | Description |
| --- | --- |
| `spatialglue_cluster_sweep.csv` | Leiden resolution sweep summary. Includes the resolution, raw and merged cluster keys, number of clusters before and after small-cluster merging, Moran's I, `min_cluster_size`, and `n_neighbors`. If `chosen_resolution` is set, this contains the single requested resolution. |
| `atac-ge_vs_rna_spearman.csv` | Spearman correlation table comparing RNA expression and gene accessibility for genes that pass the minimum RNA expression fraction filter. Includes correlation, p-value, Benjamini-Hochberg q-value, mean RNA, mean gene accessibility, and absolute correlation. |
| `atac_rna_spearman_all_genes.csv` | Notebook-friendly version of the filtered correlation table with `spearman_r`, mean UMI, expression fraction, p-value, q-value, and absolute correlation. |
| `gene_stats.csv` | Per-gene RNA UMI and gene accessibility summary statistics, including expression-rate filter metadata and merged correlation results when available. This is still produced if no genes pass the correlation filter. |
| `per_cluster_rna_atac_ge.csv` | Per-cluster mean RNA and mean gene accessibility values for either the requested genes of interest or the top correlated genes. |
| `umi_per_cluster_genes_of_interest.csv` | Per-cluster UMI summary for the requested genes of interest or fallback report genes. Includes total UMI, mean UMI per spot, and percent of spots expressing each gene. |
| `rna_deg_clusters.csv` | RNA cluster marker table from Scanpy Wilcoxon ranking after filtering mitochondrial and ribosomal genes. |
| `rna_deg_clusters_top50.csv` | Top RNA marker genes per SpatialGlue cluster. |
| `rna_cluster_marker_heatmap_top50.csv` | Matrix used for the RNA marker heatmap. Values are column-wise z-scores of mean expression, clipped to `[-3, 3]`. |
| `ge_deg_clusters.csv` | Gene accessibility cluster marker table from Scanpy Wilcoxon ranking. |
| `ge_deg_clusters_top50.csv` | Top gene accessibility marker genes per SpatialGlue cluster. |
| `ge_cluster_marker_heatmap_top50.csv` | Matrix used for the gene accessibility marker heatmap. Values are column-wise z-scores of mean accessibility, clipped to `[-3, 3]`. |
| `svg_rna.csv` | Spatial autocorrelation results for RNA genes, produced when spatial coordinates are available. |
| `svg_ge.csv` | Spatial autocorrelation results for gene accessibility features, produced when spatial coordinates are available. |

Marker, per-cluster, and spatially variable gene outputs can be skipped when the required cluster labels, spatial coordinates, or sufficient genes are unavailable.

## Figures

All static figures are written under:

```text
glue_outs/{project_name}/figures/
```

| Output | Description |
| --- | --- |
| `spatial_sg_clusters.png` | Spatial plot of the final selected SpatialGlue clusters. This is a copy of the final cluster plot from `figures/clustering/` for easier discovery. |
| `spatial_clusters.png` | Spatial cluster report split by sample using the final SpatialGlue cluster labels. |
| `umap.png` | Scanpy UMAP plot colored by `sg_clusters`. |
| `atac_rna_correlation_overview.png` | Correlation QC figure showing abundance versus Spearman correlation and the distribution of correlations. |
| `top_genes_bar.png` | Bar plot of the top correlated genes. |
| `corr_volcano.png` | Correlation volcano plot highlighting genes by correlation strength and q-value. |
| `rna_cluster_marker_heatmap_top50.png` | Heatmap of top RNA marker genes by SpatialGlue cluster. |
| `ge_cluster_marker_heatmap_top50.png` | Heatmap of top gene accessibility marker genes by SpatialGlue cluster. |
| `svg_spatial_rna_{sample}.png` | Spatial expression maps for top RNA spatially variable genes, written once per sample. |
| `svg_spatial_ge_{sample}.png` | Spatial maps for top gene accessibility spatially variable features, written once per sample. |

### Clustering Figures

Resolution-specific clustering figures are written under:

```text
glue_outs/{project_name}/figures/clustering/
```

| Pattern | Description |
| --- | --- |
| `umap_{cluster_key}.png` | UMAP plot colored by each merged cluster key in the resolution sweep and by the final `sg_clusters` label. |
| `spatial_{cluster_key}.png` | Spatial plot colored by each merged cluster key in the resolution sweep and by the final `sg_clusters` label. |

### Genes of Interest Figures

Gene report figures are written under:

```text
glue_outs/{project_name}/figures/genes_of_interest/
```

| Pattern | Description |
| --- | --- |
| `rna_spatial_expression_{sample}.png` | Spatial RNA expression maps for requested genes of interest, or fallback top correlated/report genes. |
| `atac_ge_spatial_expression_{sample}.png` | Spatial gene accessibility maps for the same report genes. |
| `rna_vs_atac_ge_spatial_expression_{sample}_{gene}.png` | Side-by-side RNA and gene accessibility spatial maps for each selected gene and sample. |
| `umi_violin_per_cluster_{gene}.png` | Violin plot of expression by SpatialGlue cluster for each selected report gene. |

## Launch Plots Artifact

```text
glue_outs/{project_name}/Launch_Plots/artifact.json
```

This Latch Plots artifact points the plotting template at the completed workflow output directory and records the selected coverage genome. It is used to launch the interactive plotting experience from the workflow output.

## Coverage Outputs

Coverage tracks are written under:

```text
glue_outs/{project_name}/coverages/
```

Coverage export uses ATAC tile AnnData when `atac_anndata` is provided. If no ATAC tile AnnData is provided, it uses `archr_project` when available. If neither is available, coverage generation is skipped and the workflow writes:

```text
coverages/coverage_skipped.txt
```

When coverage export runs, the output can include:

| Output | Description |
| --- | --- |
| `CoPro_cluster_coverages/*.bw` | BigWig coverage tracks grouped by final SpatialGlue `sg_clusters`. |
| `RNA_cluster_coverages/*.bw` | BigWig coverage tracks grouped by RNA cluster metadata when a suitable RNA cluster column is present. |
| `ATAC_cluster_coverages/*.bw` | BigWig coverage tracks grouped by ATAC cluster metadata when a suitable ATAC/ArchR cluster column is present. |
| `sample_coverages/*.bw` | BigWig coverage tracks grouped by sample when sample metadata is available. |
| `condition_coverages/*.bw` | BigWig coverage tracks grouped by condition when condition metadata is available. |
| `metadata_coverages/{metadata_column}/*.bw` | Additional ArchR-derived coverage groups for useful metadata columns not covered by the standard categories. |
| `archr_sg_clusters.csv` | Metadata table used to transfer SpatialGlue cluster assignments onto ArchR cells. Produced only by the ArchR coverage path. |

The exact BigWig file names are generated by SnapATAC2 or ArchR from the group labels and suffixes.

## Peak2Gene Outputs

Peak2Gene outputs are written under:

```text
glue_outs/{project_name}/peak2gene/
```

If `peak2gene_archr_project` is not provided, or if ArchR Peak2Gene generation fails, the workflow writes:

```text
peak2gene/peak2gene_skipped.txt
```

When Peak2Gene generation succeeds, the output can include:

| Output | Description |
| --- | --- |
| `tables/peak_to_gene_links.csv` | Full ArchR Peak2Gene link table sorted by FDR and absolute correlation. Includes ArchR link statistics plus gene and peak genomic coordinates. |
| `tables/peak_to_gene_summary.csv` | Summary of Peak2Gene matching and output counts, including number of ArchR cells, matched cells, RNA genes, matched genes, reduced dimensions used, and total links. |
| `bedpe/peak_to_gene_links.bedpe` | BEDPE-format representation of the full Peak2Gene link set. |
| `peak_to_gene_loops.rds` | RDS file containing ArchR loop GRanges when ArchR can return loop-formatted Peak2Gene links. |
| `genes_of_interest/{gene}_peak_to_gene_links.csv` | Per-gene subset of Peak2Gene links for each requested gene of interest. |
| `genes_of_interest/{gene}_peak_to_gene_links.bedpe` | BEDPE-format per-gene Peak2Gene links for each requested gene of interest. |

If ArchR runs successfully but no Peak2Gene links pass the configured cutoffs, the workflow still writes empty `peak_to_gene_links.csv`, empty `peak_to_gene_links.bedpe`, and `peak_to_gene_summary.csv` with `n_links = 0`.

## Conditional Output Notes

- Spatial figures and spatially variable gene tables require spatial coordinates in the input AnnData objects.
- Gene-of-interest outputs use the `genes_of_interest` parameter when provided. If it is empty, the workflow falls back to top correlated genes or the first available genes, depending on which stage is running.
- Correlation plots are skipped when no genes pass `min_frac_expressing`, but empty correlation tables and `gene_stats.csv` are still written.
- Coverage outputs require either `atac_anndata` or `archr_project`.
- Peak2Gene outputs require `peak2gene_archr_project` and RNA counts that can be passed to ArchR.
