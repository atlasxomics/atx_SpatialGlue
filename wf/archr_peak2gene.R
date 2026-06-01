args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6) {
  stop(
    "Usage: Rscript archr_peak2gene.R ",
    "<archr_project_path> <rna_counts_mtx> <rna_cells_csv> ",
    "<rna_genes_csv> <out_dir> <genes_of_interest_txt>"
  )
}

archr_project_path <- normalizePath(args[[1]], mustWork = TRUE)
counts_mtx <- normalizePath(args[[2]], mustWork = TRUE)
cells_csv <- normalizePath(args[[3]], mustWork = TRUE)
genes_csv <- normalizePath(args[[4]], mustWork = TRUE)
out_dir <- normalizePath(args[[5]], mustWork = FALSE)
genes_of_interest_txt <- normalizePath(args[[6]], mustWork = FALSE)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
tables_dir <- file.path(out_dir, "tables")
bedpe_dir <- file.path(out_dir, "bedpe")
goi_dir <- file.path(out_dir, "genes_of_interest")
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(bedpe_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(goi_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(ArchR)
  library(Matrix)
  library(SummarizedExperiment)
  library(GenomicRanges)
  library(S4Vectors)
})

load_bsgenome_object <- function(genome_pkg) {
  if (!grepl("^BSgenome\\.", genome_pkg)) {
    return(invisible(FALSE))
  }

  message("Loading BSgenome package: ", genome_pkg)
  suppressPackageStartupMessages(
    library(genome_pkg, character.only = TRUE)
  )

  genome_obj <- tryCatch(
    get(genome_pkg, envir = asNamespace(genome_pkg)),
    error = function(e) {
      get(genome_pkg, inherits = TRUE)
    }
  )
  assign(genome_pkg, genome_obj, envir = .GlobalEnv)
  message("Registered BSgenome object in .GlobalEnv: ", genome_pkg)
  invisible(TRUE)
}

for (genome_pkg in c(
  "BSgenome.Hsapiens.UCSC.hg38",
  "BSgenome.Mmusculus.UCSC.mm10",
  "BSgenome.Rnorvegicus.UCSC.rn6"
)) {
  if (requireNamespace(genome_pkg, quietly = TRUE)) {
    load_bsgenome_object(genome_pkg)
  }
}

threads <- suppressWarnings(as.integer(Sys.getenv("ARCHR_THREADS", "8")))
if (is.na(threads) || threads < 1) {
  threads <- 8
}
addArchRThreads(threads = threads)

extract_barcode <- function(x) {
  x <- toupper(as.character(x))
  matches <- regexpr("[ATCG]{16}", x, perl = TRUE)
  out <- rep(NA_character_, length(x))
  has_match <- !is.na(matches) & matches > 0
  if (any(has_match)) {
    out[has_match] <- substring(
      x[has_match],
      matches[has_match],
      matches[has_match] + attr(matches, "match.length")[has_match] - 1
    )
  }
  out
}

extract_donor <- function(x) {
  x <- as.character(x)
  matches <- regexpr("D[0-9]{5}", x, perl = TRUE)
  out <- rep(NA_character_, length(x))
  has_match <- !is.na(matches) & matches > 0
  if (any(has_match)) {
    out[has_match] <- substring(
      x[has_match],
      matches[has_match],
      matches[has_match] + attr(matches, "match.length")[has_match] - 1
    )
  }
  out
}

donor_barcode_key <- function(x) {
  donor <- extract_donor(x)
  barcode <- extract_barcode(x)
  ifelse(!is.na(donor) & !is.na(barcode), paste0(donor, "#", barcode), NA_character_)
}

assign_by_key <- function(current, target_keys, source_keys, source_values) {
  valid <- !is.na(source_keys) & !duplicated(source_keys) & !duplicated(source_keys, fromLast = TRUE)
  source_keys <- source_keys[valid]
  source_values <- source_values[valid]

  idx <- which(is.na(current) & !is.na(target_keys))
  hit <- match(target_keys[idx], source_keys)
  use <- !is.na(hit)
  current[idx[use]] <- source_values[hit[use]]
  current
}

load_project_genome <- function(proj) {
  genome <- tryCatch(proj@genome, error = function(e) NA_character_)
  if (all(is.na(genome))) {
    genome <- tryCatch(proj@genomeAnnotation$genome, error = function(e) NA_character_)
  }
  genome <- as.character(genome)[1]

  if (is.na(genome) || !nzchar(genome)) {
    message("Could not determine ArchR project genome package; continuing without explicit BSgenome load.")
    return(invisible(FALSE))
  }

  genome_pkg <- genome
  if (identical(genome, "hg38")) {
    genome_pkg <- "BSgenome.Hsapiens.UCSC.hg38"
  } else if (identical(genome, "hg19")) {
    genome_pkg <- "BSgenome.Hsapiens.UCSC.hg19"
  } else if (identical(genome, "mm10")) {
    genome_pkg <- "BSgenome.Mmusculus.UCSC.mm10"
  } else if (identical(genome, "mm9")) {
    genome_pkg <- "BSgenome.Mmusculus.UCSC.mm9"
  }

  if (!grepl("^BSgenome\\.", genome_pkg)) {
    message("ArchR project genome is not a BSgenome package name: ", genome)
    return(invisible(FALSE))
  }

  load_bsgenome_object(genome_pkg)
}

strip_stale_group_coverages <- function(project_path) {
  rds_path <- file.path(project_path, "Save-ArchR-Project.rds")
  if (!file.exists(rds_path)) {
    return(invisible(FALSE))
  }

  proj_rds <- readRDS(rds_path)
  group_coverages <- tryCatch(
    proj_rds@projectMetadata$GroupCoverages,
    error = function(e) NULL
  )
  if (is.null(group_coverages) || length(group_coverages) == 0) {
    return(invisible(FALSE))
  }

  old_output_dir <- tryCatch(
    as.character(proj_rds@projectMetadata$outputDirectory)[1],
    error = function(e) NA_character_
  )
  new_output_dir <- normalizePath(project_path, mustWork = TRUE)
  missing_files <- character()

  for (z in seq_along(group_coverages)) {
    zdata <- group_coverages[[z]]$coverageMetadata
    if (is.null(zdata) || !"File" %in% colnames(zdata)) {
      next
    }
    zfiles <- as.character(zdata$File)
    if (!is.na(old_output_dir) && nzchar(old_output_dir)) {
      zfiles <- gsub(old_output_dir, new_output_dir, zfiles, fixed = TRUE)
    }
    missing_files <- c(missing_files, zfiles[!file.exists(zfiles)])
  }

  if (length(missing_files) == 0) {
    return(invisible(FALSE))
  }

  message(
    "Removing stale ArchR GroupCoverages metadata before loading project; ",
    length(unique(missing_files)),
    " saved coverage files are absent from the downloaded project."
  )
  proj_rds@projectMetadata$GroupCoverages <- S4Vectors::SimpleList()
  saveRDS(proj_rds, rds_path)
  invisible(TRUE)
}

available_reduced_dims <- function(proj) {
  out <- tryCatch(names(proj@reducedDims@listData), error = function(e) character())
  if (length(out) == 0) {
    out <- tryCatch(names(proj@reducedDims), error = function(e) character())
  }
  if (is.null(out)) {
    character()
  } else {
    out
  }
}

choose_reduced_dims <- function(proj) {
  rd_names <- available_reduced_dims(proj)
  for (candidate in c("Spectral", "X_spectral_harmony", "X_spectral", "Harmony", "IterativeLSI")) {
    if (candidate %in% rd_names) {
      return(list(project = proj, name = candidate))
    }
  }

  matrices <- tryCatch(getAvailableMatrices(proj), error = function(e) character())
  if ("PeakMatrix" %in% matrices) {
    use_matrix <- "PeakMatrix"
  } else if ("TileMatrix" %in% matrices) {
    use_matrix <- "TileMatrix"
  } else {
    stop("ArchRProject has no reducedDims and no PeakMatrix/TileMatrix for addIterativeLSI.")
  }

  message("No usable reducedDims found; creating IterativeLSI from ", use_matrix)
  proj <- addIterativeLSI(
    ArchRProj = proj,
    useMatrix = use_matrix,
    name = "IterativeLSI",
    iterations = 2,
    varFeatures = 25000,
    force = TRUE
  )
  list(project = proj, name = "IterativeLSI")
}

gene_names_from_ranges <- function(gene_set) {
  cols <- colnames(mcols(gene_set))
  for (candidate in c("name", "symbol", "gene_name", "geneName")) {
    if (candidate %in% cols) {
      return(as.character(mcols(gene_set)[[candidate]]))
    }
  }
  nm <- names(gene_set)
  if (!is.null(nm)) {
    return(as.character(nm))
  }
  rep(NA_character_, length(gene_set))
}

match_genes <- function(rna_genes, gene_names) {
  hit <- match(rna_genes, gene_names)
  missing <- is.na(hit)
  if (any(missing)) {
    upper_gene_names <- toupper(gene_names)
    unique_upper <- !duplicated(upper_gene_names) & !duplicated(upper_gene_names, fromLast = TRUE)
    upper_hit <- match(toupper(rna_genes[missing]), upper_gene_names[unique_upper])
    map_idx <- which(unique_upper)
    use <- !is.na(upper_hit)
    hit[which(missing)[use]] <- map_idx[upper_hit[use]]
  }
  hit
}

write_bedpe <- function(df, path) {
  bedpe_cols <- c(
    "peak_chr", "peak_start0", "peak_end",
    "gene_chr", "gene_start0", "gene_end",
    "name", "score", "strand1", "strand2"
  )
  if (nrow(df) == 0) {
    write.table(
      data.frame(matrix(ncol = length(bedpe_cols), nrow = 0)),
      file = path,
      quote = FALSE,
      sep = "\t",
      row.names = FALSE,
      col.names = FALSE
    )
    return(invisible(FALSE))
  }

  score <- pmax(0, pmin(1000, round(abs(df$Correlation) * 1000)))
  name <- paste(df$geneName, df$peakName, sep = "|")
  bedpe <- data.frame(
    peak_chr = df$peak_chr,
    peak_start0 = pmax(0, df$peak_start - 1),
    peak_end = df$peak_end,
    gene_chr = df$gene_chr,
    gene_start0 = pmax(0, df$gene_start - 1),
    gene_end = df$gene_end,
    name = name,
    score = score,
    strand1 = ".",
    strand2 = ".",
    stringsAsFactors = FALSE
  )
  write.table(
    bedpe[, bedpe_cols],
    file = path,
    quote = FALSE,
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE
  )
}

message("Loading ArchRProject: ", archr_project_path)
strip_stale_group_coverages(archr_project_path)
proj <- tryCatch(
  loadArchRProject(path = archr_project_path, force = TRUE, showLogo = FALSE),
  error = function(e) {
    message("Retrying loadArchRProject without showLogo argument: ", conditionMessage(e))
    loadArchRProject(path = archr_project_path, force = TRUE)
  }
)
load_project_genome(proj)

message("Reading RNA matrix inputs")
counts <- readMM(counts_mtx)
counts <- as(counts, "dgCMatrix")
cells <- read.csv(cells_csv, stringsAsFactors = FALSE)
genes <- read.csv(genes_csv, stringsAsFactors = FALSE)

if (nrow(counts) != nrow(genes)) {
  stop("RNA count matrix row count does not match rna_genes.csv.")
}
if (ncol(counts) != nrow(cells)) {
  stop("RNA count matrix column count does not match rna_cells.csv.")
}

archr_cells <- as.character(proj$cellNames)
rna_indices <- rep(NA_integer_, length(archr_cells))
rna_indices <- assign_by_key(
  rna_indices,
  archr_cells,
  as.character(cells$cell_id),
  seq_len(nrow(cells))
)
rna_indices <- assign_by_key(
  rna_indices,
  donor_barcode_key(archr_cells),
  donor_barcode_key(cells$cell_id),
  seq_len(nrow(cells))
)
rna_indices <- assign_by_key(
  rna_indices,
  extract_barcode(archr_cells),
  as.character(cells$barcode),
  seq_len(nrow(cells))
)

matched <- !is.na(rna_indices)
message("Matched ", sum(matched), " / ", length(archr_cells), " ArchR cells to RNA expression")
if (!any(matched)) {
  stop("No ArchR cells matched RNA cells by exact ID, donor barcode, or barcode.")
}

proj <- proj[proj$cellNames %in% archr_cells[matched]]
matched_archr_cells <- as.character(proj$cellNames)
matched_rna_indices <- rna_indices[match(matched_archr_cells, archr_cells)]

gene_set <- getGenes(proj)
gene_names <- gene_names_from_ranges(gene_set)
gene_idx <- match_genes(as.character(genes$gene), gene_names)
keep_gene <- !is.na(gene_idx)
message("Matched ", sum(keep_gene), " / ", length(keep_gene), " RNA genes to ArchR gene annotations")
if (sum(keep_gene) < 100) {
  stop("Fewer than 100 RNA genes matched ArchR gene annotations; refusing to run Peak2Gene.")
}

counts <- counts[keep_gene, matched_rna_indices, drop = FALSE]
rownames(counts) <- gene_names[gene_idx[keep_gene]]
colnames(counts) <- matched_archr_cells
seRNA <- SummarizedExperiment(
  assays = list(counts = counts),
  rowRanges = gene_set[gene_idx[keep_gene]]
)

message("Adding RNA GeneExpressionMatrix from rna_glue.h5ad counts")
proj <- addGeneExpressionMatrix(
  input = proj,
  seRNA = seRNA,
  strictMatch = TRUE,
  force = TRUE,
  threads = threads
)

matrices <- getAvailableMatrices(proj)
if (!"PeakMatrix" %in% matrices) {
  message("PeakMatrix not found; adding PeakMatrix from project peak set.")
  proj <- addPeakMatrix(proj, force = TRUE)
}

rd <- choose_reduced_dims(proj)
proj <- rd$project
reduced_dims <- rd$name
message("Using reducedDims for Peak2Gene: ", reduced_dims)

message("Running addPeak2GeneLinks")
proj <- addPeak2GeneLinks(
  ArchRProj = proj,
  reducedDims = reduced_dims,
  useMatrix = "GeneExpressionMatrix",
  cellsToUse = matched_archr_cells,
  threads = max(floor(threads / 2), 1)
)

message("Retrieving Peak2Gene links")
p2g <- getPeak2GeneLinks(
  ArchRProj = proj,
  corCutOff = 0,
  resolution = 1,
  returnLoops = FALSE
)

if (is.null(p2g) || nrow(p2g) == 0) {
  message("No Peak2Gene links passed ArchR cutoffs.")
  empty_links <- data.frame()
  write.csv(empty_links, file.path(tables_dir, "peak_to_gene_links.csv"), row.names = FALSE)
  write_bedpe(empty_links, file.path(bedpe_dir, "peak_to_gene_links.bedpe"))
  summary_df <- data.frame(
    n_archr_cells = length(archr_cells),
    n_matched_cells = length(matched_archr_cells),
    n_rna_genes = nrow(genes),
    n_matched_genes = sum(keep_gene),
    reduced_dims = reduced_dims,
    n_links = 0,
    stringsAsFactors = FALSE
  )
  write.csv(summary_df, file.path(tables_dir, "peak_to_gene_summary.csv"), row.names = FALSE)
  quit(save = "no", status = 0)
}

p2g_df <- as.data.frame(p2g)
peak_set <- metadata(p2g)$peakSet
p2g_gene_set <- metadata(p2g)$geneSet
p2g_gene_names <- gene_names_from_ranges(p2g_gene_set)

p2g_df$geneName <- p2g_gene_names[p2g_df$idxRNA]
p2g_df$peakName <- paste0(
  as.character(seqnames(peak_set))[p2g_df$idxATAC],
  "_",
  start(peak_set)[p2g_df$idxATAC],
  "_",
  end(peak_set)[p2g_df$idxATAC]
)
p2g_df$peak_chr <- as.character(seqnames(peak_set))[p2g_df$idxATAC]
p2g_df$peak_start <- start(peak_set)[p2g_df$idxATAC]
p2g_df$peak_end <- end(peak_set)[p2g_df$idxATAC]
p2g_df$gene_chr <- as.character(seqnames(p2g_gene_set))[p2g_df$idxRNA]
p2g_df$gene_start <- start(p2g_gene_set)[p2g_df$idxRNA]
p2g_df$gene_end <- end(p2g_gene_set)[p2g_df$idxRNA]

p2g_df <- p2g_df[order(p2g_df$FDR, -abs(p2g_df$Correlation)), , drop = FALSE]
write.csv(p2g_df, file.path(tables_dir, "peak_to_gene_links.csv"), row.names = FALSE)
write_bedpe(p2g_df, file.path(bedpe_dir, "peak_to_gene_links.bedpe"))

loops <- tryCatch(
  getPeak2GeneLinks(
    ArchRProj = proj,
    corCutOff = 0,
    resolution = 1,
    returnLoops = TRUE
  ),
  error = function(e) {
    warning("Unable to create Peak2Gene loop GRanges: ", conditionMessage(e))
    NULL
  }
)
if (!is.null(loops)) {
  saveRDS(loops, file.path(out_dir, "peak_to_gene_loops.rds"))
}

goi <- character()
if (file.exists(genes_of_interest_txt)) {
  goi <- readLines(genes_of_interest_txt, warn = FALSE)
  goi <- unique(goi[nzchar(goi)])
}
if (length(goi) > 0) {
  for (gene in goi) {
    safe_gene <- gsub("[^A-Za-z0-9_.-]+", "_", gene)
    sub <- p2g_df[p2g_df$geneName == gene, , drop = FALSE]
    write.csv(
      sub,
      file.path(goi_dir, paste0(safe_gene, "_peak_to_gene_links.csv")),
      row.names = FALSE
    )
    write_bedpe(
      sub,
      file.path(goi_dir, paste0(safe_gene, "_peak_to_gene_links.bedpe"))
    )
  }
}

summary_df <- data.frame(
  n_archr_cells = length(archr_cells),
  n_matched_cells = length(matched_archr_cells),
  n_rna_genes = nrow(genes),
  n_matched_genes = sum(keep_gene),
  reduced_dims = reduced_dims,
  n_links = nrow(p2g_df),
  stringsAsFactors = FALSE
)
write.csv(summary_df, file.path(tables_dir, "peak_to_gene_summary.csv"), row.names = FALSE)

message("Peak2Gene export complete.")
