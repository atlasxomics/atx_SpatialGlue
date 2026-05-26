args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop(
    "Usage: Rscript archr_coverages.R ",
    "<archr_project_path> <cluster_csv> <out_dir>"
  )
}

archr_project_path <- normalizePath(args[[1]], mustWork = TRUE)
cluster_csv <- normalizePath(args[[2]], mustWork = TRUE)
out_dir <- normalizePath(args[[3]], mustWork = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(ArchR)
})

threads <- suppressWarnings(as.integer(Sys.getenv("ARCHR_THREADS", "8")))
if (is.na(threads) || threads < 1) {
  threads <- 8
}
addArchRThreads(threads = threads)

extract_barcode <- function(x) {
  x <- toupper(as.character(x))
  matches <- regexpr("[ATCG]{16}", x, perl = TRUE)
  out <- rep(NA_character_, length(x))
  has_match <- matches > 0
  out[has_match] <- regmatches(x[has_match], matches[has_match])
  out
}

extract_donor <- function(x) {
  x <- as.character(x)
  matches <- regexpr("D[0-9]{5}", x, perl = TRUE)
  out <- rep(NA_character_, length(x))
  has_match <- matches > 0
  out[has_match] <- regmatches(x[has_match], matches[has_match])
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

copy_bigwigs <- function(source_paths, target_dir) {
  source_paths <- unique(as.character(source_paths))
  source_paths <- source_paths[file.exists(source_paths) & grepl("\\.bw$", source_paths)]
  dir.create(target_dir, recursive = TRUE, showWarnings = FALSE)

  if (length(source_paths) == 0) {
    warning("No .bw files found for ", target_dir)
    return(invisible(character()))
  }

  copied <- file.copy(
    from = source_paths,
    to = file.path(target_dir, basename(source_paths)),
    overwrite = TRUE
  )
  invisible(source_paths[copied])
}

copy_archr_group <- function(group_dir_name, target_subdir) {
  source_dir <- file.path(archr_project_path, "GroupBigWigs", group_dir_name)
  if (!dir.exists(source_dir)) {
    message("ArchR GroupBigWigs directory not found: ", source_dir)
    return(invisible(character()))
  }

  paths <- list.files(source_dir, pattern = "\\.bw$", full.names = TRUE)
  copy_bigwigs(paths, file.path(out_dir, target_subdir))
}

message("Loading ArchRProject: ", archr_project_path)
proj <- tryCatch(
  loadArchRProject(path = archr_project_path, showLogo = FALSE),
  error = function(e) {
    message("Retrying loadArchRProject without showLogo argument: ", conditionMessage(e))
    loadArchRProject(path = archr_project_path)
  }
)

clusters <- read.csv(cluster_csv, stringsAsFactors = FALSE)
required_cols <- c("cell_id", "barcode", "sg_clusters")
missing_cols <- setdiff(required_cols, colnames(clusters))
if (length(missing_cols) > 0) {
  stop("Cluster CSV is missing required columns: ", paste(missing_cols, collapse = ", "))
}

archr_cells <- as.character(proj$cellNames)
assignments <- rep(NA_character_, length(archr_cells))

assignments <- assign_by_key(
  assignments,
  archr_cells,
  as.character(clusters$cell_id),
  as.character(clusters$sg_clusters)
)
assignments <- assign_by_key(
  assignments,
  donor_barcode_key(archr_cells),
  donor_barcode_key(clusters$cell_id),
  as.character(clusters$sg_clusters)
)
assignments <- assign_by_key(
  assignments,
  extract_barcode(archr_cells),
  as.character(clusters$barcode),
  as.character(clusters$sg_clusters)
)

matched <- !is.na(assignments)
message("Matched ", sum(matched), " / ", length(archr_cells), " ArchR cells to sg_clusters")
if (!any(matched)) {
  stop("No ArchR cells matched RNA sg_clusters by exact ID, donor barcode, or barcode.")
}

if (sum(!matched) > 0) {
  warning("Leaving ", sum(!matched), " unmatched ArchR cells out of sg_clusters coverages.")
}

proj <- proj[proj$cellNames %in% archr_cells[matched]]
proj <- addCellColData(
  ArchRProj = proj,
  data = assignments[matched],
  cells = archr_cells[matched],
  name = "sg_clusters",
  force = TRUE
)

message("Creating ArchR group coverages for sg_clusters")
proj <- addGroupCoverages(
  ArchRProj = proj,
  groupBy = "sg_clusters",
  maxCells = 1500,
  force = TRUE
)

message("Writing ArchR group BigWigs for sg_clusters")
bws <- getGroupBW(
  ArchRProj = proj,
  groupBy = "sg_clusters",
  normMethod = "ReadsInTSS",
  tileSize = 100,
  maxCells = 1000,
  ceiling = 4,
  verbose = TRUE,
  threads = threads,
  logFile = createLogFile("getGroupBW_sg_clusters")
)

bw_paths <- as.character(unlist(bws))
if (length(bw_paths) == 0 || !any(file.exists(bw_paths))) {
  bw_paths <- list.files(
    file.path(archr_project_path, "GroupBigWigs"),
    pattern = "\\.bw$",
    recursive = TRUE,
    full.names = TRUE
  )
  bw_paths <- bw_paths[grepl("sg_clusters", bw_paths)]
}

copy_bigwigs(bw_paths, file.path(out_dir, "glue_cluster_coverages"))
copy_archr_group("Clusters", file.path("atac_cluster_coverages", "Clusters"))
copy_archr_group("Sample", "sample_coverages")

message("ArchR coverage export complete.")
