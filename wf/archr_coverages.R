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

load_bsgenome_object <- function(genome_pkg) {
  if (!grepl("^BSgenome\\.", genome_pkg)) {
    return(invisible(FALSE))
  }

  if (!requireNamespace(genome_pkg, quietly = TRUE)) {
    warning(
      "BSgenome package is not installed: ",
      genome_pkg,
      ". Continuing without explicit BSgenome registration."
    )
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
  "BSgenome.Mmusculus.UCSC.mm39",
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

safe_name <- function(x) {
  safe <- gsub("[^A-Za-z0-9_.-]+", "_", as.character(x))
  safe <- gsub("^_+|_+$", "", safe)
  ifelse(nzchar(safe), safe, "labels")
}

valid_group_values <- function(values) {
  values <- as.character(values)
  values <- values[!is.na(values) & nzchar(values)]
  n_unique <- length(unique(values))
  n_unique >= 2 && n_unique < length(values)
}

assign_metadata_column <- function(column_name, archr_cells, clusters) {
  assignments <- rep(NA_character_, length(archr_cells))
  values <- as.character(clusters[[column_name]])
  assignments <- assign_by_key(
    assignments,
    archr_cells,
    as.character(clusters$cell_id),
    values
  )
  assignments <- assign_by_key(
    assignments,
    donor_barcode_key(archr_cells),
    donor_barcode_key(clusters$cell_id),
    values
  )
  assignments <- assign_by_key(
    assignments,
    extract_barcode(archr_cells),
    as.character(clusters$barcode),
    values
  )
  assignments
}

export_archr_group_bigwigs <- function(proj, group_by, target_subdir, required = FALSE) {
  tryCatch(
    {
      message("Creating ArchR group coverages for ", group_by)
      proj <- addGroupCoverages(
        ArchRProj = proj,
        groupBy = group_by,
        maxCells = 1500,
        force = TRUE
      )

      message("Writing ArchR group BigWigs for ", group_by)
      bws <- getGroupBW(
        ArchRProj = proj,
        groupBy = group_by,
        normMethod = "ReadsInTSS",
        tileSize = 100,
        maxCells = 1000,
        ceiling = 4,
        verbose = TRUE,
        threads = threads,
        logFile = createLogFile(paste0("getGroupBW_", safe_name(group_by)))
      )

      bw_paths <- as.character(unlist(bws))
      if (length(bw_paths) == 0 || !any(file.exists(bw_paths))) {
        bw_paths <- list.files(
          file.path(archr_project_path, "GroupBigWigs"),
          pattern = "\\.bw$",
          recursive = TRUE,
          full.names = TRUE
        )
        bw_paths <- bw_paths[grepl(group_by, bw_paths, fixed = TRUE)]
      }

      copy_bigwigs(bw_paths, file.path(out_dir, target_subdir))
      proj
    },
    error = function(e) {
      msg <- paste0("Unable to export ArchR coverages for ", group_by, ": ", conditionMessage(e))
      if (isTRUE(required)) {
        stop(msg)
      }
      warning(msg)
      proj
    }
  )
}

candidate_archr_cluster_columns <- function(proj) {
  cell_data <- as.data.frame(getCellColData(proj))
  cols <- colnames(cell_data)
  cols <- cols[grepl("cluster|leiden|louvain", tolower(cols))]
  cols <- cols[!cols %in% c("sg_clusters")]
  cols <- cols[!grepl("^rna_", cols)]
  valid <- vapply(cols, function(col) valid_group_values(cell_data[[col]]), logical(1))
  cols[valid]
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
  } else if (identical(genome, "mm39")) {
    genome_pkg <- "BSgenome.Mmusculus.UCSC.mm39"
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

clusters <- read.csv(cluster_csv, stringsAsFactors = FALSE)
required_cols <- c("cell_id", "barcode", "sg_clusters")
missing_cols <- setdiff(required_cols, colnames(clusters))
if (length(missing_cols) > 0) {
  stop("Cluster CSV is missing required columns: ", paste(missing_cols, collapse = ", "))
}

archr_cells <- as.character(proj$cellNames)
assignments <- assign_metadata_column("sg_clusters", archr_cells, clusters)

matched <- !is.na(assignments)
message("Matched ", sum(matched), " / ", length(archr_cells), " ArchR cells to sg_clusters")
if (!any(matched)) {
  stop("No ArchR cells matched RNA sg_clusters by exact ID, donor barcode, or barcode.")
}

if (sum(!matched) > 0) {
  warning("Leaving ", sum(!matched), " unmatched ArchR cells out of sg_clusters coverages.")
}

proj <- proj[proj$cellNames %in% archr_cells[matched]]
metadata_cols <- setdiff(colnames(clusters), c("cell_id", "barcode"))
valid_metadata_cols <- character()
for (col in metadata_cols) {
  col_assignments <- assign_metadata_column(col, archr_cells, clusters)
  col_values <- col_assignments[matched]
  if (!identical(col, "sg_clusters") && !valid_group_values(col_values)) {
    message("Skipping ArchR metadata group ", col, ": not a useful grouping.")
    next
  }
  proj <- addCellColData(
    ArchRProj = proj,
    data = col_values,
    cells = archr_cells[matched],
    name = col,
    force = TRUE
  )
  valid_metadata_cols <- c(valid_metadata_cols, col)
}

metadata_targets <- list(
  sg_clusters = "glue_cluster_coverages",
  sample = "sample_coverages",
  condition = "condition_coverages"
)
for (col in valid_metadata_cols) {
  if (col %in% names(metadata_targets)) {
    target_subdir <- metadata_targets[[col]]
  } else if (grepl("^rna_", col)) {
    target_subdir <- file.path("rna_cluster_coverages", sub("^rna_", "", col))
  } else {
    target_subdir <- file.path("metadata_coverages", safe_name(col))
  }
  proj <- export_archr_group_bigwigs(
    proj,
    group_by = col,
    target_subdir = target_subdir,
    required = identical(col, "sg_clusters")
  )
}

atac_cluster_cols <- candidate_archr_cluster_columns(proj)
if (length(atac_cluster_cols) > 0) {
  for (col in atac_cluster_cols) {
    proj <- export_archr_group_bigwigs(
      proj,
      group_by = col,
      target_subdir = file.path("atac_cluster_coverages", safe_name(col)),
      required = FALSE
    )
  }
} else {
  message("No ArchR cluster-like colData columns found for ATAC cluster coverages.")
}

if (!"sample" %in% valid_metadata_cols) {
  copy_archr_group("Sample", "sample_coverages")
}
if (!"Clusters" %in% atac_cluster_cols) {
  copy_archr_group("Clusters", file.path("atac_cluster_coverages", "Clusters"))
}

message("ArchR coverage export complete.")
