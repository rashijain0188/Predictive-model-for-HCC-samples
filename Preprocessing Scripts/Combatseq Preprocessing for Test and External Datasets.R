# =============================================================================
# ComBat-seq Batch Adjustment: Align a New Dataset to Your Training Dataset
# =============================================================================
#
# PURPOSE
# -------
# To correct batch effects on a new dataset using CombatSeq in R.
#
# -----------------------------------------------------------------------------
# INPUT FILE REQUIREMENTS
# -----------------------------------------------------------------------------
# TRAIN dataset (e.g. "train_dataset.csv"):
#   - Rows    = samples
#   - Columns = genes, "sample_type"
#   - First column = sample ID (used as rownames)
#
# NEW dataset (e.g. "new_dataset.csv"):
#   - Rows    = genes   (standard raw-counts / GEO layout)
#   - Columns = samples
#   - First column = gene ID / symbol (used as rownames)
#
# NEW_SAMPLE_TYPE_FILE (optional):
#   - Only needed if you already know the sample_type (e.g. Tumor/Normal)
#     of the new samples and want ComBat-seq to use it as a covariate.
#   - Two columns: "Sample" and "sample_type"
#   - If not provided, correction runs WITHOUT a biological covariate.
#
# =============================================================================

## Install dependencies and run combatseq -----------------------------------------
# install.packages("BiocManager")
# BiocManager::install(c("sva", "edgeR"))
# install.packages(c("readr", "dplyr", "tibble"))

library(readr)
library(dplyr)
library(tibble)
library(edgeR)
library(sva)

TRAIN_FILE           <- "Train Dataset.csv"  
NEW_FILE             <- "new_dataset.csv"     # genes x samples raw counts
NEW_SAMPLE_TYPE_FILE <- NULL                  # optional: "Sample,sample_type" csv
OUTPUT_FILE          <- "new_dataset_adjusted.csv"

TRAIN_ID_COL   <- 1   # column holding sample IDs in TRAIN_FILE
NEW_GENE_ID_COL <- 1  # column holding gene IDs in NEW_FILE

GENE_LIST <- NULL

train_df <- as.data.frame(read_csv(TRAIN_FILE, show_col_types = FALSE))
rownames(train_df) <- train_df[[TRAIN_ID_COL]]
train_df <- train_df[, -TRAIN_ID_COL, drop = FALSE]

if (!"sample_type" %in% colnames(train_df)) {
  stop("TRAIN_FILE must contain a column named 'sample_type'.")
}

train_sample_type <- train_df$sample_type
train_counts <- train_df %>% select(-sample_type)

# Train data is samples x genes -> transpose to genes x samples for ComBat-seq
train_counts_t <- as.data.frame(t(train_counts))

new_df <- as.data.frame(read_csv(NEW_FILE, show_col_types = FALSE))
rownames(new_df) <- new_df[[NEW_GENE_ID_COL]]
new_counts <- new_df[, -NEW_GENE_ID_COL, drop = FALSE]

common_genes <- intersect(rownames(train_counts_t), rownames(new_counts))
cat(length(common_genes), "genes shared between TRAIN and NEW datasets.\n")

if (length(common_genes) == 0) {
  stop("No shared genes found. Check that gene identifiers match between ",
       "TRAIN_FILE column names and NEW_FILE's gene ID column.")
}

train_counts_t <- train_counts_t[common_genes, , drop = FALSE]
new_counts     <- new_counts[common_genes, , drop = FALSE]

train_counts_t <- train_counts_t[order(rownames(train_counts_t)), ]
new_counts     <- new_counts[order(rownames(new_counts)), ]

merged_counts <- cbind(train_counts_t, new_counts)

sample_type_new <- rep(NA, ncol(new_counts))
names(sample_type_new) <- colnames(new_counts)

if (!is.null(NEW_SAMPLE_TYPE_FILE)) {
  st <- read_csv(NEW_SAMPLE_TYPE_FILE, show_col_types = FALSE)
  sample_type_new[st$Sample] <- st$sample_type
}

sample_type_all <- c(train_sample_type, sample_type_new)
batch_all <- c(rep("Train", ncol(train_counts_t)), rep("New", ncol(new_counts)))

sample_metadata <- data.frame(
  Sample      = colnames(merged_counts),
  sample_type = sample_type_all,
  Batch       = batch_all,
  row.names   = colnames(merged_counts)
)

count_matrix <- as.matrix(merged_counts)
storage.mode(count_matrix) <- "integer"  # ComBat-seq expects integer counts

covar_mod <- NULL
if (!any(is.na(sample_metadata$sample_type))) {
  covar_mod <- model.matrix(~ sample_type, data = sample_metadata)
} else {
  message("sample_type is missing for the NEW dataset -> running ComBat-seq ",
          "WITHOUT a biological covariate. Supply NEW_SAMPLE_TYPE_FILE if you ",
          "want covariate-adjusted correction.")
}

adjusted_counts <- ComBat_seq(
  count_matrix,
  batch     = sample_metadata$Batch,
  group     = NULL,
  covar_mod = covar_mod
)

cat("ComBat-seq correction complete. Dimensions:", dim(adjusted_counts), "\n")

new_sample_names <- colnames(new_counts)
adjusted_new <- adjusted_counts[, new_sample_names, drop = FALSE]

if (!is.null(GENE_LIST)) {
  genes_present <- GENE_LIST[GENE_LIST %in% rownames(adjusted_new)]
  missing_genes <- setdiff(GENE_LIST, genes_present)
  if (length(missing_genes) > 0) {
    warning("Genes from GENE_LIST not found in the data (skipped): ",
            paste(missing_genes, collapse = ", "))
  }
  adjusted_new <- adjusted_new[genes_present, , drop = FALSE]
}

adjusted_new_out <- as.data.frame(t(adjusted_new))
adjusted_new_out <- rownames_to_column(adjusted_new_out, "Sample")

write_csv(adjusted_new_out, OUTPUT_FILE)
cat("Saved batch-adjusted new dataset to:", OUTPUT_FILE, "\n")
