#!/usr/bin/env Rscript

suppressMessages({
  library(dplyr)
  library(stringr)
  library(argparse)
})

# ------------------------------------------------------------
# CLI arguments
# ------------------------------------------------------------
parser <- ArgumentParser(description = "Prepare flow CSV data for Omnibenchmark cross-validation.")

parser$add_argument("--data_path", type = "character", required = TRUE,
                    help = "Folder containing annotated CSV files.")
parser$add_argument("--output_dir", type = "character", required = TRUE,
                    help = "Directory to write outputs.")
parser$add_argument("--name", type = "character", default = "dataset",
                    help = "Output name prefix.")
parser$add_argument("--seed", type = "integer", default = 0,
                    help = "Seed selecting which file is used as training.")
parser$add_argument("--subset_nrows", type = "integer", default = -1,
                    help = "Optional: number of rows to read per file (for debugging).")

args <- parser$parse_args()

data_path     <- args$data_path
output_dir    <- args$output_dir
name          <- args$name
seed          <- args$seed
subset_nrows  <- args$subset_nrows

if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)

# ------------------------------------------------------------
# List files
# ------------------------------------------------------------
files <- list.files(data_path, pattern = "\\.csv$", full.names = TRUE)

if (length(files) < 2)
  stop("Need at least two CSV files in data_path: one for training, rest for testing.")

files <- sort(files)

# Select training file based on seed (rotation)
train_index <- (seed %% length(files)) + 1

train_file <- files[train_index]
test_files <- files[-train_index]

message("Training file: ", train_file)
message("Test files: ", paste(test_files, collapse = ", "))

# ------------------------------------------------------------
# Helper: read CSV with optional row limit
# ------------------------------------------------------------
read_flow_csv <- function(path) {
  if (subset_nrows > 0) {
    read.csv(path, nrows = subset_nrows)
  } else {
    read.csv(path)
  }
}

# ------------------------------------------------------------
# Load training and test data
# ------------------------------------------------------------
train_df <- read_flow_csv(train_file)
test_list <- lapply(test_files, read_flow_csv)

# ------------------------------------------------------------
# Split matrix (features) + labels
# ------------------------------------------------------------
extract_features_labels <- function(df) {
  if (!"label" %in% colnames(df))
    stop("CSV file has no 'label' column: ", df)
  
  y <- df$label
  X <- df %>% select(-label, -cell_id)
  list(X = X, y = y)
}

train_split <- extract_features_labels(train_df)
train_X <- train_split$X
train_y <- train_split$y

test_X_list <- lapply(test_list, function(df) extract_features_labels(df)$X)
test_y_list <- lapply(test_list, function(df) extract_features_labels(df)$y)

# ------------------------------------------------------------
# Construct output paths (Omnibenchmark style)
# ------------------------------------------------------------
file_train_matrix <- file.path(output_dir, paste0(name, ".train.matrix"))
file_train_labels <- file.path(output_dir, paste0(name, ".train.labels"))
file_test_matrix  <- file.path(output_dir, paste0(name, ".test.matrix"))
file_test_labels  <- file.path(output_dir, paste0(name, ".test.labels"))

# ------------------------------------------------------------
# Combine all test sets into one test matrix
# ------------------------------------------------------------
test_X <- bind_rows(test_X_list)
test_y <- unlist(test_y_list)

# ------------------------------------------------------------
# Write outputs
# ------------------------------------------------------------

write.table(train_X, file_train_matrix,
            sep = ",", row.names = FALSE, col.names = TRUE,
            quote = FALSE)


write.table(train_y, file_train_labels,
            sep = ",", row.names = FALSE, col.names = FALSE,
            quote = FALSE)


write.table(test_X, file_test_matrix,
            sep = ",", row.names = FALSE, col.names = TRUE,
            quote = FALSE)


write.table(test_y, file_test_labels,
            sep = ",", row.names = FALSE, col.names = FALSE,
            quote = FALSE)





message("Done! Outputs written to: ", output_dir)
