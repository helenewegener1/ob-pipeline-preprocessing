# Load packages 
library(flowCore)

# Specify the path to your FCS file
# fcs_file <- "~/Documents/courses/Benchmarking/repos/ob-flow-datasets/data/Levine_13dim_notransform.fcs"
fcs_file <- "~/Documents/courses/Benchmarking/repos/ob-flow-datasets/data/Levine_13dim_notransform.fcs"
# fcs_file <- "~/Documents/courses/Benchmarking/repos/ob-flow-datasets/data/FlowCAP_ND.fcs"
# fcs_file <- "~/Documents/courses/Benchmarking/repos/ob-flow-datasets/data/FlowCAP_WNV.fcs"
# fcs_file <- "~/Documents/courses/Benchmarking/repos/ob-flow-datasets/data/covid/export_COVID19 samples 23_04_20_ST3_COVID19_ICU_changedW_049_O ST3 230420_009_Live_cells.fcs"
# fcs_file <- "~/Documents/courses/Benchmarking/repos/ob-flow-datasets/data/covid/export_COVID19 samples 21_04_20_ST3_COVID19_ICU_014_A ST3 210420_073_Live_cells.fcs"
fcs_file <- "/Users/srz223/Documents/courses/Benchmarking/repos/ob-pipeline-data/out/data/data_import/data_import.data"

# Look at header
read.FCSheader(fcs_file)

# Load the FCS file
# fcs_data <- read.FCS(fcs_file, alter.names = TRUE, emptyValue = FALSE, truncate_max_range = FALSE)
fcs_data <- read.FCS(fcs_file, transformation = FALSE, emptyValue = TRUE, truncate_max_range = FALSE)

# Get expression matrix from fcs file
exprs <- fcs_data@exprs
head(exprs)
exprs[, colnames(exprs) == "label"] %>% unique()

# colnames1 <- colnames(exprs)
# colnames2 <- colnames(exprs)

# Extract data mat# Extract data mat# Extract data matrix: All column except the last which is lables
data.matrix <- exprs[,1:ncol(exprs)]

# Get lables
# labels <- exprs[,ncol(exprs)]
# 
# unique(labels)
# unique(sample)


################################# COVID MERGE ##################################

library(dplyr)

# Folder containing the FCS files
data_dir <- "~/Documents/courses/Benchmarking/repos/ob-flow-datasets/data/covid/"   

# List all .fcs files
files <- list.files(data_dir, pattern = "\\.fcs$", full.names = TRUE)

# Extract column names for each file
get_cols <- function(path) {
  ff <- read.FCS(path, transformation = FALSE)
  colnames(exprs(ff))
}

cols_list <- lapply(files, get_cols)
names(cols_list) <- basename(files)

# Check if all sets of column names are identical
all_same <- all(sapply(cols_list, function(x) identical(x, cols_list[[1]])))

all_same

# Function to read one FCS file and add a "sample" column
read_fcs_with_sample <- function(path) {
  message("Reading: ", path)
  ff <- read.FCS(path, transformation = FALSE)
  df <- as.data.frame(exprs(ff))
  
  # sample name = filename without extension
  sample_name <- tools::file_path_sans_ext(basename(path))
  df$sample <- sample_name
  
  return(df)
}

# Read all files and merge
df_list <- lapply(files, read_fcs_with_sample)
merged_df <- bind_rows(df_list)
merged_df$Time <- NULL

# Inspect
dim(merged_df)
head(merged_df)
table(merged_df$sample)

# Convert sample to numeric 
merged_df$sample_id <- as.numeric(factor(merged_df$sample))

# Keep only numeric columns
merged_numeric <- merged_df[, sapply(merged_df, is.numeric)]

# Create flowFrame
ff <- flowFrame(as.matrix(merged_numeric))

# Export
write.FCS(ff, "Documents/courses/Benchmarking/repos/ob-flow-datasets/data/covid_merged_samples.fcs")
