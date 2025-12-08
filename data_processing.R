# Load packages 
library(flowCore)

# Specify the path to your FCS file
# fcs_file <- "~/Documents/courses/Benchmarking/repos/ob-flow-datasets/data/Levine_13dim_notransform.fcs"
# fcs_file <- "~/Documents/courses/Benchmarking/repos/ob-flow-datasets/data/Levine_32dim_notransform.fcs"
fcs_file <- "~/Documents/courses/Benchmarking/repos/ob-flow-datasets/data/FlowCAP_ND.fcs"
# fcs_file <- "~/Documents/courses/Benchmarking/repos/ob-flow-datasets/data/FlowCAP_WNV.fcs"

# Look at header
read.FCSheader(fcs_file)

# Load the FCS file
# fcs_data <- read.FCS(fcs_file, alter.names = TRUE, emptyValue = FALSE, truncate_max_range = FALSE)
fcs_data <- read.FCS(fcs_file, transformation = FALSE, emptyValue = TRUE, truncate_max_range = FALSE)

# Get expression matrix from fcs file
exprs <- fcs_data@exprs
head(exprs)

# Extract data mat# Extract data mat# Extract data matrix: All column except the last which is lables
data.matrix <- exprs[,1:ncol(exprs)]

# Get lables
labels <- exprs[,ncol(exprs)]

unique(labels)
unique(sample)
