# Test script to debug file path issues
library(dplyr)

# Test the file loading function
load_outer_cv_data <- function(model_type, method, cv_type = "CV", base_path = "../out/outer_cv") {
  # Construct filename pattern
  filename_pattern <- paste0(
    model_type, "_outer_cv_", cv_type, "_", method, "_*.csv"
  )
  
  # Find matching file
  model_dir <- file.path(base_path, model_type)
  cat("Looking in directory:", model_dir, "\n")
  cat("Pattern:", filename_pattern, "\n")
  
  files <- list.files(model_dir, pattern = filename_pattern, full.names = TRUE)
  cat("Found files:", length(files), "\n")
  if (length(files) > 0) {
    cat("Files:", paste(files, collapse = ", "), "\n")
  }
  
  if (length(files) == 0) {
    stop(paste("No files found matching pattern:", filename_pattern))
  }
  
  # Load the most recent file (assuming timestamp in filename)
  data <- read.csv(files[1])
  
  # For OvO method, create class column if it doesn't exist
  if (method == "OvO" && !"class" %in% colnames(data)) {
    data$class <- paste(data$class_0, data$class_1, sep = "_")
  }
  
  data
}

# Test the metadata loading
load_metadata <- function(base_path = "..") {
  cat("Loading metadata from:", base_path, "\n")
  label_mapping <- read.csv(file.path(base_path, "label_mapping_df.csv"))
  sample_indices <- read.csv(file.path(base_path, "sample_indices.csv"))
  
  list(
    label_mapping = label_mapping,
    sample_indices = sample_indices
  )
}

# Test the functions
cat("Current working directory:", getwd(), "\n")
cat("Testing metadata loading...\n")
metadata <- load_metadata()
cat("Metadata loaded successfully\n")

cat("Testing XGBOOST standard file loading...\n")
xgb_standard <- load_outer_cv_data("XGBOOST", "standard", "CV")
cat("XGBOOST standard loaded successfully\n") 