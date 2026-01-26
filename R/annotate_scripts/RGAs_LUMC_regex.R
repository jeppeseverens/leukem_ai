LUMC_meta <- read.csv("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/LUMC/meta_LUMC_AML.csv")

library(dplyr)
library(purrr)
mutations_LUMC <- read.csv("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/LUMC_AML/100AML/Genetics/mutations.csv")
mutations_LUMC$sample_id <- make.names(mutations_LUMC$sample_id)

fusions <- read.csv("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/LUMC_AML/100AML/Genetics/fusion_calling.csv")
fusions$sample_id <- make.names(fusions$sample_id)
fusions <- fusions[fusions$left_gene != "No",]
fusions_vector <- fusions %>%
  mutate(fusion = paste0(left_gene, "::", right_gene)) %>%
  group_by(sample_id) %>%
  summarise(fusions = paste(fusion, sep = ", ", collapse = ", "))
fusions_vector <- data.frame(fusions_vector)
rownames(fusions_vector) <- fusions_vector$sample_id
fusions_vector <- fusions_vector[LUMC_meta$sample_id,]
fusions_vector[is.na(fusions_vector)] <- "none"
fusions_vector <- unlist(fusions_vector$fusions)

prim_diagnoses <- read.csv("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/LUMC_AML/100AML/Genetics/HAMLET_supp_table.csv")
prim_diagnoses$sample_id <- gsub(" ", "", prim_diagnoses$Sample )
prim_diagnoses$sample_id <- make.names(paste0("AML.", prim_diagnoses$sample_id))
prim_diagnoses <- prim_diagnoses[prim_diagnoses$sample_id %in% LUMC_meta$sample_id,]

# regex for abnormalities (with fusion precedence)
source("/Users/jsevere2/leukem_ai/R/regex_abnormalities_new.R")

# Separate fusion and karyotype vectors
fusion_vectors <- list(
  fusions = fusions_vector,
  primary_diagnosis = prim_diagnoses$WHO.diagnosis
)
karyotype_vectors <- list(
  ISCN = LUMC_meta$karyotyping
)

# Initialize results
results <- data.frame(Sample.ID = LUMC_meta$Aliquot.ID)

# First, check which samples have ANY fusion match (for precedence logic)
sample_has_any_fusion <- check_any_fusion_match(fusion_vectors)

# Apply pattern matching with fusion precedence
# Key: if a sample has ANY fusion match, we don't fall back to karyotype for any pattern
for (abn in get_all_abnormality_names()) {
  matched <- match_abnormality_with_precedence(
    abn, fusion_vectors, karyotype_vectors, 
    diagnosis_vectors = fusion_vectors,
    sample_has_any_fusion = sample_has_any_fusion
  )
  # Check if all data is NA
  all_na <- is.na(LUMC_meta$karyotyping) &
            is.na(prim_diagnoses$WHO.diagnosis) &
            is.na(fusions_vector)
  results[[abn]] <- ifelse(all_na, NA, as.integer(matched))
}

# Apply perl patterns (complex karyotype, other KMT2A)
for (abn in names(abnormalities_perl)) {
  pattern <- abnormalities_perl[[abn]]
  results[[abn]] <- ifelse(
    is.na(LUMC_meta$karyotyping) & is.na(prim_diagnoses$WHO.diagnosis),
    NA,
    as.integer(
      grepl(pattern, LUMC_meta$karyotyping, ignore.case = TRUE, perl = TRUE) |
      grepl(pattern, prim_diagnoses$WHO.diagnosis, ignore.case = TRUE, perl = TRUE) |
      grepl(pattern, fusions_vector, ignore.case = TRUE, perl = TRUE)
    )
  )
}

# MDS genes
genes_of_interest <- c("TP53", "ASXL1", "BCOR", "EZH2", "RUNX1",
                       "SF3B1", "SRSF2", "STAG2", "U2AF1", "ZRSR2")

mutation_calls <- mutations_LUMC %>%
  filter(gene_id %in% genes_of_interest)

# For each gene, update (or create) the corresponding one-hot column in results.
for (gene in genes_of_interest) {
  col_name <- switch(gene,
                     "TP53"   = "mutated_TP53",
                     "ASXL1"  = "ASXL1_mut",
                     "BCOR"   = "BCOR_mut",
                     "EZH2"   = "EZH2_mut",
                     "RUNX1"  = "RUNX1_mut",
                     "SF3B1"  = "SF3B1_mut",
                     "SRSF2"  = "SRSF2_mut",
                     "STAG2"  = "STAG2_mut",
                     "U2AF1"  = "U2AF1_mut",
                     "ZRSR2"  = "ZRSR2_mut")

  # Initialize column
  if (!(col_name %in% names(results))) {
    results[[col_name]] <- ifelse(results$Sample.ID %in% mutations_LUMC$sample_id, 0, NA)
  }

  # Get unique sample IDs with mutations
  mutated_samples <- mutation_calls %>%
    filter(gene_id == gene) %>%
    pull(sample_id) %>%
    unique()

  # Update column
  results[[col_name]] <- ifelse(results$Sample.ID %in% mutations_LUMC$sample_id,
                                pmax(results[[col_name]],
                                     ifelse(results$Sample.ID %in% mutated_samples, 1, 0)),
                                NA)
}

# Process NPM1 mutations
mutation_calls_NPM1 <- mutations_LUMC %>%
  filter(gene_id == "NPM1") %>%
  filter(grepl("frameshift_variant", variant, ignore.case = TRUE))
table(results[["mutated_NPM1"]])
results[["mutated_NPM1"]][results$Sample.ID %in% mutation_calls_NPM1$sample_id] <- 1
table(results[["mutated_NPM1"]])

# Process CEBPA inframe mut
results[["in_frame_bZIP_CEBPA"]][results$Sample.ID %in% mutations_LUMC$sample_id[mutations_LUMC$gene_id == "CEBPA_bzip_indel"]] <- 1
table(results$in_frame_bZIP_CEBPA)

table(results$`t(9;11)/MLLT3::KMT2A`)
sum(results[,grepl("MECOM", colnames(results))], na.rm = TRUE)
write.csv(results,"/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/LUMC/RGAs_LUMC_regex.csv")
