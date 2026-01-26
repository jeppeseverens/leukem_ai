BEAT_meta <- readxl::read_xlsx("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/BEAT/beataml_wv1to4_clinical.xlsx")
BEAT_meta <- BEAT_meta[!is.na(BEAT_meta$dbgap_rnaseq_sample),]
BEAT_meta <- BEAT_meta[BEAT_meta$diseaseStageAtSpecimenCollection == "Initial Diagnosis",]
BEAT_meta <- BEAT_meta[grepl("Acute|AML|Therapy", BEAT_meta$specificDxAtInclusion),]

mutations <- read.csv("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/BEAT/beataml_wes_wv1to4_mutations_dbgap.txt", skip = 0, sep = "\t")
mutations$dbgap_sample_id <- gsub("D$", "R", mutations$dbgap_sample_id)
fusions <- read.csv("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/BEAT/Fusions.csv", skip = 3)
fusions$SampleID <- sub("^(.*\\d)$", "\\1R", fusions$SampleID )

library(dplyr)
library(purrr)

# regex for abnormalities (with fusion precedence)
source("/Users/jsevere2/leukem_ai/R/regex_abnormalities_new.R")

# Separate fusion and karyotype vectors
fusion_vectors <- list(
  specificDx = BEAT_meta$specificDxAtInclusion,
  consensusFusions = BEAT_meta$consensusAMLFusions
)
karyotype_vectors <- list(
  ISCN = BEAT_meta$karyotype
)

# Initialize results
results <- data.frame(Sample.ID = BEAT_meta$dbgap_rnaseq_sample)

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
  all_na <- is.na(BEAT_meta$karyotype) & is.na(BEAT_meta$specificDxAtInclusion)
  results[[abn]] <- ifelse(all_na, NA, as.integer(matched))
}

# Apply perl patterns (complex karyotype, other KMT2A)
for (abn in names(abnormalities_perl)) {
  pattern <- abnormalities_perl[[abn]]
  results[[abn]] <- ifelse(
    is.na(BEAT_meta$karyotype) & is.na(BEAT_meta$specificDxAtInclusion),
    NA,
    as.integer(
      grepl(pattern, BEAT_meta$karyotype, ignore.case = TRUE, perl = TRUE) |
      grepl(pattern, BEAT_meta$specificDxAtInclusion, ignore.case = TRUE, perl = TRUE) |
      grepl(pattern, BEAT_meta$consensusAMLFusions, ignore.case = TRUE, perl = TRUE)
    )
  )
}

# NPM1
results$mutated_NPM1 <- NA

# Process NPM1 mutations: only frameshift mutations
mutation_calls_NPM1 <- mutations %>%
  filter(symbol == "NPM1") %>%
  filter(grepl("frameshift_variant", variant_classification, ignore.case = TRUE)) %>%
  filter(!grepl("-", codons))

results[["mutated_NPM1"]] <- pmax(results[["mutated_NPM1"]],
                                  ifelse(results$Sample.ID %in% mutation_calls_NPM1$dbgap_sample_id, 1, 0), na.rm = TRUE)
results$mutated_NPM1[BEAT_meta$specificDxAtInclusion == "AML with mutated NPM1"] <- 1

# CEBPA
results$in_frame_bZIP_CEBPA <- ifelse(BEAT_meta$CEBPA_Biallelic == "bi", 1, ifelse(BEAT_meta$CEBPA_Biallelic != "bi", 0, NA))
# Process CEBPA mutations: only in-frame mutations affecting bZIP region (>=282)
mutation_calls_CEBPA <- mutations %>%
  filter(symbol == "CEBPA") %>%
  filter(variant_classification %in% c("inframe_insertion", "inframe_deletion")) %>%
  filter(as.numeric(gsub("p\\.[A-Z]([0-9]+).*", "\\1", hgvsp_short)) >= 282)

results[["in_frame_bZIP_CEBPA"]] <- pmax(results[["in_frame_bZIP_CEBPA"]],
                                         ifelse(results$Sample.ID %in% mutation_calls_CEBPA$dbgap_sample_id, 1, 0), na.rm = TRUE)
results$in_frame_bZIP_CEBPA[BEAT_meta$CEBPA_Biallelic == "bi"] <- 1
results$in_frame_bZIP_CEBPA[BEAT_meta$specificDxAtInclusion == "AML with mutated CEBPA"] <- 1

# Define the genes of interest for mutation calls
genes_of_interest <- c("TP53", "ASXL1", "BCOR", "EZH2", "RUNX1",
                       "SF3B1", "SRSF2", "STAG2", "U2AF1", "ZRSR2")

mutations$gene <- NULL
mutation_calls <- mutations %>%
  filter(symbol %in% genes_of_interest)

# For each gene, update (or create) the corresponding one-hot column
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

  if (!(col_name %in% names(results))) {
    results[[col_name]] <- ifelse(results$Sample.ID %in% mutations$dbgap_sample_id, 0, NA)
  }

  mutated_samples <- mutation_calls %>%
    filter(symbol == gene) %>%
    pull(dbgap_sample_id) %>%
    unique()

  results[[col_name]] <- ifelse(results$Sample.ID %in% mutations$dbgap_sample_id,
                                pmax(results[[col_name]],
                                     ifelse(results$Sample.ID %in% mutated_samples, 1, 0)),
                                NA)
}

results$RUNX1_mut[!is.na(BEAT_meta$RUNX1)] <- 1
results$ASXL1_mut[!is.na(BEAT_meta$ASXL1)] <- 1
results$mutated_TP53[!is.na(BEAT_meta$TP53)] <- 1

for (gene in c("TP53", "ASXL1", "BCOR", "EZH2", "RUNX1",
               "SF3B1", "SRSF2", "STAG2", "U2AF1", "ZRSR2")){
  if(gene == "TP53"){
    colname <- "mutated_TP53"
  } else{
    colname <- paste0(gene, "_mut")
  }
  results[[colname]][grepl(gene, BEAT_meta$variantSummary)] <- 1
}

# Fusions
remove <- grepl("-", fusions$left_gene) | grepl("-", fusions$right_gene)
fusions <- fusions[!remove,]
fusions <- fusions[fusions$SampleID %in% BEAT_meta$dbgap_rnaseq_sample,]
# Create combined column with both gene orders
fusions$combined <- paste(
  paste(fusions$right_gene, fusions$left_gene, sep = "::"),
  paste(fusions$left_gene, fusions$right_gene, sep = "::"),
  sep = " "
)
results$Case.ID <- BEAT_meta$dbgap_subject_id
write.csv(results,"/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/BEAT/RGAs_BEAT.csv")

table(results$`t(9;11)/MLLT3::KMT2A`)
sum(results[,grepl("MECOM", colnames(results))], na.rm = TRUE)
