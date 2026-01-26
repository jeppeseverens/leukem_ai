TCGA_meta1 <- read.csv("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/TCGA/nationwidechildrens.org_clinical_patient_laml.txt", sep = "\t")
TCGA_meta1 <- TCGA_meta1[-c(1,2),]

TCGA_meta2 <- readxl::read_xlsx("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/TCGA/TCGA_meta_NEJM2013.xlsx")
TCGA_meta2$`TCGA Patient ID` <- paste0("TCGA-AB-", TCGA_meta2$`TCGA Patient ID`)
TCGA_meta2 <- TCGA_meta2[match(TCGA_meta1$bcr_patient_barcode, TCGA_meta2$`TCGA Patient ID`), ]

mutations <- read.csv("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/TCGA/SupplementalTable06.tsv", sep = "\t")
mutations$Sample.ID <- mutations$TCGA_id

fusions <- read.csv("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/TCGA/SupplementalTable13.tsv", sep = "\t")
fusions <- fusions[fusions$Contig_ID!="no events found",]
fusions <- fusions[!grepl("NA\\(NA\\)", fusions$Fusion.genes), ]
fusions$Fusion.genes <- gsub("\\bMLL\\b", "KMT2A", fusions$Fusion.genes)
fusions$Fusion.genes <- gsub("^([^()]+)\\([^()]+\\)([^()]+)\\([^()]+\\)$", "\\1::\\2", fusions$Fusion.genes)

library(dplyr)
library(purrr)

# regex for abnormalities (with fusion precedence)
source("/Users/jsevere2/leukem_ai/R/regex_abnormalities_new.R")

# Separate fusion and karyotype vectors
fusion_vectors <- list(
  Molecular_Classification = TCGA_meta2$`Molecular Classification`,
  Other_fusions = TCGA_meta2$`Other in -frame fusions`,
  Inferred_rearrangement = TCGA_meta2$`Inferred genomic rearrangement (from RNA-Seq fusion)`
)
karyotype_vectors <- list(
  ISCN = TCGA_meta2$Cytogenetics,
  Cytogenetic_Classification = TCGA_meta2$`Cytogenetic Classification`
)

# Initialize results
results <- data.frame(Sample.ID = TCGA_meta1$bcr_patient_barcode)

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
  all_na <- is.na(TCGA_meta2$Cytogenetics) &
            is.na(TCGA_meta2$`Molecular Classification`) &
            is.na(TCGA_meta2$`Other in -frame fusions`) & 
            is.na(TCGA_meta2$`Inferred genomic rearrangement (from RNA-Seq fusion)`)
  results[[abn]] <- ifelse(all_na, NA, as.integer(matched))
}

# Apply perl patterns (complex karyotype, other KMT2A)
all_vectors <- c(fusion_vectors, karyotype_vectors)
for (abn in names(abnormalities_perl)) {
  pattern <- abnormalities_perl[[abn]]
  
  hits <- sapply(all_vectors, function(v) {
    grepl(pattern, v, ignore.case = TRUE, perl = TRUE)
  })
  
  all_na <- is.na(TCGA_meta2$Cytogenetics) &
            is.na(TCGA_meta2$`Molecular Classification`) &
            is.na(TCGA_meta2$`Other in -frame fusions`) & 
            is.na(TCGA_meta2$`Inferred genomic rearrangement (from RNA-Seq fusion)`)
  
  results[[abn]] <- ifelse(all_na, NA, as.integer(rowSums(hits, na.rm = TRUE) > 0))
}

mapping_NPM1 <- ifelse(grepl("NPMc Positive",TCGA_meta1$molecular_abnormality_results), 1, 
                       ifelse(grepl("NPMc Negative",TCGA_meta1$molecular_abnormality_results), 0, NA))
results$mutated_NPM1 <- mapping_NPM1

# Define the genes of interest for mutation calls
genes_of_interest <- c("TP53", "ASXL1", "BCOR", "EZH2", "RUNX1",
                       "SF3B1", "SRSF2", "STAG2", "U2AF1", "ZRSR2")

mutation_calls <- mutations %>%
  filter(gene_name %in% genes_of_interest)

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
    results[[col_name]] <- ifelse(results$Sample.ID %in% mutations$Sample.ID, 0, NA)
  }

  mutated_samples <- mutation_calls %>%
    filter(gene_name == gene) %>%
    pull(Sample.ID) %>%
    unique()

  results[[col_name]] <- ifelse(results$Sample.ID %in% mutations$Sample.ID,
                                pmax(results[[col_name]],
                                     ifelse(results$Sample.ID %in% mutated_samples, 1, 0)),
                                NA)
}

# Process NPM1 mutations: only frameshift mutations
mutation_calls_NPM1 <- mutations %>%
  filter(gene_name == "NPM1") %>%
  filter(grepl("frame_shift_ins", trv_type, ignore.case = TRUE)) %>%
  filter(as.numeric(gsub("p\\.[A-Za-z]*?(\\d+).*", "\\1", amino_acid_change)) >= 283)

NPM1_oh <- pmax(results[["mutated_NPM1"]],
                ifelse(results$Sample.ID %in% mutation_calls_NPM1$TCGA_id, 1, 0), na.rm = TRUE)
results[["mutated_NPM1"]] <- NPM1_oh

# Process CEBPA mutations: only in-frame mutations affecting bZIP region (>=282)
mutation_calls_CEBPA <- mutations %>%
  filter(gene_name == "CEBPA") %>%
  filter(grepl("in_frame", trv_type, ignore.case = TRUE)) %>%
  filter(as.numeric(sub("p\\.[A-Za-z]*?(\\d+).*", "\\1", amino_acid_change)) >= 282)

CEBPA_oh <- pmax(results[["in_frame_bZIP_CEBPA"]],
                 ifelse(results$Sample.ID %in% mutation_calls_CEBPA$TCGA_id, 1, 0), na.rm = TRUE)
results[["in_frame_bZIP_CEBPA"]] <- CEBPA_oh

# Apply fusion patterns to direct fusion calling data (takes precedence)
for (abn in names(abnormalities_fusion)) {
  pattern <- abnormalities_fusion[[abn]]
  has_fusion <- grepl(pattern, fusions$Fusion.genes, ignore.case = TRUE)
  has_fusion <- fusions$Patient_ID[has_fusion]
  cat(abn, "old:", sum(results[[abn]], na.rm = TRUE), "new:", sum(results$Sample.ID %in% has_fusion, na.rm = TRUE), "\n")
  results[[abn]][results$Sample.ID %in% has_fusion] <- 1
}

# BCR_ABL1
mapping_BCR_ABL1 <- ifelse(grepl("BCR-ABL Positive", TCGA_meta1$molecular_abnormality_results), 1, 0)
results$`t(9;22)/BCR::ABL1` <- pmax(results$`t(9;22)/BCR::ABL1`, mapping_BCR_ABL1, na.rm = TRUE)

results$Case.ID <- TCGA_meta1$bcr_patient_barcode

table(results$`t(9;11)/MLLT3::KMT2A`)
sum(results[,grepl("MECOM", colnames(results))], na.rm = TRUE)

write.csv(results,"/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/TCGA/RGAs_TCGA.csv")
