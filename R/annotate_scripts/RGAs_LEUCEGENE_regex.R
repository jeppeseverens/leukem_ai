setwd("~/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/LEUCEGENE")
meta_LEUCEGENE <- read.csv("../LEUCEGENE_AML/Clinical/LEUCEGENE_clinical_meta.csv")

# Clinical meta
clinical_LEUCEGENE <- read.csv("../LEUCEGENE_AML/Clinical/Leucegene_Annotations_new_01032022.tsv", sep = "\t")
rownames(clinical_LEUCEGENE) <- clinical_LEUCEGENE$sample_id
clinical_LEUCEGENE <- clinical_LEUCEGENE[meta_LEUCEGENE$sample_id,]

# Fusions and mutations
LEUCEGENE_2 <- openxlsx::read.xlsx("../LEUCEGENE_AML/advancesadv2020003443-suppl3.xlsx")
colnames(LEUCEGENE_2) <- LEUCEGENE_2[1,]
LEUCEGENE_2 <- LEUCEGENE_2[-1,]
LEUCEGENE_2$sample_ID <- gsub(".+_", "", LEUCEGENE_2[,1])
LEUCEGENE_2 <- LEUCEGENE_2[,-1]
LEUCEGENE_2 <- dplyr::left_join(data.frame("sample_ID" = meta_LEUCEGENE$sample_id), LEUCEGENE_2)
LEUCEGENE_2[is.na(LEUCEGENE_2)] <- "-"
LEUCEGENE_2$MLLs <- gsub("\\.[A-Z][0-9]+-", "-", LEUCEGENE_2$MLLs)
LEUCEGENE_2$MLLs <- gsub("\\..+", "", LEUCEGENE_2$MLLs)
LEUCEGENE_2$MLLs <- gsub("ELL-KMT2A", "KMT2A-ELL", LEUCEGENE_2$MLLs)

# Mutations
LEUCEGENE_mutations <- read.csv("../LEUCEGENE_AML/Genetics/leucegene_website_mutations_2nov2022.csv")
LEUCEGEN_gse_to_id <- read.table("../LEUCEGENE_AML/Clinical/Leucegene_Annotations_new_01032022.tsv", sep = "\t", header = T)
LEUCEGENE_mutations$sample_ID <- (LEUCEGEN_gse_to_id$sample_id[match(LEUCEGENE_mutations$Sample.ID, LEUCEGEN_gse_to_id$Geo_accession_sample)])
LEUCEGENE_mutations$AA_change <- LEUCEGENE_mutations$Protein.impact

LEUCEGENE_mutations <- LEUCEGENE_mutations[LEUCEGENE_mutations$sample_ID %in% meta_LEUCEGENE$sample_id,]
LEUCEGEN_gse_to_id$sample_id <- (LEUCEGEN_gse_to_id$sample_id)
LEUCEGEN_gse_to_id <- LEUCEGEN_gse_to_id[LEUCEGEN_gse_to_id$sample_id %in% meta_LEUCEGENE$sample_id,]
rownames(LEUCEGEN_gse_to_id) <- LEUCEGEN_gse_to_id$sample_id
LEUCEGEN_gse_to_id <- LEUCEGEN_gse_to_id[meta_LEUCEGENE$sample_id,]

LEUCEGENE_mutations$Category[grepl("Frameshift", LEUCEGENE_mutations$Category)] <- "frameshift_variant"
LEUCEGENE_mutations$Category[grepl("Inframe", LEUCEGENE_mutations$Category)] <- "inframe_indel"
LEUCEGENE_mutations$Category[grepl("Missense", LEUCEGENE_mutations$Category)] <- "missense_variant"
LEUCEGENE_mutations$Category[grepl("Termination", LEUCEGENE_mutations$Category)] <- "start_stop_gain_loss_variant"

LEUCEGENE_mutations$sample_id <- LEUCEGENE_mutations$sample_ID

library(dplyr)
library(purrr)

# regex for abnormalities (with fusion precedence)
source("/Users/jsevere2/leukem_ai/R/regex_abnormalities_new.R")

# Prepare data vectors
meta_LEUCEGENE$karyotyping <- gsub("_MLL\\b", " KMT2A",meta_LEUCEGENE$karyotyping)
meta_LEUCEGENE$primary_diagnosis <- gsub("MLLT3-MLL", "KMT2A-MLLT3",meta_LEUCEGENE$primary_diagnosis)

# Separate fusion and karyotype vectors
fusion_vectors <- list(
  MLLs = LEUCEGENE_2$MLLs,
  primary_diagnosis = meta_LEUCEGENE$primary_diagnosis
)
karyotype_vectors <- list(
  ISCN = meta_LEUCEGENE$karyotyping
)

# Initialize results
results <- data.frame(Sample.ID = meta_LEUCEGENE$sample_id)

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
  all_na <- is.na(meta_LEUCEGENE$karyotyping) & 
            is.na(meta_LEUCEGENE$primary_diagnosis) & 
            is.na(LEUCEGENE_2$MLLs)
  results[[abn]] <- ifelse(all_na, NA, as.integer(matched))
}

# Apply perl patterns (complex karyotype, other KMT2A)
for (abn in names(abnormalities_perl)) {
  pattern <- abnormalities_perl[[abn]]
  results[[abn]] <- ifelse(
    is.na(meta_LEUCEGENE$karyotyping) & is.na(meta_LEUCEGENE$primary_diagnosis) & is.na(LEUCEGENE_2$MLLs),
    NA,
    as.integer(
      grepl(pattern, meta_LEUCEGENE$karyotyping, ignore.case = TRUE, perl = TRUE) | 
      grepl(pattern, meta_LEUCEGENE$primary_diagnosis, ignore.case = TRUE, perl = TRUE) |
      grepl(pattern, LEUCEGENE_2$MLLs, ignore.case = TRUE, perl = TRUE)
    )
  )
}

# Override with direct fusion calling data where available
results$`t(15;17)/PML::RARA`[LEUCEGENE_2$PML.RARA != "-"] <- 1
results$`inv(16)/t(16;16)/CBFB::MYH11`[LEUCEGENE_2$MYH11.CBFB != "-"] <- 1
results$`t(8;21)/RUNX1::RUNX1T1`[LEUCEGENE_2$RUNX1.RUNX1T1 != "-"] <- 1
results$`t(6;9)/DEK::NUP214`[LEUCEGENE_2$DEK.NUP214 != "-"] <- 1
results$`t(9;22)/BCR::ABL1`[LEUCEGENE_2$BCR.ABL1 != "-"] <- 1
results$`t(5;11)/NUP98::NSD1`[LEUCEGENE_2$NUP98.NSD1 != "-"] <- 1

# MDS genes
genes_of_interest <- c("TP53", "ASXL1", "BCOR", "EZH2", "RUNX1",
                       "SF3B1", "SRSF2", "STAG2", "U2AF1", "ZRSR2")

mutation_calls <- LEUCEGENE_mutations %>%
  filter(genes %in% genes_of_interest)

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

  # Initialize the column if it doesn't exist
  if (!(col_name %in% names(results))) {
    results[[col_name]] <- ifelse(results$Sample.ID %in% LEUCEGENE_mutations$sample_ID, 0, NA)
  }

  # Get unique sample IDs with mutations in this gene
  mutated_samples <- mutation_calls %>%
    filter(genes == gene) %>%
    pull(sample_id) %>%
    unique()

  # Update column
  results[[col_name]] <- ifelse(results$Sample.ID %in% LEUCEGENE_mutations$sample_ID,
                                pmax(results[[col_name]],
                                     ifelse(results$Sample.ID %in% mutated_samples, 1, 0)),
                                NA)
}

# NPM1
results[["mutated_NPM1"]][LEUCEGENE_2$NPM1 != "-"] <- 1

mutation_calls_NPM1 <- LEUCEGENE_mutations %>%
  filter(genes == "NPM1") %>%
  filter(grepl("frameshift_variant", Category, ignore.case = TRUE))
table(results[["mutated_NPM1"]])
results[["mutated_NPM1"]][results$Sample.ID %in% mutation_calls_NPM1$sample_ID] <- 1
table(results[["mutated_NPM1"]])

# CEBPA
mutation_calls_CEBPA <- LEUCEGENE_mutations %>%
  filter(genes == "CEBPA") %>%
  filter(Category %in% c("inframe_indel")) %>%
  filter(as.numeric(gsub("[A-Z][a-z][a-z]([0-9]+).*", "\\1", Protein.impact)) >= 282)
results[["in_frame_bZIP_CEBPA"]][results$Sample.ID %in% mutation_calls_CEBPA$sample_ID] <- 1

results$Sample.ID <- paste0("X", results$Sample.ID)

table(results$`t(9;11)/MLLT3::KMT2A`)
sum(results[,grepl("MECOM", colnames(results))], na.rm = TRUE)

write.csv(results,"/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/LEUCEGENE/RGAs_LEUCEGENE_regex.csv")
