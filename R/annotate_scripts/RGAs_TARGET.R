library(readxl)
TARGET_meta <- read.csv("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/TARGET/meta_TARGET_AML.csv")

library(dplyr)
library(purrr)
#https://www.nature.com/articles/s41588-023-01640-3#MOESM4 ???!!!
AAML1031_extra <- read_excel("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/TARGET_Newmaybe/41588_2023_1640_MOESM4_ESM(1).xlsx", sheet = 22, skip = 1)
AAML1031_extra <- AAML1031_extra %>%
  mutate(
    KMT2A_fusions = ifelse(
      `molecular category` == "KMT2Ar",
      paste0("KMT2A::", `Subgroup_fusion (KMT2A, NUP98)`),
      NA_character_
    ),
    NUP98_fusions = ifelse(
      `molecular category` == "NUP98r",
      paste0("NUP98::", `Subgroup_fusion (KMT2A, NUP98)`),
      NA_character_
    )
  )
AAML1031_extra <- AAML1031_extra[match(TARGET_meta$TARGET.USI, AAML1031_extra$TARGET.USI),]

## https://onlinelibrary.wiley.com/doi/10.1002/pbc.30251
pbc.30251 <- read_excel("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/TARGET_Newmaybe/pbc30251-sup-0003-tables1.xlsx")
pbc.30251 <- pbc.30251[!apply(pbc.30251, 1, function(x) all(is.na(x))),]
pbc.30251 <- pbc.30251[match(TARGET_meta$TARGET.USI, paste0("TARGET-20-",pbc.30251$`Unique Sample Identifier`)),]

base_dir <- "/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/TARGET_Newmaybe/mafs/gdc_download_20250210_103453.649248/"

maf_gz_files <- list.files(
  path = base_dir,
  pattern = "\\.maf\\.gz$",
  recursive = TRUE,
  full.names = TRUE
)

# Check if any files were found
if (length(maf_gz_files) == 0) {
  stop("No .maf.gz files were found in the directory!")
}

# Read each gzipped MAF file into a list
maf_list <- lapply(maf_gz_files, function(file) {
  cat("Reading file:", file, "\n")
  read.delim(gzfile(file),
             comment.char = "#",
             header = TRUE,
             stringsAsFactors = FALSE)
})

# Combine all data frames
combined_maf <- do.call(rbind, maf_list)
combined_maf$TARGET.USI <- sub("^((?:[^-]+-){2}[^-]+).*", "\\1", combined_maf$Tumor_Sample_Barcode)

# regex for abnormalities (with fusion precedence)
source("/Users/jsevere2/leukem_ai/R/regex_abnormalities_new.R")

# New data from publication

mcm <- read_excel("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/TARGET_Newmaybe/mmc2.xlsx", sheet = 2)
patients <- TARGET_meta[,c("TARGET.USI", "Protocol")]
patients$`Patient identifier` <- gsub("TARGET-..-","", patients$TARGET.USI)
mcm <- left_join(patients, mcm)

mcm2 <- read_excel("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/TARGET_Newmaybe/41588_2025_2321_MOESM3_ESM.xlsx", sheet = 1)
mcm3 <- read_excel("/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/TARGET_Newmaybe/41588_2025_2321_MOESM3_ESM.xlsx", sheet = 3)
mcm2$`Patient identifier` <- gsub("TARGET-..-","", mcm2$patient_id)
mcm3$`Patient identifier` <- gsub("TARGET-..-","", mcm3$alternate_accession)

mcm2 <- left_join(patients, mcm2)
mcm3 <- left_join(patients, mcm3)

table(mcm$Protocol, is.na(mcm$`Primary aberrations`))
table(mcm2$Protocol, is.na(mcm2$gene_fusion))
table(mcm3$Protocol, is.na(mcm3$gene_fusion))

table(mcm$Protocol, (is.na(mcm$`Primary aberrations`) & is.na(mcm2$gene_fusion) & is.na(mcm3$gene_fusion))) %>% addmargins()

# Separate fusion and karyotype vectors
fusion_vectors <- list(
  Gene_Fusion = TARGET_meta$Gene.Fusion,
  Primary_aberrations = mcm$`Primary aberrations`,
  Secondary_aberrations = mcm$`Secondary aberrations`,
  gene_fusion_mcm2 = mcm2$gene_fusion,
  gene_fusion_mcm3 = mcm3$gene_fusion,
  AAML1031_extra$`molecular category`,
  AAML1031_extra$KMT2A_fusions,
  AAML1031_extra$NUP98_fusions,
  pbc.30251$`2016 WHO Classification`,
  pbc.30251$`Classification for statistics`,
  pbc.30251$`Primary Fusion`
)
karyotype_vectors <- list(
  ISCN = TARGET_meta$ISCN
)

# Initialize results
results <- data.frame(Sample.ID = TARGET_meta$TARGET.USI)

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
  all_na <- Reduce(`&`, lapply(c(fusion_vectors, karyotype_vectors), is.na))
  results[[abn]] <- ifelse(all_na, NA, as.integer(matched))
}

# Apply perl patterns (complex karyotype, other KMT2A)
all_vectors <- c(fusion_vectors, karyotype_vectors)
for (abn in names(abnormalities_perl)) {
  pattern <- abnormalities_perl[[abn]]

  hits <- sapply(all_vectors, function(v) {
    grepl(pattern, v, ignore.case = TRUE, perl = TRUE)
  })

  all_na <- Reduce(`&`, lapply(all_vectors, is.na))
  results[[abn]] <- ifelse(all_na, NA, as.integer(rowSums(hits, na.rm = TRUE) > 0))
}

# Mapping additional columns from TARGET_meta
column_mapping <- list(
  "NPM.mutation"       = "mutated_NPM1",
  "CEBPA.mutation"     = "in_frame_bZIP_CEBPA",
  "t.6.9."            = "t(6;9)/DEK::NUP214",
  "t.8.21."           = "t(8;21)/RUNX1::RUNX1T1",
  "t.3.5..q25.q34."    = "t(3;5)/NPM1::MLF1",
  #"t.6.11..q27.q23."   = "t(6;11)/AFDN::KMT2A",
  #"t.9.11..p22.q23."   = "t(9;11)/MLLT3::KMT2A",
  #"t.10.11..p11.2.q23."= "t(10;11)/MLLT10::KMT2A",
  #"t.11.19..q23.p13.1."= "t(11;19)(q23;p13.1)/KMT2A::ELL",
  "inv.16."           = "inv(16)/t(16;16)/CBFB::MYH11",
  "del5q"             = "del(5q)",
  "del7q"             = "del(7q)",
  "del9q"             = "del(9q)",
  "monosomy.5"        = "monosomy_5",
  "monosomy.7"        = "monosomy_7",
  "trisomy.8"         = "trisomy_8",
  "trisomy.21"        = "trisomy_21"
)

for (col in names(column_mapping)) {
  harmonized_name <- column_mapping[[col]]
  if (col %in% names(TARGET_meta)) {
    results[[harmonized_name]][TARGET_meta[[col]] == "Yes"] <- 1
  } else {
    warning(paste("Column", col, "is not in the TARGET_meta dataset."))
  }
}

# Define the genes of interest for mutation calls
genes_of_interest <- c("TP53", "FLT3",
                       "ASXL1", "BCOR", "EZH2", "RUNX1",
                       "SF3B1", "SRSF2", "STAG2", "U2AF1", "ZRSR2")

mutation_calls <- combined_maf %>%
  filter(Hugo_Symbol %in% genes_of_interest)

for (gene in genes_of_interest) {
  col_name <- switch(gene,
                     "TP53"   = "mutated_TP53",
                     "FLT3"   = "FLT3_ITD",
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
    results[[col_name]] <- ifelse(results$Sample.ID %in% combined_maf$TARGET.USI, 0, NA)
  }

  mutated_samples <- mutation_calls %>%
    filter(Hugo_Symbol == gene) %>%
    pull(TARGET.USI) %>%
    unique()

  results[[col_name]] <- ifelse(results$Sample.ID %in% combined_maf$TARGET.USI,
                                pmax(results[[col_name]],
                                     ifelse(results$Sample.ID %in% mutated_samples, 1, 0)),
                                NA)
}

# Process NPM1 mutations: only frameshift mutations in exon 11/11
mutation_calls_NPM1 <- combined_maf %>%
  filter(Hugo_Symbol == "NPM1") %>%
  filter(grepl("fs", HGVSp_Short, ignore.case = TRUE)) %>%
  filter(Exon_Number == "11/11")

results[["mutated_NPM1"]] <- pmax(results[["mutated_NPM1"]],
                                  ifelse(results$Sample.ID %in% mutation_calls_NPM1$TARGET.USI, 1, 0), na.rm = TRUE)

results[["mutated_NPM1"]][mcm$`Secondary aberrations` == "NPM1 mutation"] <- 1
results[["mutated_NPM1"]][mcm3$NPM1 == 1] <- 1
results[["mutated_NPM1"]][AAML1031_extra$`molecular category` == "NPM1"] <- 1

# Process CEBPA mutations: only in-frame mutations affecting bZIP region (>=282)
mutation_calls_CEBPA <- combined_maf %>%
  filter(Hugo_Symbol == "CEBPA") %>%
  filter(Variant_Classification %in% c("In_Frame_Ins", "In_Frame_Del")) %>%
  filter(as.numeric(gsub("p\\.[A-Z]([0-9]+).*", "\\1", HGVSp_Short)) >= 282)

results[["in_frame_bZIP_CEBPA"]] <- pmax(results[["in_frame_bZIP_CEBPA"]],
                                         ifelse(results$Sample.ID %in% mutation_calls_CEBPA$TARGET.USI, 1, 0), na.rm = TRUE)

table(results[["in_frame_bZIP_CEBPA"]])
results[["in_frame_bZIP_CEBPA"]][mcm$`Primary aberrations` == "CEBPA mutation"] <- 1
results[["in_frame_bZIP_CEBPA"]][mcm3$CEBPA_bZIP_in_frame_indel == 1] <- 1
results[["in_frame_bZIP_CEBPA"]][AAML1031_extra$`molecular category` == "CEBPA"] <- 1

table(results[["in_frame_bZIP_CEBPA"]])

mutation_map <- c(
  RUNX1_mut    = "RUNX1",
  mutated_TP53 = "TP53",
  ASXL1_mut    = "ASXL1",
  BCOR_mut     = "BCOR",
  EZH2_mut     = "EZH2",
  SF3B1_mut    = "SF3B1",
  SRSF2_mut    = "SRSF2",
  STAG2_mut    = "STAG2",
  ZRSR2_mut    = "ZRSR2"
)

# Fill mutation flags
table(results[["mutated_TP53"]])
for (res_col in names(mutation_map)) {
  source_col <- mutation_map[[res_col]]
  results[[res_col]][mcm3[[source_col]] == 1] <- 1
  results[[res_col]][mcm3[[source_col]] == 0] <- 0
}

# Mapping additional columns from TARGET_meta to results
column_mapping <- list(
  "NPM.mutation"       = "mutated_NPM1",
  "CEBPA.mutation"     = "in_frame_bZIP_CEBPA",
  "t.6.9."            = "t(6;9)/DEK::NUP214",
  "t.8.21."           = "t(8;21)/RUNX1::RUNX1T1",
  "t.3.5..q25.q34."    = "t(3;5)/NPM1::MLF1",
  # "t.6.11..q27.q23."   = "t(6;11)/AFDN::KMT2A",
  # "t.9.11..p22.q23."   = "t(9;11)/MLLT3::KMT2A",
  # "t.10.11..p11.2.q23."= "t(10;11)/MLLT10::KMT2A",
  # "t.11.19..q23.p13.1."= "t(11;19)(q23;p13.1)/KMT2A::ELL",
  "inv.16."           = "inv(16)/t(16;16)/CBFB::MYH11",
  "del5q"             = "del(5q)",
  "del7q"             = "del(7q)",
  "del9q"             = "del(9q)",
  "monosomy.5"        = "monosomy_5",
  "monosomy.7"        = "monosomy_7",
  "trisomy.8"         = "trisomy_8",
  "trisomy.21"        = "trisomy_21"
)

#Loop through the mapping and update the results.
#We treat NA as 0 so that if either the regex (old result) or the mapped value equals 1,
#the final value will be 1.
for (col in names(column_mapping)) {
  harmonized_name <- column_mapping[[col]]

  if (col %in% names(TARGET_meta)) {
    results[[harmonized_name]][TARGET_meta[[col]] == "Yes"] <- 1
  } else {
    warning(paste("Column", col, "is not in the TARGET_meta dataset."))
  }
}

sum(results[,unlist(column_mapping)],na.rm = TRUE)
table(results$`t(16;21)/RUNX1::CBFA2T2`)
sum(results[,grepl("MECOM", colnames(results))], na.rm = TRUE)
write.csv(results,"/Users/jsevere2/Library/CloudStorage/OneDrive-UMCUtrecht/AML/data/TARGET_Newmaybe/RGAs_TARGET_new.csv")
