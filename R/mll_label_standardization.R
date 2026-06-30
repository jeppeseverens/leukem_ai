# Shared MLL truth / prediction label harmonization (MLL_results.Rmd, run_mll_calibration_comparison.R).

MLL_ICC_MAPPING <- c(
  "AML-MR_cyto" = "AML with MDS-related cytogenetic abnormalities",
  "AML-MR_mut" = "AML with MDS-related gene mutations",
  "AML-CEBPA" = "AML with in frame bZIP CEBPA",
  "AML-CBFB::MYH11" = "AML with inv(16)/t(16;16)/CBFB::MYH11",
  "AML-NPM1" = "AML with mutated NPM1",
  "AML-TP53" = "AML with mutated TP53",
  "AML-DEK::NUP214" = "AML with t(6;9)/DEK::NUP214",
  "AML-RUNX1::RUNX1T1" = "AML with t(8;21)/RUNX1::RUNX1T1",
  "AML-KMT2A::MLLT3" = "AML with t(9;11)/KMT2A::MLLT3",
  "APL-PML::RARA" = "APL, t(15;17)/PML::RARA",
  "AML-GATA2::MECOM" = "GATA2;MECOM",
  "AML-otherMECOM" = "MECOM, other"
)

KMT2A_MLLT3_MODEL_LABEL <- "AML.with.t.9.11..KMT2A..MLLT3"

# MLL ICC buckets non-MLLT3 KMT2A fusions (unmerged truth: AML-otherKMT2A).
MLL_OTHER_KMT2A_UNMERGED_LABEL <- "AML.otherKMT2A"
MLL_OTHER_KMT2A_COLLAPSED_LABEL <- "other.KMT2A"

# MLL ICC buckets multiple rare fusions under one truth class.
MLL_RARE_TRANSLOC_MODEL_LABEL <- "AML.other.rare.transloc"
MLL_RARE_TRANSLOC_PRED_PATTERN <- paste(
  "KAT6A", "ETV6", "NUP98", "CBFA2T3", "PICALM", "RBM15", "FUS\\.\\.ERG", "RUNX1\\.\\.CBFA2T3",
  sep = "|"
)

# merged_maxprob shares the collapsed vocabulary of merged_summed (only the within-family
# probability combine rule differs upstream), so both are harmonized identically here.
MLL_MERGED_LABEL_SETS <- c("merged_summed", "merged_maxprob")
is_merged_mll_label_set <- function(label_set_key) label_set_key %in% MLL_MERGED_LABEL_SETS

map_truth_to_canonical <- function(truth_labels, mapping = MLL_ICC_MAPPING) {
  sapply(truth_labels, function(x) {
    if (x %in% names(mapping)) mapping[[x]] else x
  }, USE.NAMES = FALSE)
}

merge_mll_truth_labels <- function(labels) {
  labels <- as.character(labels)
  labels[grepl("MDS|TP53", labels, ignore.case = TRUE)] <- "MDS.r"
  labels[grepl("KMT2A", labels, ignore.case = TRUE) &
           !grepl("MLLT3", labels, ignore.case = TRUE)] <- "other.KMT2A"
  labels[grepl("MECOM", labels, ignore.case = TRUE)] <- "MECOM"
  gsub("_", ".", make.names(labels))
}

harmonize_kmt2a_mllt3_label <- function(labels) {
  labels <- as.character(labels)
  fuse_idx <- grepl("KMT2A", labels, ignore.case = TRUE) &
    grepl("MLLT3", labels, ignore.case = TRUE) &
    grepl("9\\.11|t\\.9\\.11", labels, ignore.case = TRUE)
  labels[fuse_idx] <- KMT2A_MLLT3_MODEL_LABEL
  labels
}

#' Map model non-MLLT3 KMT2A fusion classes to MLL ICC Other KMT2A bucket (predictions only).
harmonize_mll_other_kmt2a_prediction <- function(
    labels, label_set_key = c("merged_summed", "merged_maxprob", "unmerged_maxprob")) {
  label_set_key <- match.arg(label_set_key)
  labels <- as.character(labels)
  other_idx <- grepl("KMT2A", labels, ignore.case = TRUE) &
    !grepl("MLLT3", labels, ignore.case = TRUE)
  bucket <- if (is_merged_mll_label_set(label_set_key)) {
    MLL_OTHER_KMT2A_COLLAPSED_LABEL
  } else {
    MLL_OTHER_KMT2A_UNMERGED_LABEL
  }
  labels[other_idx] <- bucket
  labels
}

#' Map model rare-fusion class names to MLL ICC truth bucket (predictions only).
harmonize_mll_rare_transloc_prediction <- function(labels) {
  labels <- as.character(labels)
  rare_idx <- grepl(MLL_RARE_TRANSLOC_PRED_PATTERN, labels, ignore.case = TRUE)
  labels[rare_idx] <- MLL_RARE_TRANSLOC_MODEL_LABEL
  labels
}

model_label_from_raw <- function(x) {
  gsub("_", ".", make.names(as.character(x)))
}

standardize_mll_truth <- function(truth_raw, label_set_key = c("merged_summed", "merged_maxprob", "unmerged_maxprob")) {
  label_set_key <- match.arg(label_set_key)
  canonical <- map_truth_to_canonical(truth_raw)
  label <- if (is_merged_mll_label_set(label_set_key)) {
    merge_mll_truth_labels(canonical)
  } else {
    model_label_from_raw(canonical)
  }
  harmonize_kmt2a_mllt3_label(label)
}

standardize_mll_prediction <- function(pred_raw, label_set_key = c("merged_summed", "merged_maxprob", "unmerged_maxprob")) {
  label_set_key <- match.arg(label_set_key)
  label <- model_label_from_raw(pred_raw)
  if (is_merged_mll_label_set(label_set_key)) {
    label <- merge_mll_truth_labels(label)
  }
  label <- harmonize_kmt2a_mllt3_label(label)
  label <- harmonize_mll_other_kmt2a_prediction(label, label_set_key)
  harmonize_mll_rare_transloc_prediction(label)
}

mll_to_display <- c(
  MLL_ICC_MAPPING,
  "MDS.r" = "MDS-related",
  "MECOM" = "MECOM rearrangement",
  "other.KMT2A" = "Other KMT2A rearrangements",
  "AML..NOS" = "AML NOS",
  MLL_OTHER_KMT2A_UNMERGED_LABEL = "Other KMT2A",
  MLL_RARE_TRANSLOC_MODEL_LABEL = "AML-other rare transloc"
)
mll_to_display[[KMT2A_MLLT3_MODEL_LABEL]] <- "KMT2A::MLLT3"

map_mll_to_display <- function(x) {
  out <- mll_to_display[as.character(x)]
  out[is.na(out)] <- as.character(x)[is.na(out)]
  unname(out)
}
