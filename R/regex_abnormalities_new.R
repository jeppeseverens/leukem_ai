# Regex patterns for genetic abnormalities
# Split into fusion-only and karyotype-only patterns for precedence handling

MECOM_ALIAS <- "(?:MECOM|EVI1|MDS1)"
FUSION_SEP  <- "(?:[-:_]{1,2}|::|--|_)"

# =============================================================================
# FUSION-ONLY PATTERNS (match gene fusion notation like GENE1::GENE2)
# =============================================================================
abnormalities_fusion <- list(
  # APL
  "t(15;17)/PML::RARA" = "PML\\s*[-:_]{1,2}\\s*RARA|RARA\\s*[-:_]{1,2}\\s*PML",
  "IRF2BP2::RARA" = "IRF2BP2\\s*[-:_]{1,2}\\s*RARA|RARA\\s*[-:_]{1,2}\\s*IRF2BP2",
  "NPM1::RARA" = "NPM1\\s*[-:_]{1,2}\\s*RARA|RARA\\s*[-:_]{1,2}\\s*NPM1",
  "ZBTB16::RARA" = "ZBTB16\\s*[-:_]{1,2}\\s*RARA|RARA\\s*[-:_]{1,2}\\s*ZBTB16",
  "STAT5B::RARA" = "STAT5B\\s*[-:_]{1,2}\\s*RARA|RARA\\s*[-:_]{1,2}\\s*STAT5B",
  "STAT3::RARA" = "STAT3\\s*[-:_]{1,2}\\s*RARA|RARA\\s*[-:_]{1,2}\\s*STAT3",
  "TBL1XR1::RARA" = "TBL1XR1\\s*[-:_]{1,2}\\s*RARA|RARA\\s*[-:_]{1,2}\\s*TBL1XR1",
  "FIP1L1::RARA" = "FIP1L1\\s*[-:_]{1,2}\\s*RARA|RARA\\s*[-:_]{1,2}\\s*FIP1L1",
  "BCOR::RARA" = "BCOR\\s*[-:_]{1,2}\\s*RARA|RARA\\s*[-:_]{1,2}\\s*BCOR",
  
  # Core binding factor
 "t(8;21)/RUNX1::RUNX1T1" = "(?:RUNX1\\s*[-:_]{1,2}\\s*RUNX1T1)|(?:RUNX1T1\\s*[-:_]{1,2}\\s*RUNX1)",
  "inv(16)/t(16;16)/CBFB::MYH11" = "CBFB\\s*[-:_]{1,2}\\s*MYH11|MYH11\\s*[-:_]{1,2}\\s*CBFB",
  
  # KMT2A fusions - STRICT partner matching
  "t(9;11)/MLLT3::KMT2A" = "(?:\\bMLLT3\\b\\s*[-:_]{1,2}\\s*\\bKMT2A\\b)|(?:\\bKMT2A\\b\\s*[-:_]{1,2}\\s*\\bMLLT3\\b)|MLLT3-MLL",
  "t(4;11)/AFF1::KMT2A" = "(?:\\bAFF1\\b\\s*[-:_]{1,2}\\s*\\bKMT2A\\b)|(?:\\bKMT2A\\b\\s*[-:_]{1,2}\\s*\\bAFF1\\b)",
  "t(10;11)/MLLT10::KMT2A" = "(?:\\bMLLT10\\b\\s*[-:_]{1,2}\\s*\\bKMT2A\\b)|(?:\\bKMT2A\\b\\s*[-:_]{1,2}\\s*\\bMLLT10\\b)",
  "t(11;19)(q23;p13.3)/KMT2A::MLLT1" = "(?:\\bKMT2A\\b\\s*[-:_]{1,2}\\s*\\bMLLT1\\b)|(?:\\bMLLT1\\b\\s*[-:_]{1,2}\\s*\\bKMT2A\\b)",
  "t(10;11)(q21.3;q23.3)/TET1::KMT2A" = "(?:\\bTET1\\b\\s*[-:_]{1,2}\\s*\\bKMT2A\\b)|(?:\\bKMT2A\\b\\s*[-:_]{1,2}\\s*\\bTET1\\b)",
  "t(10;11)/PICALM::MLLT10" = "(?:\\bPICALM\\b\\s*[-:_]{1,2}\\s*\\bMLLT10\\b)|(?:\\bMLLT10\\b\\s*[-:_]{1,2}\\s*\\bPICALM\\b)",
  "t(11;19)(q23;p13.1)/KMT2A::ELL" = "(?:\\bKMT2A\\b\\s*[-:_]{1,2}\\s*\\bELL\\b)|(?:\\bELL\\b\\s*[-:_]{1,2}\\s*\\bKMT2A\\b)",
  "t(6;11)(q27;q23)/KMT2A::MLLT4" = "MLLT4\\s*[-:_]{1,2}\\s*KMT2A|KMT2A\\s*[-:_]{1,2}\\s*MLLT4|AFDN\\s*[-:_]{1,2}\\s*KMT2A|KMT2A\\s*[-:_]{1,2}\\s*AFDN",
  "KMT2A::MLLT6" = "MLLT6\\s*[-:_]{1,2}\\s*KMT2A|KMT2A\\s*[-:_]{1,2}\\s*MLLT6",
  "KMT2A::SEPT6" = "SEPT6\\s*[-:_]{1,2}\\s*KMT2A|KMT2A\\s*[-:_]{1,2}\\s*SEPT6",
  "KMT2A::MLLT11" = "(?i)(?:KMT2A\\s*[-:_]{1,2}\\s*MLLT11)|(?:MLLT11\\s*[-:_]{1,2}\\s*KMT2A)",
  "KMT2A::LASP1" = "(?i)(?:KMT2A\\s*[-:_]{1,2}\\s*LASP1)|(?:LASP1\\s*[-:_]{1,2}\\s*KMT2A)",
  "KMT2A::MYO1F" = "(?i)(?:\\bKMT2A\\b\\s*[-:_]{1,2}\\s*\\bMYO1F\\b)|(?:\\bMYO1F\\b\\s*[-:_]{1,2}\\s*\\bKMT2A\\b)",
  "KMT2A::SEPT9" = "(?i)(?:KMT2A\\s*[-:_]{1,2}\\s*SEPT9)|(?:SEPT9\\s*[-:_]{1,2}\\s*KMT2A)",
  "KMT2A::FNBP1" = "FNBP1\\s*[-:_]{1,2}\\s*KMT2A|KMT2A\\s*[-:_]{1,2}\\s*FNBP1",
  "KMT2A::EPS15" = "EPS15\\s*[-:_]{1,2}\\s*KMT2A|KMT2A\\s*[-:_]{1,2}\\s*EPS15",
  
  # DEK::NUP214
  "t(6;9)/DEK::NUP214" = "(?:DEK\\s*[-:_]{1,2}\\s*NUP214)|(?:NUP214\\s*[-:_]{1,2}\\s*DEK)",
  
  # MECOM fusions
  "inv(3)/t(3;3)/GATA2;MECOM" = paste0(
    "GATA2\\s*", FUSION_SEP, "\\s*", MECOM_ALIAS,
    "|", MECOM_ALIAS, "\\s*", FUSION_SEP, "\\s*GATA2",
    "|RPN1\\s*", FUSION_SEP, "\\s*", MECOM_ALIAS,
    "|", MECOM_ALIAS, "\\s*", FUSION_SEP, "\\s*RPN1"
  ),
  "t(3;8)/MYC,MECOM" = paste0(
    "MYC\\s*", FUSION_SEP, "\\s*", MECOM_ALIAS,
    "|", MECOM_ALIAS, "\\s*", FUSION_SEP, "\\s*MYC"
  ),
  "t(3;12)/ETV6::MECOM" = paste0(
    "ETV6\\s*", FUSION_SEP, "\\s*", MECOM_ALIAS,
    "|", MECOM_ALIAS, "\\s*", FUSION_SEP, "\\s*ETV6"
  ),
  "t(3;21)/MECOM::RUNX1" = paste0(
    "RUNX1\\s*", FUSION_SEP, "\\s*", MECOM_ALIAS,
    "|", MECOM_ALIAS, "\\s*", FUSION_SEP, "\\s*RUNX1"
  ),
  "MECOM_fusion_any_partner" = paste0(
    "(?:^|\\b)(",
    MECOM_ALIAS, "\\s*", FUSION_SEP, "\\s*[A-Z0-9]+",
    "|[A-Z0-9]+\\s*", FUSION_SEP, "\\s*", MECOM_ALIAS,
    ")"
  ),
  
  # BCR::ABL1
  "t(9;22)/BCR::ABL1" = "BCR\\s*[-:_]{1,2}\\s*ABL1|ABL1\\s*[-:_]{1,2}\\s*BCR",
  
  # Rare fusions
  "t(1;3)/PRDM16::RPN1" = "(?:PRDM16\\s*[-:_]{1,2}\\s*RPN1)|(?:RPN1\\s*[-:_]{1,2}\\s*PRDM16)",
  "t(3;5)/NPM1::MLF1" = "(?:NPM1\\s*[-:_]{1,2}\\s*MLF1)|(?:MLF1\\s*[-:_]{1,2}\\s*NPM1)",
  "t(8;16)/KAT6A::CREBBP" = "(?:KAT6A\\s*[-:_]{1,2}\\s*CREBBP)|(?:CREBBP\\s*[-:_]{1,2}\\s*KAT6A)",
  "t(1;22)/RBM15::MRTF1" = "(?:RBM15\\s*[-:_]{1,2}\\s*(?:MRTF1|MKL1))|(?:(?:MRTF1|MKL1)\\s*[-:_]{1,2}\\s*RBM15)",
  "t(5;11)/NUP98::NSD1" = "(?:NUP98\\s*[-:_]{1,2}\\s*NSD1)|(?:NSD1\\s*[-:_]{1,2}\\s*NUP98)",
  "t(11;12)/NUP98::KDM5A" = "(?:NUP98\\s*[-:_]{1,2}\\s*KDM5A)|(?:KDM5A\\s*[-:_]{1,2}\\s*NUP98)",
  "NUP98::HOXA9" = "(?:NUP98\\s*[-:_]{1,2}\\s*HOXA9)|(?:HOXA9\\s*[-:_]{1,2}\\s*NUP98)",
  "NUP98::HOXD13" = "(?:NUP98\\s*[-:_]{1,2}\\s*HOXD13)|(?:HOXD13\\s*[-:_]{1,2}\\s*NUP98)",
  "NUP98::DDX10" = "(?:NUP98\\s*[-:_]{1,2}\\s*DDX10)|(?:DDX10\\s*[-:_]{1,2}\\s*NUP98)",
  "NUP98::LEDGF" = "(?:NUP98\\s*[-:_]{1,2}\\s*LEDGF)|(?:LEDGF\\s*[-:_]{1,2}\\s*NUP98)",
  "NUP98::RARG" = "(?:NUP98\\s*[-:_]{1,2}\\s*RARG)|(?:RARG\\s*[-:_]{1,2}\\s*NUP98)",
  "NUP98::HOXC11" = "(?:NUP98\\s*[-:_]{1,2}\\s*HOXC11)|(?:HOXC11\\s*[-:_]{1,2}\\s*NUP98)",
  "NUP98::HOXC13" = "(?:NUP98\\s*[-:_]{1,2}\\s*HOXC13)|(?:HOXC13\\s*[-:_]{1,2}\\s*NUP98)",
  "NUP98::PHF23" = "(?:NUP98\\s*[-:_]{1,2}\\s*PHF23)|(?:PHF23\\s*[-:_]{1,2}\\s*NUP98)",
  "NUP98_all" = "(?i)NUP98\\s*[-:_]{1,2}\\s*[A-Z0-9_]+|[A-Z0-9_]+\\s*[-:_]{1,2}\\s*NUP98",
  "t(7;12)/ETV6::MNX1" = "(?:ETV6\\s*[-:_]{1,2}\\s*MNX1)|(?:MNX1\\s*[-:_]{1,2}\\s*ETV6)",
  "t(16;21)/FUS::ERG" = "(?:FUS\\s*[-:_]{1,2}\\s*ERG)|(?:ERG\\s*[-:_]{1,2}\\s*FUS)",
  "t(16;21)/RUNX1::CBFA2T3" = "(?:RUNX1\\s*[-:_]{1,2}\\s*CBFA2T3)|(?:CBFA2T3\\s*[-:_]{1,2}\\s*RUNX1)",
  "t(20;21)/RUNX1::CBFA2T2" = "(?:RUNX1\\s*[-:_]{1,2}\\s*CBFA2T2)|(?:CBFA2T2\\s*[-:_]{1,2}\\s*RUNX1)",
  "inv(16)/CBFA2T3::GLIS2" = "(?:CBFA2T3\\s*[-:_]{1,2}\\s*GLIS2)|(?:GLIS2\\s*[-:_]{1,2}\\s*CBFA2T3)",
  "MYB::GATA1" = paste0(
    "MYB\\s*", FUSION_SEP, "\\s*GATA1",
    "|GATA1\\s*", FUSION_SEP, "\\s*MYB"
  ),
  "CBFA2T3::GLIS3" = paste0(
    "CBFA2T3\\s*", FUSION_SEP, "\\s*GLIS3",
    "|GLIS3\\s*", FUSION_SEP, "\\s*CBFA2T3"
  ),
  "ZEB2::BCL11B" = "(?:ZEB2\\s*[-:_]{1,2}\\s*BCL11B)|(?:BCL11B\\s*[-:_]{1,2}\\s*ZEB2)"
)

# =============================================================================
# KARYOTYPE-ONLY PATTERNS (match ISCN notation like t(X;Y)(...))
# Strict patterns based on actual gene cytogenetic locations
# =============================================================================
abnormalities_karyotype <- list(
  # ===== APL (RARA at 17q21) =====
  # PML at 15q24
  "t(15;17)/PML::RARA" = "t\\(15[;:]17\\)\\(q2[24](?:\\.\\d+)?[;:]q21(?:\\.\\d+)?\\)|t\\(17[;:]15\\)\\(q21(?:\\.\\d+)?[;:]q2[24](?:\\.\\d+)?\\)",
  # IRF2BP2 at 1q42
  "IRF2BP2::RARA" = "t\\(1[;:]17\\)\\(q42(?:\\.\\d+)?[;:]q21(?:\\.\\d+)?\\)|t\\(17[;:]1\\)\\(q21(?:\\.\\d+)?[;:]q42(?:\\.\\d+)?\\)",
  # NPM1 at 5q35
  "NPM1::RARA" = "t\\(5[;:]17\\)\\(q35(?:\\.\\d+)?[;:]q21(?:\\.\\d+)?\\)|t\\(17[;:]5\\)\\(q21(?:\\.\\d+)?[;:]q35(?:\\.\\d+)?\\)",
  # ZBTB16 at 11q23 - NOTE: conflicts with KMT2A, need specific 11q23.2 vs 11q23.3
  "ZBTB16::RARA" = "t\\(11[;:]17\\)\\(q23(?:\\.\\d+)?[;:]q21(?:\\.\\d+)?\\)|t\\(17[;:]11\\)\\(q21(?:\\.\\d+)?[;:]q23(?:\\.\\d+)?\\)",
  
  # ===== Core binding factor =====
  # RUNX1T1 at 8q21-22, RUNX1 at 21q22
  "t(8;21)/RUNX1::RUNX1T1" = "t\\(8[;:]21\\)\\(q2[12](?:\\.\\d+)?[;:]q22(?:\\.\\d+)?\\)|t\\(21[;:]8\\)\\(q22(?:\\.\\d+)?[;:]q2[12](?:\\.\\d+)?\\)",
  # CBFB at 16q22, MYH11 at 16p13
  "inv(16)/t(16;16)/CBFB::MYH11" = "inv\\(16\\)\\(p13(?:\\.\\d+)?q22(?:\\.\\d+)?\\)|t\\(16[;:]16\\)\\(p13(?:\\.\\d+)?[;:]q22(?:\\.\\d+)?\\)",
  
  # ===== KMT2A (MLL) at 11q23.3 =====
  # MLLT3 (AF9) at 9p21-22
  "t(9;11)/MLLT3::KMT2A" = "t\\(9[;:]11\\)\\(p2[123](?:\\.\\d+)?[;:]q23(?:\\.\\d+)?\\)|t\\(11[;:]9\\)\\(q23(?:\\.\\d+)?[;:]p2[123](?:\\.\\d+)?\\)|MLL translocation, t\\(9;11\\)",
  # AFF1 (AF4) at 4q21
  "t(4;11)/AFF1::KMT2A" = "t\\(4[;:]11\\)\\(q21(?:\\.\\d+)?[;:]q23(?:\\.\\d+)?\\)|t\\(11[;:]4\\)\\(q23(?:\\.\\d+)?[;:]q21(?:\\.\\d+)?\\)",
  # MLLT10 (AF10) at 10p12 - STRICT: requires q23 on chr11 (not q14 or q21)
  "t(10;11)/MLLT10::KMT2A" = "t\\(10[;:]11\\)\\(p1[12](?:\\.\\d+)?[;:]q23(?:\\.\\d+)?\\)|t\\(11[;:]10\\)\\(q23(?:\\.\\d+)?[;:]p1[12](?:\\.\\d+)?\\)",
  # PICALM at 11q14 - STRICT: requires q14 on chr11 (not q23)
  "t(10;11)/PICALM::MLLT10" = "t\\(10[;:]11\\)\\(p1[12](?:\\.\\d+)?[;:]q14(?:\\.\\d+)?\\)|t\\(11[;:]10\\)\\(q14(?:\\.\\d+)?[;:]p1[12](?:\\.\\d+)?\\)",
  # TET1 at 10q21.3 - STRICT: requires q21 on chr10 (not p12)
  "t(10;11)(q21.3;q23.3)/TET1::KMT2A" = "t\\(10[;:]11\\)\\(q21(?:\\.\\d+)?[;:]q23(?:\\.\\d+)?\\)|t\\(11[;:]10\\)\\(q23(?:\\.\\d+)?[;:]q21(?:\\.\\d+)?\\)",
  # MLLT1 (ENL) at 19p13.3 - STRICT: p13.3 must end the breakpoint (followed by ) or ;)
  # Pattern matches: t(11;19)(q23;p13.3) or t(11;19)(q23.3;p13.3) but NOT p13.31
  "t(11;19)(q23;p13.3)/KMT2A::MLLT1" = "t\\(11[;:]19\\)\\(q23(?:\\.\\d+)?[;:]p13\\.3[);]|t\\(19[;:]11\\)\\(p13\\.3[;:)]q23(?:\\.\\d+)?\\)",
  # ELL at 19p13.1 - STRICT: p13.1 must end the breakpoint (NOT p13.11 or p13.12)
  # Pattern matches: t(11;19)(q23;p13.1) but NOT t(11;19)(q23;p13.11)
  "t(11;19)(q23;p13.1)/KMT2A::ELL" = "t\\(11[;:]19\\)\\(q23(?:\\.\\d+)?[;:]p13\\.1[);]|t\\(19[;:]11\\)\\(p13\\.1[;:)]q23(?:\\.\\d+)?\\)",
  # MYO1F at 19p13.11 - requires exact p13.11 or p13.2
  "KMT2A::MYO1F" = "t\\(11[;:]19\\)\\(q23(?:\\.\\d+)?[;:]p13\\.(?:11|2)[);]|t\\(19[;:]11\\)\\(p13\\.(?:11|2)[;:)]q23(?:\\.\\d+)?\\)",
  # MLLT4 (AF6/AFDN) at 6q27
  "t(6;11)(q27;q23)/KMT2A::MLLT4" = "t\\(6[;:]11\\)\\(q27(?:\\.\\d+)?[;:]q23(?:\\.\\d+)?\\)|t\\(11[;:]6\\)\\(q23(?:\\.\\d+)?[;:]q27(?:\\.\\d+)?\\)",
  # MLLT6 (AF17) - partner gene at 17q21, KMT2A at 11q23
  "KMT2A::MLLT6" = "t\\(11[;:]17\\)\\(q23(?:\\.\\d+)?[;:]q21(?:\\.\\d+)?\\)|t\\(17[;:]11\\)\\(q21(?:\\.\\d+)?[;:]q23(?:\\.\\d+)?\\)",
  # SEPT6 at Xq24 (historically reported at 2q37)
  "KMT2A::SEPT6" = "t\\(X[;:]11\\)\\(q24(?:\\.\\d+)?[;:]q23(?:\\.\\d+)?\\)|t\\(11[;:]X\\)\\(q23(?:\\.\\d+)?[;:]q24(?:\\.\\d+)?\\)|t\\(2[;:]11\\)\\(q37(?:\\.\\d+)?[;:]q23(?:\\.\\d+)?\\)|t\\(11[;:]2\\)\\(q23(?:\\.\\d+)?[;:]q37(?:\\.\\d+)?\\)",
  # MLLT11 (AF1Q) at 1q21
  "KMT2A::MLLT11" = "t\\(1[;:]11\\)\\(q21(?:\\.\\d+)?[;:]q23(?:\\.\\d+)?\\)|t\\(11[;:]1\\)\\(q23(?:\\.\\d+)?[;:]q21(?:\\.\\d+)?\\)",
  # LASP1 at 17q12
  "KMT2A::LASP1" = "t\\(11[;:]17\\)\\(q23(?:\\.\\d+)?[;:]q12(?:\\.\\d+)?\\)|t\\(17[;:]11\\)\\(q12(?:\\.\\d+)?[;:]q23(?:\\.\\d+)?\\)",
  # SEPT9 at 17q25
  "KMT2A::SEPT9" = "t\\(11[;:]17\\)\\(q23(?:\\.\\d+)?[;:]q25(?:\\.\\d+)?\\)|t\\(17[;:]11\\)\\(q25(?:\\.\\d+)?[;:]q23(?:\\.\\d+)?\\)",
  # FNBP1 at 9q34
  "KMT2A::FNBP1" = "t\\(9[;:]11\\)\\(q34(?:\\.\\d+)?[;:]q23(?:\\.\\d+)?\\)|t\\(11[;:]9\\)\\(q23(?:\\.\\d+)?[;:]q34(?:\\.\\d+)?\\)",
  # EPS15 at 1p32
  "KMT2A::EPS15" = "t\\(1[;:]11\\)\\(p3[12](?:\\.\\d+)?[;:]q23(?:\\.\\d+)?\\)|t\\(11[;:]1\\)\\(q23(?:\\.\\d+)?[;:]p3[12](?:\\.\\d+)?\\)",
  
  # ===== DEK::NUP214 =====
  # DEK at 6p22-23, NUP214 at 9q34
  "t(6;9)/DEK::NUP214" = "t\\(6[;:]9\\)\\(p2[23](?:\\.\\d+)?[;:]q34(?:\\.\\d+)?\\)|t\\(9[;:]6\\)\\(q34(?:\\.\\d+)?[;:]p2[23](?:\\.\\d+)?\\)",
  
  # ===== MECOM (EVI1) at 3q26 =====
  # GATA2 at 3q21, RPN1 at 3q21
  "inv(3)/t(3;3)/GATA2;MECOM" = paste0(
    "GATA2, MECOM|GATA2,MECOM|",
    "inv\\(3\\)\\(q21(?:\\.\\d+)?q26(?:\\.\\d+)?\\)",
    "|t\\(3[;:]3\\)\\(q21(?:\\.\\d+)?[;:]q26(?:\\.\\d+)?\\)",
    "|der\\(3\\)t\\(3[;:]3\\)\\(q21(?:\\.\\d+)?[;:]q26(?:\\.\\d+)?\\)"
  ),
  # t(2;3) with MECOM
  "t(2;3)/MECOM" = "t\\(2[;:]3\\)\\(p1[01](?:\\.\\d+)?[;:]q26(?:\\.\\d+)?\\)|t\\(3[;:]2\\)\\(q26(?:\\.\\d+)?[;:]p1[01](?:\\.\\d+)?\\)",
  # MYC at 8q24
  "t(3;8)/MYC,MECOM" = "t\\(3[;:]8\\)\\(q26(?:\\.\\d+)?[;:]q24(?:\\.\\d+)?\\)|t\\(8[;:]3\\)\\(q24(?:\\.\\d+)?[;:]q26(?:\\.\\d+)?\\)",
  # ETV6 at 12p13
  "t(3;12)/ETV6::MECOM" = "t\\(3[;:]12\\)\\(q26(?:\\.\\d+)?[;:]p13(?:\\.\\d+)?\\)|t\\(12[;:]3\\)\\(p13(?:\\.\\d+)?[;:]q26(?:\\.\\d+)?\\)",
  # RUNX1 at 21q22
  "t(3;21)/MECOM::RUNX1" = "t\\(3[;:]21\\)\\(q26(?:\\.\\d+)?[;:]q22(?:\\.\\d+)?\\)|t\\(21[;:]3\\)\\(q22(?:\\.\\d+)?[;:]q26(?:\\.\\d+)?\\)",
  # t(3;6) with MECOM - partner at 6q24-25
  "t(3;6)/MECOM" = "t\\(3[;:]6\\)\\(q26(?:\\.\\d+)?[;:]q2[345](?:\\.\\d+)?\\)|t\\(6[;:]3\\)\\(q2[345](?:\\.\\d+)?[;:]q26(?:\\.\\d+)?\\)",
  # Unspecified 3q26 rearrangement
  "3q26_rearrangement_unspecified_MECOM" = "3q26(?:\\.\\d+)?\\s*(?:rearrangement|aberration|alteration|translocation|inversion|amplification|abnormality|abn)",
  
  # ===== BCR::ABL1 =====
  # BCR at 22q11, ABL1 at 9q34
  "t(9;22)/BCR::ABL1" = "t\\(9[;:]22\\)\\(q34(?:\\.\\d+)?[;:]q11(?:\\.\\d+)?\\)|t\\(22[;:]9\\)\\(q11(?:\\.\\d+)?[;:]q34(?:\\.\\d+)?\\)",
  
  # ===== Rare fusions =====
  # PRDM16 at 1p36, RPN1 at 3q21
  "t(1;3)/PRDM16::RPN1" = "t\\(1[;:]3\\)\\(p36(?:\\.\\d+)?[;:]q21(?:\\.\\d+)?\\)|t\\(3[;:]1\\)\\(q21(?:\\.\\d+)?[;:]p36(?:\\.\\d+)?\\)",
  # NPM1 at 5q35, MLF1 at 3q25
  "t(3;5)/NPM1::MLF1" = "t\\(3[;:]5\\)\\(q25(?:\\.\\d+)?[;:]q35(?:\\.\\d+)?\\)|t\\(5[;:]3\\)\\(q35(?:\\.\\d+)?[;:]q25(?:\\.\\d+)?\\)",
  # KAT6A (MOZ) at 8p11, CREBBP at 16p13
  "t(8;16)/KAT6A::CREBBP" = "t\\(8[;:]16\\)\\(p11(?:\\.\\d+)?[;:]p13(?:\\.\\d+)?\\)|t\\(16[;:]8\\)\\(p13(?:\\.\\d+)?[;:]p11(?:\\.\\d+)?\\)",
  # RBM15 at 1p13, MRTF1/MKL1 at 22q13
  "t(1;22)/RBM15::MRTF1" = "t\\(1[;:]22\\)\\(p13(?:\\.\\d+)?[;:]q13(?:\\.\\d+)?\\)|t\\(22[;:]1\\)\\(q13(?:\\.\\d+)?[;:]p13(?:\\.\\d+)?\\)",
  
  # ===== NUP98 at 11p15 =====
  # NSD1 at 5q35
  "t(5;11)/NUP98::NSD1" = "t\\(5[;:]11\\)\\(q35(?:\\.\\d+)?[;:]p15(?:\\.\\d+)?\\)|t\\(11[;:]5\\)\\(p15(?:\\.\\d+)?[;:]q35(?:\\.\\d+)?\\)",
  # KDM5A at 12p13
  "t(11;12)/NUP98::KDM5A" = "t\\(11[;:]12\\)\\(p15(?:\\.\\d+)?[;:]p13(?:\\.\\d+)?\\)|t\\(12[;:]11\\)\\(p13(?:\\.\\d+)?[;:]p15(?:\\.\\d+)?\\)",
  # HOXA9 at 7p15
  "NUP98::HOXA9" = "t\\(7[;:]11\\)\\(p15(?:\\.\\d+)?[;:]p15(?:\\.\\d+)?\\)|t\\(11[;:]7\\)\\(p15(?:\\.\\d+)?[;:]p15(?:\\.\\d+)?\\)",
  # HOXD13 at 2q31
  "NUP98::HOXD13" = "t\\(2[;:]11\\)\\(q31(?:\\.\\d+)?[;:]p15(?:\\.\\d+)?\\)|t\\(11[;:]2\\)\\(p15(?:\\.\\d+)?[;:]q31(?:\\.\\d+)?\\)",
  # DDX10 at 11q22 (intrachromosomal, but often reported as t(10;11))
  "NUP98::DDX10" = "inv\\(11\\)\\(p15(?:\\.\\d+)?q22(?:\\.\\d+)?\\)|t\\(11[;:]11\\)\\(p15(?:\\.\\d+)?[;:]q22(?:\\.\\d+)?\\)",
  # LEDGF/PSIP1 at 9p22
  "NUP98::LEDGF" = "t\\(9[;:]11\\)\\(p22(?:\\.\\d+)?[;:]p15(?:\\.\\d+)?\\)|t\\(11[;:]9\\)\\(p15(?:\\.\\d+)?[;:]p22(?:\\.\\d+)?\\)",
  # RARG at 12q13
  "NUP98::RARG" = "t\\(11[;:]12\\)\\(p15(?:\\.\\d+)?[;:]q13(?:\\.\\d+)?\\)|t\\(12[;:]11\\)\\(q13(?:\\.\\d+)?[;:]p15(?:\\.\\d+)?\\)",
  # HOXC11/HOXC13 at 12q13
  "NUP98::HOXC11" = "t\\(11[;:]12\\)\\(p15(?:\\.\\d+)?[;:]q13(?:\\.\\d+)?\\)|t\\(12[;:]11\\)\\(q13(?:\\.\\d+)?[;:]p15(?:\\.\\d+)?\\)",
  "NUP98::HOXC13" = "t\\(11[;:]12\\)\\(p15(?:\\.\\d+)?[;:]q13(?:\\.\\d+)?\\)|t\\(12[;:]11\\)\\(q13(?:\\.\\d+)?[;:]p15(?:\\.\\d+)?\\)",
  # PHF23 at 17p13
  "NUP98::PHF23" = "t\\(11[;:]17\\)\\(p15(?:\\.\\d+)?[;:]p13(?:\\.\\d+)?\\)|t\\(17[;:]11\\)\\(p13(?:\\.\\d+)?[;:]p15(?:\\.\\d+)?\\)",
  # Catch-all for NUP98 fusions involving 11p15
  "NUP98_all" = "t\\([^)]+[;:]11\\)\\([^)]+[;:]p15(?:\\.\\d+)?\\)|t\\(11[;:][^)]+\\)\\(p15(?:\\.\\d+)?[;:][^)]+\\)|inv\\(11\\)\\(p15(?:\\.\\d+)?[^)]+\\)",
  
  # ===== Other rare fusions =====
  # MNX1 at 7q36, ETV6 at 12p13
  "t(7;12)/ETV6::MNX1" = "t\\(7[;:]12\\)\\(q36(?:\\.\\d+)?[;:]p13(?:\\.\\d+)?\\)|t\\(12[;:]7\\)\\(p13(?:\\.\\d+)?[;:]q36(?:\\.\\d+)?\\)",
  # MYB at 6q23, GATA1 at Xp11
  "t(X;6)/MYB::GATA1" = "t\\(X[;:]6\\)\\(p11(?:\\.\\d+)?[;:]q23(?:\\.\\d+)?\\)|t\\(6[;:]X\\)\\(q23(?:\\.\\d+)?[;:]p11(?:\\.\\d+)?\\)",
  # FUS at 16p11, ERG at 21q22
  "t(16;21)/FUS::ERG" = "t\\(16[;:]21\\)\\(p11(?:\\.\\d+)?[;:]q22(?:\\.\\d+)?\\)|t\\(21[;:]16\\)\\(q22(?:\\.\\d+)?[;:]p11(?:\\.\\d+)?\\)",
  # RUNX1 at 21q22, CBFA2T3 at 16q24
  "t(16;21)/RUNX1::CBFA2T3" = "t\\(16[;:]21\\)\\(q24(?:\\.\\d+)?[;:]q22(?:\\.\\d+)?\\)|t\\(21[;:]16\\)\\(q22(?:\\.\\d+)?[;:]q24(?:\\.\\d+)?\\)",
  # RUNX1 at 21q22, CBFA2T2 at 20q11
  "t(20;21)/RUNX1::CBFA2T2" = "t\\(20[;:]21\\)\\(q11(?:\\.\\d+)?[;:]q22(?:\\.\\d+)?\\)|t\\(21[;:]20\\)\\(q22(?:\\.\\d+)?[;:]q11(?:\\.\\d+)?\\)",
  # CBFA2T3 at 16q24, GLIS2 at 16p13
  "inv(16)/CBFA2T3::GLIS2" = "inv\\(16\\)\\(p13(?:\\.\\d+)?q24(?:\\.\\d+)?\\)",
  # CBFA2T3 at 16q24, GLIS3 at 9p24
  "t(9;16)/CBFA2T3::GLIS3" = "t\\(9[;:]16\\)\\(p24(?:\\.\\d+)?[;:]q24(?:\\.\\d+)?\\)|t\\(16[;:]9\\)\\(q24(?:\\.\\d+)?[;:]p24(?:\\.\\d+)?\\)",
  
  # ===== Cytogenetic abnormalities (karyotype-only) =====
  "del(5q)" = "del\\(5\\)\\(q\\d+.*?\\)",
  "del(7q)" = "del\\(7\\)\\(q\\d+.*?\\)",
  "del(9q)" = "del\\(9\\)\\(q\\d+.*?\\)",
  # Monosomy patterns: match -5, -7, etc. preceded by comma/space/start and followed by comma/bracket/end
  "monosomy_5" = "(?:^|[,;\\s])-5(?:[,;\\[\\]\\s]|$)",
  "monosomy_7" = "(?:^|[,;\\s])-7(?:[,;\\[\\]\\s]|$)",
  "trisomy_8" = "(?:^|[,;\\s])\\+8(?:[,;\\[\\]\\s]|$)",
  "trisomy_21" = "(?:^|[,;\\s])\\+21(?:[,;\\[\\]\\s]|$)",
  "i(17q)" = "i\\(17\\)\\(q10\\)|i\\(17q\\)",
  "del(17p)" = "del\\(17\\)\\(p\\d+.*?\\)",
  "add(17p)" = "add\\(17\\)\\(p\\d+.*?\\)",
  "monosomy_17" = "(?:^|[,;\\s])-17(?:[,;\\[\\]\\s]|$)",
  "del(20q)" = "del\\(20\\)\\(q\\d+.*?\\)",
  "idic(X)(q13)" = "idic\\(X\\)\\(q13(?:\\.\\d+)?\\)",
  "5q_abn" = "(?:del|t|add)\\(5\\)\\(q.*?\\)",
  "12p_abn" = "(?:del|t|add)\\(12\\)\\(p.*?\\)"
)

# =============================================================================
# MUTATION PATTERNS (search in diagnosis/clinical text)
# =============================================================================
abnormalities_mutation <- list(
  "mutated_NPM1" = "(?i)(?:mutated|mutation).*?NPM1|NPM1.*?(?:mutated|mutation)",
  "in_frame_bZIP_CEBPA" = "(?i)(?:mutated|mutation).*?CEBPA|CEBPA.*?(?:mutated|mutation)"
)

# =============================================================================
# PERL PATTERNS (require perl=TRUE)
# =============================================================================
abnormalities_perl <- list(
  "complex_karyotype" = "^(?=(?:.*?(?:del|dup|t|i|\\+|-)\\([^\\)]+\\)){3,})|complex|Complex Cytogenetics|Complex|myelodysplasia-related changes",
  "other_KMT2A_rearrangements" = "(?i)(?:(?:t\\([^\\)]+;11\\)|t\\(11[;:][^\\)]+\\)).*?(?:q23|KMT2A|MLL))|(KMT2A(?:[-:_]{1,2})\\w+)|(\\w+(?:[-:_]{1,2})KMT2A)"
)

# =============================================================================
# HELPER FUNCTION: Apply patterns with fusion precedence
# =============================================================================
#' Match abnormalities with fusion calling taking precedence over karyotyping
#'
#' @param abn_name Name of the abnormality to match
#' @param fusion_vectors List or vector of fusion calling data (takes precedence)
#' @param karyotype_vectors List or vector of karyotype/ISCN data (fallback)
#' @param diagnosis_vectors Optional list or vector of diagnosis text
#' @param sample_has_any_fusion Optional logical vector indicating if sample has ANY fusion match
#' @return Logical vector indicating matches
match_abnormality_with_precedence <- function(abn_name,
                                               fusion_vectors,
                                               karyotype_vectors,
                                               diagnosis_vectors = NULL,
                                               sample_has_any_fusion = NULL) {
  
  # Convert single vectors to lists for uniform handling
  if (!is.list(fusion_vectors)) fusion_vectors <- list(fusion_vectors)
  if (!is.list(karyotype_vectors)) karyotype_vectors <- list(karyotype_vectors)
  if (!is.null(diagnosis_vectors) && !is.list(diagnosis_vectors)) {
    diagnosis_vectors <- list(diagnosis_vectors)
  }
  
  n <- length(fusion_vectors[[1]])
  
  # Initialize result
  result <- rep(FALSE, n)
  fusion_matched <- rep(FALSE, n)
  
  # Step 1: Check fusion patterns (takes precedence)
  if (abn_name %in% names(abnormalities_fusion)) {
    fusion_pattern <- abnormalities_fusion[[abn_name]]
    for (vec in fusion_vectors) {
      fusion_matched <- fusion_matched | grepl(fusion_pattern, vec, ignore.case = TRUE)
    }
    result <- fusion_matched
  }
  
  # Step 2: For samples without ANY fusion match, check karyotype patterns
  # Key change: only use karyotype if the sample has NO fusion data at all
  if (abn_name %in% names(abnormalities_karyotype)) {
    karyotype_pattern <- abnormalities_karyotype[[abn_name]]
    karyotype_matched <- rep(FALSE, n)
    for (vec in karyotype_vectors) {
      karyotype_matched <- karyotype_matched | grepl(karyotype_pattern, vec, ignore.case = TRUE)
    }
    # Only use karyotype match if sample has NO fusion data matching ANY pattern
    if (!is.null(sample_has_any_fusion)) {
      result <- result | (!sample_has_any_fusion & karyotype_matched)
    } else {
      # Fallback to old behavior if sample_has_any_fusion not provided
      result <- result | (!fusion_matched & karyotype_matched)
    }
  }
  
  # Step 3: Check diagnosis/clinical text for mutation patterns
  if (!is.null(diagnosis_vectors) && abn_name %in% names(abnormalities_mutation)) {
    mutation_pattern <- abnormalities_mutation[[abn_name]]
    for (vec in diagnosis_vectors) {
      result <- result | grepl(mutation_pattern, vec, ignore.case = TRUE)
    }
  }
  
  return(result)
}

#' Check if samples have ANY fusion pattern match
#' @param fusion_vectors List of fusion calling vectors
#' @return Logical vector indicating if sample has any fusion match
check_any_fusion_match <- function(fusion_vectors) {
  if (!is.list(fusion_vectors)) fusion_vectors <- list(fusion_vectors)
  n <- length(fusion_vectors[[1]])
  any_match <- rep(FALSE, n)
  
  for (abn_name in names(abnormalities_fusion)) {
    fusion_pattern <- abnormalities_fusion[[abn_name]]
    for (vec in fusion_vectors) {
      any_match <- any_match | grepl(fusion_pattern, vec, ignore.case = TRUE)
    }
  }
  return(any_match)
}

#' Get all abnormality names that have patterns defined
#' @return Character vector of abnormality names
get_all_abnormality_names <- function() {
  unique(c(names(abnormalities_fusion), 
           names(abnormalities_karyotype),
           names(abnormalities_mutation)))
}

#' Apply all abnormality patterns to data with fusion precedence
#'
#' @param sample_ids Vector of sample IDs
#' @param fusion_vectors List of fusion calling vectors
#' @param karyotype_vectors List of karyotype/ISCN vectors
#' @param diagnosis_vectors Optional list of diagnosis text vectors
#' @return Data frame with sample IDs and one-hot encoded abnormalities
apply_all_abnormalities <- function(sample_ids,
                                    fusion_vectors,
                                    karyotype_vectors,
                                    diagnosis_vectors = NULL) {
  
  results <- data.frame(Sample.ID = sample_ids)
  abn_names <- get_all_abnormality_names()
  
  # Check if all data is NA for a sample
  all_na <- Reduce(`&`, lapply(c(fusion_vectors, karyotype_vectors), is.na))
  
  # First, check which samples have ANY fusion match (for precedence logic)
  sample_has_any_fusion <- check_any_fusion_match(fusion_vectors)
  
  for (abn in abn_names) {
    matched <- match_abnormality_with_precedence(
      abn, fusion_vectors, karyotype_vectors, diagnosis_vectors,
      sample_has_any_fusion = sample_has_any_fusion
    )
    results[[abn]] <- ifelse(all_na, NA, as.integer(matched))
  }
  
  return(results)
}

# =============================================================================
# LEGACY SUPPORT: Combined patterns for backwards compatibility
# =============================================================================
# These combine fusion and karyotype patterns with OR for scripts that don't
# use the new precedence system yet
abnormalities <- list()
for (abn in get_all_abnormality_names()) {
  patterns <- c()
  if (abn %in% names(abnormalities_fusion)) {
    patterns <- c(patterns, abnormalities_fusion[[abn]])
  }
  if (abn %in% names(abnormalities_karyotype)) {
    patterns <- c(patterns, abnormalities_karyotype[[abn]])
  }
  if (abn %in% names(abnormalities_mutation)) {
    patterns <- c(patterns, abnormalities_mutation[[abn]])
  }
  if (length(patterns) > 0) {
    abnormalities[[abn]] <- paste(patterns, collapse = "|")
  }
}
