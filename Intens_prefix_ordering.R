# ============================================================================
# Informational predictors of prefix linearization in Italian intensifiers
# Author: Anonymous
# Date: April, 2026
# ============================================================================
#
# This script analyzes the linearization of stacking intensifying prefixes
# (arci-, extra-, iper-, stra-, super-, ultra-) in Italian multi-prefix
# constructions. It tests whether two information-theoretic measures --
# contextual entropy (ICTRANS) and paradigmatic surprisal (ICLOCAL) 
# -- predict the order in which prefixes appear when stacked.
#
# Pipeline:
#   1. Load per-prefix corpus data and compute ICTRANS + ICLOCAL per prefix
#      (with Miller-Madow bias correction for entropy).
#   2. Correlate the two measures.
#   3. Load multi-prefix stacking data; construct adjacent and non-adjacent
#      pairwise contrasts.
#   4. Fit mixed-effects logistic regressions to test whether the
#      informational asymmetry between two prefixes predicts which one
#      appears leftward.
#   5. Run residual diagnostics to identify any prefix-specific deviations
#      from the informational model.
#   6. Fit Plackett-Luce robustness check.
#   7. Transitivity analysis: test whether pairwise ordering proportions
#      form a strict total order, and recover the best-fitting global
#      ranking of the six prefixes.
#
# Inputs (on ~/Desktop by default):
#   - <prefix>_intensifiers.csv for each of the six prefixes
#   - prefix_stackings.xlsx (multi-prefix construction tokens)
# ============================================================================

# ----------------------------------------------------------------------------
# 0. Libraries
# ----------------------------------------------------------------------------

install.packages(c(
  "dplyr",
  "tidyr",
  "readr",
  "readxl",
  "purrr",
  "lme4",
  "broom.mixed",
  "PlackettLuce",
  "ggplot2",
  "ggrepel",
  "Cairo",
  "combinat"
))

library(dplyr)
library(tidyr)
library(readr)
library(readxl)
library(purrr)
library(lme4)
library(broom.mixed)
library(PlackettLuce)
library(ggplot2)
library(ggrepel)
library(Cairo)
library(combinat)

# ----------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------
folder_path <- "~/Desktop"
prefixes    <- c("arci", "extra", "iper", "stra", "super", "ultra")

# ----------------------------------------------------------------------------
# 2. Load per-prefix intensifier data
# ----------------------------------------------------------------------------
# Each CSV has two columns: Word (the base) and (implicitly) the prefix,
# which we attach manually. We also exclude "stragrande" because it is
# lexicalized and would distort the stra- results.

all_data <- prefixes %>%
  map_df(function(pref) {
    file <- file.path(folder_path, paste0(pref, "_intensifiers.csv"))
    df   <- read_csv(file, show_col_types = FALSE)
    df$Prefix <- pref
    df
  }) %>%
  filter(!(Prefix == "stra" & Word == "grande"))

head(all_data)

# ----------------------------------------------------------------------------
# 3. Compute ICTRANS (contextual entropy of bases given prefix)
# ----------------------------------------------------------------------------
# For each prefix, we treat the distribution of bases as a probability
# distribution and compute its Shannon entropy. The Miller-Madow correction
# adds (k-1)/(2N) nats to the raw plug-in entropy estimate, then converts
# to bits by dividing by ln(2). This corrects the downward bias of the
# plug-in estimator on finite samples.

freq_table <- all_data %>%
  group_by(Prefix, Word) %>%
  summarise(Freq = n(), .groups = "drop")

ICTRANS <- freq_table %>%
  group_by(Prefix) %>%
  mutate(P_base_given_prefix = Freq / sum(Freq)) %>%
  summarise(
    ICTRANS_raw  = -sum(P_base_given_prefix * log2(P_base_given_prefix)),
    prefix_total = sum(Freq),
    k            = n(),
    .groups = "drop"
  ) %>%
  mutate(
    MM_correction = (k - 1) / (2 * prefix_total) / log(2),
    ICTRANS_MM    = ICTRANS_raw + MM_correction
  )

# ----------------------------------------------------------------------------
# 4. Compute ICLOCAL (Shannon surprisal of each prefix overall)
# ----------------------------------------------------------------------------
# ICLOCAL = -log2(P(prefix)), where P(prefix) is the prefix's share of the
# pooled prefix-token corpus.

total_tokens <- sum(ICTRANS$prefix_total)

info_measures <- ICTRANS %>%
  mutate(
    Pw      = prefix_total / total_tokens,
    ICLOCAL = -log2(Pw)
  ) %>%
  select(Prefix, prefix_total, Pw, ICLOCAL,
         ICTRANS_raw, ICTRANS_MM, MM_correction)

print(info_measures)

write.csv(
  info_measures,
  file.path(folder_path, "prefix_information_values_MillerMadow.csv"),
  row.names = FALSE
)

# ----------------------------------------------------------------------------
# 5. Correlation between the two measures
# ----------------------------------------------------------------------------

cor.test(info_measures$ICTRANS_MM, info_measures$ICLOCAL, method = "spearman")

# ============================================================================
# STACKING ANALYSIS
# ============================================================================

# ----------------------------------------------------------------------------
# 6. Load the multi-prefix stacking dataset
# ----------------------------------------------------------------------------
# Expected columns: TYPE, token, prefix_1, prefix_2, prefix_3, prefix_4, base.
# prefix_1 is the outermost (leftmost) prefix; prefix_4 is the innermost.

stackings <- read_excel(file.path(folder_path, "prefix_stackings.xlsx"))

stopifnot("TYPE must be unique per row" = !anyDuplicated(stackings$TYPE))

# ----------------------------------------------------------------------------
# 7. Build pairwise contrasts
# ----------------------------------------------------------------------------
# Two kinds of pairs:
#   - Adjacent pairs: consecutive positions (prefix_1-prefix_2,
#                     prefix_2-prefix_3, prefix_3-prefix_4).
#   - Non-adjacent pairs: outermost vs. innermost when there are at least
#                         3 stacked prefixes.
# Each pair is tagged with an adjacency column.

# --- 7a. Adjacent pairs ---
stacking_pairs_adjacent <- stackings %>%
  mutate(
    row_id = row_number(),
    across(starts_with("prefix_"), as.character)
  ) %>%
  pivot_longer(
    cols      = starts_with("prefix_"),
    names_to  = "pos",
    values_to = "prefix"
  ) %>%
  filter(!is.na(prefix)) %>%
  mutate(pos_num = as.integer(gsub("prefix_", "", pos))) %>%
  arrange(row_id, pos_num) %>%
  group_by(row_id, TYPE, base) %>%
  mutate(next_prefix = lead(prefix)) %>%
  ungroup() %>%
  filter(!is.na(next_prefix)) %>%
  rename(prefix_left = prefix, prefix_right = next_prefix) %>%
  mutate(adjacency = "adjacent") %>%
  select(TYPE, base, prefix_left, prefix_right, adjacency)

# --- 7b. Non-adjacent pairs ---
stacking_pairs_nonadjacent <- stackings %>%
  mutate(
    row_id = row_number(),
    across(starts_with("prefix_"), as.character)
  ) %>%
  pivot_longer(
    cols      = starts_with("prefix_"),
    names_to  = "pos",
    values_to = "prefix"
  ) %>%
  filter(!is.na(prefix)) %>%
  mutate(pos_num = as.integer(gsub("prefix_", "", pos))) %>%
  group_by(row_id, TYPE, base) %>%
  filter(n() >= 3) %>%
  summarise(
    prefix_left  = prefix[pos_num == min(pos_num)],
    prefix_right = prefix[pos_num == max(pos_num)],
    .groups = "drop"
  ) %>%
  filter(prefix_left != prefix_right) %>%
  mutate(adjacency = "non_adjacent") %>%
  select(TYPE, base, prefix_left, prefix_right, adjacency)

stacking_pairs <- bind_rows(stacking_pairs_adjacent,
                            stacking_pairs_nonadjacent)

# ----------------------------------------------------------------------------
# 8. Attach informational values to each side of each pair
# ----------------------------------------------------------------------------
# Merge in ICTRANS_MM and ICLOCAL for both prefix_left and prefix_right,
# then compute the left-minus-right differences. These are descriptive
# only (they are NOT used as predictors in the logistic model below, where
# we recode to a direction-independent reference frame -- see Section 10).

stacked_full <- stacking_pairs %>%
  left_join(info_measures %>% select(Prefix, ICTRANS_MM, ICLOCAL),
            by = c("prefix_left" = "Prefix")) %>%
  rename(ICTRANS_MM_left = ICTRANS_MM, ICLOCAL_left = ICLOCAL) %>%
  left_join(info_measures %>% select(Prefix, ICTRANS_MM, ICLOCAL),
            by = c("prefix_right" = "Prefix")) %>%
  rename(ICTRANS_MM_right = ICTRANS_MM, ICLOCAL_right = ICLOCAL) %>%
  mutate(
    dICTRANS = ICTRANS_MM_left - ICTRANS_MM_right,
    dICLOCAL = ICLOCAL_left    - ICLOCAL_right
  )

# ----------------------------------------------------------------------------
# 9. Descriptive tests on raw left-right differences
# ----------------------------------------------------------------------------
# DIAGNOSTIC ONLY -- proportions are relative to the observed left-right
# order and are trivially inflated.

cat("Share of pairs where left prefix has higher ICTRANS:",
    mean(stacked_full$dICTRANS > 0), "\n")
cat("Share of pairs where right prefix has higher ICLOCAL:",
    mean(stacked_full$dICLOCAL < 0), "\n")

cor.test(stacked_full$dICTRANS, stacked_full$dICLOCAL, method = "spearman")

# ============================================================================
# MIXED-EFFECTS PIPELINE
# ============================================================================

# ----------------------------------------------------------------------------
# 10. Direction-consistent recoding
# ----------------------------------------------------------------------------
# For inferential modeling, we recode every pair into a fixed,
# order-independent reference frame:
#   A = alphabetically-first prefix
#   B = alphabetically-second prefix
#   observed = 1 if the attested left prefix is A, else 0
#   dICTRANS = ICTRANS(A) - ICTRANS(B)
#   dICLOCAL = ICLOCAL(A) - ICLOCAL(B)
# This ensures predictors are independent of the outcome.

stacked_model <- stacked_full %>%
  filter(prefix_left != prefix_right) %>%
  mutate(
    prefix_A = pmin(prefix_left, prefix_right),
    prefix_B = pmax(prefix_left, prefix_right),
    pair     = paste(prefix_A, prefix_B, sep = "_"),
    observed = as.integer(prefix_left == prefix_A)
  ) %>%
  left_join(info_measures %>% select(Prefix, ICTRANS_MM, ICLOCAL),
            by = c("prefix_A" = "Prefix")) %>%
  rename(ICTRANS_A = ICTRANS_MM, ICLOCAL_A = ICLOCAL) %>%
  left_join(info_measures %>% select(Prefix, ICTRANS_MM, ICLOCAL),
            by = c("prefix_B" = "Prefix")) %>%
  rename(ICTRANS_B = ICTRANS_MM, ICLOCAL_B = ICLOCAL) %>%
  mutate(
    dICTRANS = ICTRANS_A - ICTRANS_B,
    dICLOCAL = ICLOCAL_A - ICLOCAL_B
  )

# ----------------------------------------------------------------------------
# 11. Pair-by-pair counts (diagnostic)
# ----------------------------------------------------------------------------

pair_counts <- stacked_model %>%
  group_by(pair, prefix_A, prefix_B) %>%
  summarise(
    n_total     = n(),
    n_A_left    = sum(observed == 1),
    n_B_left    = sum(observed == 0),
    prop_A_left = mean(observed),
    dICTRANS    = first(dICTRANS),
    dICLOCAL    = first(dICLOCAL),
    .groups = "drop"
  ) %>%
  arrange(desc(n_total))

print(pair_counts, n = Inf)

write.csv(pair_counts,
          file.path(folder_path, "pair_counts.csv"),
          row.names = FALSE)

top3_share <- sum(head(pair_counts$n_total, 3)) / sum(pair_counts$n_total)
cat(sprintf("Top 3 pairs account for %.1f%% of all pair-tokens\n",
            100 * top3_share))

# ----------------------------------------------------------------------------
# 12. Mixed-effects logistic regression
# ----------------------------------------------------------------------------
# Random intercept for base (repeated adjectival bases).
# (1 | pair) is intentionally omitted: dICTRANS and dICLOCAL are constant
# within each pair by construction, making a pair-level random intercept
# collinear with the fixed effects.
# Both predictors are z-standardized.

stacked_model <- stacked_model %>%
  mutate(
    dICTRANS_z = as.numeric(scale(dICTRANS)),
    dICLOCAL_z = as.numeric(scale(dICLOCAL))
  )

ctrl <- glmerControl(optimizer = "bobyqa",
                     optCtrl   = list(maxfun = 2e5))

m_null <- glmer(observed ~ 1 + (1 | base),
                data    = stacked_model,
                family  = binomial,
                control = ctrl)

m_ce <- glmer(observed ~ dICTRANS_z + (1 | base),
              data    = stacked_model,
              family  = binomial,
              control = ctrl)

m_ps <- glmer(observed ~ dICLOCAL_z + (1 | base),
              data    = stacked_model,
              family  = binomial,
              control = ctrl)

m_both <- glmer(observed ~ dICTRANS_z + dICLOCAL_z + (1 | base),
                data    = stacked_model,
                family  = binomial,
                control = ctrl)

# ----------------------------------------------------------------------------
# 13. Model summaries and likelihood-ratio tests
# ----------------------------------------------------------------------------
summary(m_ce)
summary(m_ps)
summary(m_both)

cat("\n--- LRT: null vs. dICTRANS-only ---\n")
print(anova(m_null, m_ce))

cat("\n--- LRT: null vs. dICLOCAL-only ---\n")
print(anova(m_null, m_ps))

cat("\n--- LRT: dICTRANS-only vs. joint (does dICLOCAL add anything?) ---\n")
print(anova(m_ce, m_both))

cat("\n--- LRT: dICLOCAL-only vs. joint (does dICTRANS add anything?) ---\n")
print(anova(m_ps, m_both))

cat("\n--- Joint model coefficients (with 95% Wald CIs) ---\n")
print(broom.mixed::tidy(m_both, effects = "fixed", conf.int = TRUE))

# ----------------------------------------------------------------------------
# 14. Adjacency interaction (adjacent vs. non-adjacent pairs)
# ----------------------------------------------------------------------------
# Tests whether the informational gradient operates differently for
# locally adjacent vs. distant (outermost-innermost) prefix relationships.
# Two steps:
#   (a) Does adjacency shift the baseline ordering probability?
#   (b) Does adjacency modulate the informational gradient itself?

if (length(unique(stacked_model$adjacency)) > 1) {
  
  m_adj_main <- glmer(
    observed ~ dICTRANS_z + dICLOCAL_z + adjacency + (1 | base),
    data    = stacked_model,
    family  = binomial,
    control = ctrl
  )
  
  m_adj_int <- glmer(
    observed ~ (dICTRANS_z + dICLOCAL_z) * adjacency + (1 | base),
    data    = stacked_model,
    family  = binomial,
    control = ctrl
  )
  
  cat("\n--- LRT: joint vs. joint + adjacency main effect ---\n")
  print(anova(m_both, m_adj_main))
  
  cat("\n--- LRT: adjacency main effect vs. full interaction ---\n")
  print(anova(m_adj_main, m_adj_int))
  
  cat("\n--- Adjacency main effect model summary ---\n")
  print(summary(m_adj_main))
  
  cat("\n--- Adjacency interaction model summary ---\n")
  print(summary(m_adj_int))
}

# ----------------------------------------------------------------------------
# 15. Per-pair sanity check (aggregated, one row per pair)
# ----------------------------------------------------------------------------
# Correlates prop_A_left with dICTRANS and dICLOCAL across unordered pairs.
# Immune to within-pair repeated observations; interpret as complement to
# the mixed model.

cat("\n--- Per-pair Spearman correlations ---\n")
print(cor.test(pair_counts$prop_A_left, pair_counts$dICTRANS,
               method = "spearman"))
print(cor.test(pair_counts$prop_A_left, pair_counts$dICLOCAL,
               method = "spearman"))

# ----------------------------------------------------------------------------
# 16. Residual diagnostics
# ----------------------------------------------------------------------------
# Aggregate predicted vs. observed proportions by pair. Large residuals
# flag pairs the informational model fails to capture. We then test whether
# residuals are driven by sample size or by a specific prefix (arci-).

stacked_model$pred <- predict(m_both, type = "response", re.form = NULL)

pair_residuals <- stacked_model %>%
  group_by(pair) %>%
  summarise(
    n_total   = n(),
    observed  = mean(observed),
    predicted = mean(pred),
    residual  = mean(observed) - mean(pred),
    .groups   = "drop"
  ) %>%
  arrange(desc(abs(residual))) %>%
  mutate(
    has_arci = grepl("arci", pair),
    log_n    = log(n_total)
  )

print(pair_residuals)

cor.test(abs(pair_residuals$residual), pair_residuals$log_n)

pair_residuals %>%
  filter(!has_arci) %>%
  summarise(max_abs_resid  = max(abs(residual)),
            mean_abs_resid = mean(abs(residual)))

cat("\n--- Absolute residual ~ log_n + has_arci ---\n")
print(summary(lm(abs(residual) ~ log_n + has_arci,
                 data = pair_residuals)))

cat("\n--- Signed residual ~ log_n + has_arci ---\n")
print(summary(lm(residual ~ log_n + has_arci,
                 data = pair_residuals)))

# ----------------------------------------------------------------------------
# 17. Bootstrap CI for the non-arci mean absolute residual
# ----------------------------------------------------------------------------
# Bootstraps at the token level within non-arci pairs to correct for
# the downward bias of the plug-in MAR estimator on small samples.
# Each replicate resamples tokens with replacement within each pair,
# recomputes the mean absolute residual, and returns it.

set.seed(123)
n_boot <- 2000

boot_data <- stacked_model %>%
  mutate(has_arci = grepl("arci", pair)) %>%
  filter(!has_arci)

boot_one <- function(df) {
  df %>%
    group_by(pair) %>%
    slice_sample(prop = 1, replace = TRUE) %>%
    summarise(
      observed  = mean(observed),
      predicted = mean(pred),
      .groups   = "drop"
    ) %>%
    mutate(residual = observed - predicted) %>%
    summarise(mean_abs_resid = mean(abs(residual))) %>%
    pull(mean_abs_resid)
}

boot_estimates <- replicate(n_boot, boot_one(boot_data))

estimate <- mean(boot_estimates)
ci       <- quantile(boot_estimates, c(0.025, 0.5, 0.975))

cat(sprintf(
  "\nNon-arci MAR (bootstrap median): %.4f\n  95%% CI: [%.4f, %.4f]\n  Based on %d replicates\n",
  ci[2], ci[1], ci[3], n_boot
))

# ============================================================================
# PLACKETT-LUCE ROBUSTNESS CHECK
# ============================================================================

# ----------------------------------------------------------------------------
# 18. Build numeric ranking matrix
# ----------------------------------------------------------------------------
# Rows = tokens, columns = prefixes.
# Values = position in the stack (1 = leftmost); 0 = absent.

prefix_cols <- c("prefix_1", "prefix_2", "prefix_3", "prefix_4")

rank_matrix <- t(apply(stackings[, prefix_cols], 1, function(row) {
  row     <- as.character(row)
  present <- na.omit(row)
  out     <- rep(0L, length(prefixes))
  names(out) <- prefixes
  for (i in seq_along(present)) {
    if (present[i] %in% prefixes) out[present[i]] <- i
  }
  out
}))

n_present   <- rowSums(rank_matrix > 0)
rank_matrix <- rank_matrix[n_present >= 2, ]

# ----------------------------------------------------------------------------
# 19. Construct rankings object and sanity-check
# ----------------------------------------------------------------------------
R <- as.rankings(rank_matrix, input = "ranking")

stopifnot("Rankings contain ties -- check rank_matrix construction" =
            !any(grepl("=", format(head(R)))))

always_first <- apply(R, 2, function(x) all(x[x > 0] == 1))
always_last  <- apply(R, 2, function(x) {
  vals <- x[x > 0]; all(vals == max(vals))
})
if (any(always_first)) warning("Perfect separation (always first): ",
                               paste(names(which(always_first)),
                                     collapse = ", "))
if (any(always_last))  warning("Perfect separation (always last): ",
                               paste(names(which(always_last)),
                                     collapse = ", "))

cat("Item frequencies:\n")
print(colSums(R > 0))

# ----------------------------------------------------------------------------
# 20. Fit Plackett-Luce model
# ----------------------------------------------------------------------------
pl_model <- PlackettLuce(R)
summary(pl_model)

# ----------------------------------------------------------------------------
# 21. Extract worth and correlate with informational measures
# ----------------------------------------------------------------------------
# Worth is on a multiplicative scale; higher worth = stronger pull leftward.
# With n = 6 prefixes, the minimum achievable two-tailed Spearman p-value
# is 0.083, so rho and its direction are the primary result.

worth_df <- data.frame(
  Prefix = prefixes,
  worth  = coef(pl_model, log = FALSE)
) %>%
  left_join(info_measures %>% select(Prefix, ICTRANS_MM, ICLOCAL),
            by = "Prefix") %>%
  arrange(desc(worth))

print(worth_df)

cat("\n--- PL worth ~ ICTRANS_MM (Spearman) ---\n")
print(cor.test(worth_df$worth, worth_df$ICTRANS_MM, method = "spearman"))

cat("\n--- PL worth ~ ICLOCAL (Spearman) ---\n")
print(cor.test(worth_df$worth, worth_df$ICLOCAL, method = "spearman"))

# ----------------------------------------------------------------------------
# 22. Identify deviations from the informational model
# ----------------------------------------------------------------------------
# Flag prefixes whose worth rank diverges substantially from their CE/PS rank.

worth_df <- worth_df %>%
  mutate(
    rank_worth   = rank(-worth),
    rank_ICTRANS = rank(-ICTRANS_MM),
    rank_ICLOCAL = rank(ICLOCAL),
    dev_ICTRANS  = rank_worth - rank_ICTRANS,
    dev_ICLOCAL  = rank_worth - rank_ICLOCAL
  )

cat("\n--- Worth rank vs. informational rank deviations ---\n")
print(worth_df %>% select(Prefix, rank_worth, rank_ICTRANS,
                          rank_ICLOCAL, dev_ICTRANS, dev_ICLOCAL))

# ============================================================================
# TRANSITIVITY ANALYSIS
# ============================================================================
# Tests whether the 15 pairwise ordering proportions form a strict total
# order. If all triples are transitive and the best-fitting ranking is
# consistent with 100% of attested pairs, the six prefixes occupy fixed
# positions in a single global hierarchy.
#
# Three weighting schemes are compared:
#   (a) Unweighted: every pair counts equally regardless of token count
#       or preference strength.
#   (b) Magnitude-weighted: pairs with stronger ordering preferences
#       (|p - 0.5| closer to 0.5) contribute more. Reflects how
#       informative each pair is about the underlying hierarchy.
#   (c) Combined-weighted: magnitude × token count. Rewards both a
#       strong preference and a reliable estimate of that preference.
#       Closest in spirit to what a Bradley-Terry likelihood does.
#
# If all three schemes recover the same ranking, the total order is robust
# to the choice of weighting.
# ============================================================================

# ----------------------------------------------------------------------------
# 23. Build the dominance matrix
# ----------------------------------------------------------------------------
# Entry [i, j] = proportion of tokens where prefix i appears leftward
# of prefix j. Built from pair_counts, which already contains the
# direction-consistent ordering proportions.
#
# The matrix is antisymmetric by construction: if dom_matrix[i,j] = p,
# then dom_matrix[j,i] = 1 - p. Diagonal entries are set to 0.5
# (a prefix does not compete with itself).

dom_matrix <- matrix(NA,
                     nrow = 6, ncol = 6,
                     dimnames = list(prefixes, prefixes))

for (i in seq_along(prefixes)) {
  for (j in seq_along(prefixes)) {
    
    if (i == j) {
      dom_matrix[i, j] <- 0.5
      next
    }
    
    pi <- prefixes[i]
    pj <- prefixes[j]
    
    pair_row <- pair_counts %>%
      filter(
        (prefix_A == pi & prefix_B == pj) |
          (prefix_A == pj & prefix_B == pi)
      )
    
    if (nrow(pair_row) == 0) {
      dom_matrix[i, j] <- NA
      next
    }
    
    if (pair_row$prefix_A == pi) {
      dom_matrix[i, j] <- pair_row$prop_A_left
    } else {
      dom_matrix[i, j] <- 1 - pair_row$prop_A_left
    }
  }
}

cat("\n--- Dominance matrix (P(row prefix appears leftward)) ---\n")
print(round(dom_matrix, 3))

# Also build a token-count matrix for use in weighted analyses below
# Entry [i, j] = number of tokens in the pair {i, j}

token_matrix <- matrix(NA,
                       nrow = 6, ncol = 6,
                       dimnames = list(prefixes, prefixes))

for (i in seq_along(prefixes)) {
  for (j in seq_along(prefixes)) {
    
    if (i == j) {
      token_matrix[i, j] <- 0
      next
    }
    
    pi <- prefixes[i]
    pj <- prefixes[j]
    
    pair_row <- pair_counts %>%
      filter(
        (prefix_A == pi & prefix_B == pj) |
          (prefix_A == pj & prefix_B == pi)
      )
    
    token_matrix[i, j] <- if (nrow(pair_row) > 0) pair_row$n_total else NA
  }
}

# ----------------------------------------------------------------------------
# 24. Transitivity test across all 20 triples
# ----------------------------------------------------------------------------
# For each triple {A, B, C}, checks whether the three pairwise ordering
# proportions are mutually consistent with some linear ordering of A, B, C.
# A violation (Condorcet cycle) occurs when A > B and B > C but C > A,
# making it impossible to rank the three consistently.

triples <- combn(prefixes, 3, simplify = FALSE)

transitivity_results <- map_df(triples, function(triple) {
  A <- triple[1]; B <- triple[2]; C <- triple[3]
  
  pAB <- dom_matrix[A, B]
  pBC <- dom_matrix[B, C]
  pAC <- dom_matrix[A, C]
  
  if (any(is.na(c(pAB, pBC, pAC)))) {
    return(tibble(
      triple     = paste(A, B, C, sep = "-"),
      pAB        = pAB, pBC = pBC, pAC = pAC,
      AB_beats   = NA, BC_beats = NA, AC_beats = NA,
      transitive = NA,
      note       = "unattested pair"
    ))
  }
  
  AB_beats <- pAB > 0.5
  BC_beats <- pBC > 0.5
  AC_beats <- pAC > 0.5
  
  transitive <- !((AB_beats  & BC_beats  & !AC_beats) |
                    (!AB_beats & !BC_beats &  AC_beats))
  
  tibble(
    triple     = paste(A, B, C, sep = "-"),
    pAB        = round(pAB, 3),
    pBC        = round(pBC, 3),
    pAC        = round(pAC, 3),
    AB_beats   = AB_beats,
    BC_beats   = BC_beats,
    AC_beats   = AC_beats,
    transitive = transitive,
    note       = ifelse(transitive, "consistent", "VIOLATION")
  )
})

cat("\n--- Transitivity results for all 20 triples ---\n")
print(transitivity_results, n = Inf)

cat(sprintf(
  "\nTransitive triples: %d / %d (%.1f%%)\n",
  sum(transitivity_results$transitive, na.rm = TRUE),
  sum(!is.na(transitivity_results$transitive)),
  100 * mean(transitivity_results$transitive, na.rm = TRUE)
))

# ----------------------------------------------------------------------------
# 25. Win-count ranking (unweighted)
# ----------------------------------------------------------------------------
# For each prefix, count how many of the other five it tends to precede
# (dominance proportion > 0.5). Ties in win count are broken by
# mean_dominance (the average proportion of leftward appearances across
# all pairs the prefix participates in).

ranking_scores <- tibble(
  prefix = prefixes,
  wins   = map_dbl(prefixes, function(p) {
    sum(dom_matrix[p, prefixes != p] > 0.5, na.rm = TRUE)
  }),
  mean_dominance = map_dbl(prefixes, function(p) {
    mean(dom_matrix[p, prefixes != p], na.rm = TRUE)
  })
) %>%
  arrange(desc(wins), desc(mean_dominance))

cat("\n--- Win-count ranking (unweighted) ---\n")
print(ranking_scores)

# ----------------------------------------------------------------------------
# 26. Best-fitting total order: three weighting schemes
# ----------------------------------------------------------------------------
# For each of the 720 possible orderings of the six prefixes, a consistency
# score is computed. The ordering with the highest score is the best-fitting
# total order under each scheme.
#
# Scheme (a): UNWEIGHTED
#   Score = (number of correctly predicted pair directions) /
#           (total attested pairs)
#   Simple but treats all pairs equally.
#
# Scheme (b): MAGNITUDE-WEIGHTED
#   Score = sum of |p - 0.5| for correctly predicted pairs /
#           sum of |p - 0.5| for all attested pairs
#   Weight = |p - 0.5|: zero for a 50-50 split (uninformative),
#   0.5 for a 100-0 split (maximally informative). A pair where one
#   prefix almost always goes left tells us much more about the
#   underlying hierarchy than a near-balanced pair.
#
# Scheme (c): COMBINED-WEIGHTED (magnitude × token count)
#   Score = sum of n_ij * |p - 0.5| for correctly predicted pairs /
#           sum of n_ij * |p - 0.5| for all attested pairs
#   Rewards both a strong preference AND a reliable estimate of it.
#   This is the closest non-parametric analog to what Bradley-Terry
#   maximum likelihood estimation does: a 99-1 split from 88 tokens
#   contributes far more than a 55-45 split from 3 tokens.

all_rankings <- permn(prefixes)

# --- Helper: compute consistency score under a given weight function ---
compute_consistency <- function(rankings, dom_mat, token_mat,
                                weight_fn = "unweighted") {
  map_dbl(rankings, function(ranking) {
    w_consistent <- 0
    w_total      <- 0
    
    for (i in 1:5) {
      for (j in (i + 1):6) {
        left  <- ranking[i]
        right <- ranking[j]
        p     <- dom_mat[left, right]
        n     <- token_mat[left, right]
        
        if (is.na(p)) next
        
        w <- switch(weight_fn,
                    "unweighted" = 1,
                    "magnitude"  = abs(p - 0.5),
                    "combined"   = ifelse(is.na(n), 0, n * abs(p - 0.5))
        )
        
        w_total      <- w_total      + w
        w_consistent <- w_consistent + (p > 0.5) * w
      }
    }
    
    if (w_total == 0) return(NA)
    w_consistent / w_total
  })
}

# --- Compute scores under all three schemes ---
scores_unweighted <- compute_consistency(
  all_rankings, dom_matrix, token_matrix, "unweighted")
scores_magnitude  <- compute_consistency(
  all_rankings, dom_matrix, token_matrix, "magnitude")
scores_combined   <- compute_consistency(
  all_rankings, dom_matrix, token_matrix, "combined")

# --- Extract best-fitting ranking under each scheme ---
extract_best <- function(scores, rankings, label) {
  best_idx     <- which.max(scores)
  best_ranking <- rankings[[best_idx]]
  best_score   <- scores[best_idx]
  n_tied       <- sum(scores == best_score, na.rm = TRUE)
  
  cat(sprintf(
    "\n--- Best-fitting total order (%s) ---\n  %s\n  Score: %.4f\n  Rankings achieving maximum: %d / 720\n",
    label,
    paste(best_ranking, collapse = " > "),
    best_score,
    n_tied
  ))
  
  list(ranking = best_ranking, score = best_score, n_tied = n_tied)
}

best_unweighted <- extract_best(scores_unweighted, all_rankings,
                                "unweighted")
best_magnitude  <- extract_best(scores_magnitude,  all_rankings,
                                "magnitude-weighted")
best_combined   <- extract_best(scores_combined,   all_rankings,
                                "combined-weighted")

# --- Summary comparison of rankings across schemes ---
cat("\n--- Ranking stability across weighting schemes ---\n")
ranking_stability <- tibble(
  scheme     = c("Unweighted",
                 "Magnitude-weighted",
                 "Combined-weighted"),
  ranking    = c(paste(best_unweighted$ranking, collapse = " > "),
                 paste(best_magnitude$ranking,  collapse = " > "),
                 paste(best_combined$ranking,   collapse = " > ")),
  score      = c(best_unweighted$score,
                 best_magnitude$score,
                 best_combined$score),
  n_tied     = c(best_unweighted$n_tied,
                 best_magnitude$n_tied,
                 best_combined$n_tied)
)
print(ranking_stability)

# Check whether all three schemes agree on the same ranking
schemes_agree <- length(unique(ranking_stability$ranking)) == 1
cat(sprintf(
  "\nAll three schemes recover the same ranking: %s\n",
  ifelse(schemes_agree, "YES -- robust result", "NO -- check disagreements")
))

# If they disagree, identify which positions differ
if (!schemes_agree) {
  cat("\nPosition-by-position comparison:\n")
  pos_comparison <- tibble(
    position   = 1:6,
    unweighted = best_unweighted$ranking,
    magnitude  = best_magnitude$ranking,
    combined   = best_combined$ranking
  ) %>%
    mutate(
      agrees = (unweighted == magnitude) & (magnitude == combined),
      note   = ifelse(agrees, "stable", "DIFFERS")
    )
  print(pos_comparison)
}

# ----------------------------------------------------------------------------
# 27. Compare the empirical ranking to CE and PS rankings
# ----------------------------------------------------------------------------
# Spearman correlations between the empirical win-count ranking and the
# CE- and PS-derived rankings. Repeated with arci- excluded to assess
# how much of the misalignment is driven by that prefix alone.
#
# Note: rank_PS is computed as rank(ICLOCAL) because lower PS (lower
# surprisal = more frequent) corresponds to more leftward placement, so
# the PS rank and the empirical rank should correlate positively when
# defined this way.

ranking_comparison <- ranking_scores %>%
  left_join(info_measures %>% select(Prefix, ICTRANS_MM, ICLOCAL),
            by = c("prefix" = "Prefix")) %>%
  mutate(
    rank_empirical = rank(-wins),
    rank_CE        = rank(-ICTRANS_MM),
    rank_PS        = rank(ICLOCAL)
  )

cat("\n--- Ranking comparison (empirical vs. CE vs. PS) ---\n")
print(ranking_comparison %>%
        select(prefix, rank_empirical, rank_CE, rank_PS,
               wins, ICTRANS_MM, ICLOCAL))

cat("\n--- Spearman: empirical ~ CE (all prefixes) ---\n")
print(cor.test(ranking_comparison$rank_empirical,
               ranking_comparison$rank_CE,
               method = "spearman"))

cat("\n--- Spearman: empirical ~ PS (all prefixes) ---\n")
print(cor.test(ranking_comparison$rank_empirical,
               ranking_comparison$rank_PS,
               method = "spearman"))

# Without arci-: both correlations should strengthen to ~0.90
rc_no_arci <- ranking_comparison %>% filter(prefix != "arci")
cat("\n--- Spearman: empirical ~ CE and PS (excluding arci-) ---\n")
cat(sprintf(
  "  rho_CE = %.3f\n  rho_PS = %.3f\n",
  cor(rc_no_arci$rank_empirical,
      rc_no_arci$rank_CE, method = "spearman"),
  cor(rc_no_arci$rank_empirical,
      rc_no_arci$rank_PS, method = "spearman")
))

# ----------------------------------------------------------------------------
# 28. Per-pair weight summary (diagnostic)
# ----------------------------------------------------------------------------
# Shows, for each attested pair, what weight each scheme assigns.
# Useful for understanding which pairs are driving the weighted rankings
# and which contribute little (sparse pairs with near-balanced proportions).

weight_summary <- pair_counts %>%
  mutate(
    p_left        = prop_A_left,
    magnitude_w   = abs(p_left - 0.5),
    combined_w    = n_total * abs(p_left - 0.5),
    informativeness = case_when(
      abs(p_left - 0.5) >= 0.30 ~ "high (>80-20 split)",
      abs(p_left - 0.5) >= 0.15 ~ "moderate (65-35 to 80-20)",
      TRUE                      ~ "low (<65-35 split)"
    )
  ) %>%
  select(pair, n_total, p_left, magnitude_w, combined_w,
         informativeness) %>%
  arrange(desc(combined_w))

cat("\n--- Per-pair weight summary ---\n")
print(weight_summary, n = Inf)

# ============================================================================
# PLOTS
# ============================================================================

# Shared theme
theme_slides <- theme_minimal(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title       = element_text(face = "bold", size = 16),
    plot.subtitle    = element_text(size = 13, color = "grey30"),
    legend.position  = "bottom",
    axis.title       = element_text(size = 14)
  )

arci_palette <- c("FALSE" = "darkgoldenrod", "TRUE" = "firebrick4")

plot_dir <- file.path(folder_path, "plots")
dir.create(plot_dir, showWarnings = FALSE)

# ----------------------------------------------------------------------------
# Plot 1: Prefix informational profile
# ----------------------------------------------------------------------------
p1 <- ggplot(info_measures, aes(x = ICTRANS_MM, y = ICLOCAL)) +
  geom_smooth(method = "lm", se = FALSE,
              color = "grey70", linetype = "dashed", linewidth = 0.5) +
  geom_point(size = 4, color = "steelblue") +
  geom_text_repel(aes(label = Prefix),
                  size = 5, fontface = "italic",
                  box.padding = 0.6, point.padding = 0.4) +
  labs(
    title    = "Informational profile of the six intensifying prefixes",
    subtitle = expression(paste("Spearman ", rho, " = -0.77")),
    x        = "Contextual entropy (CE, bits)",
    y        = "Paradigmatic surprisal (PS, bits)"
  ) +
  theme_slides

ggsave(file.path(plot_dir, "01_prefix_profile.pdf"),
       p1, width = 7, height = 5.2, device = cairo_pdf)

# ----------------------------------------------------------------------------
# Plot 2: Model-fit plot (observed vs. predicted pair proportions)
# ----------------------------------------------------------------------------
p2 <- ggplot(pair_residuals,
             aes(x = predicted, y = observed,
                 size = n_total, color = has_arci)) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "grey50") +
  geom_point(alpha = 1) +
  geom_text_repel(aes(label = pair),
                  size = 9, show.legend = FALSE,
                  box.padding = 0.8, max.overlaps = 20) +
  scale_size_continuous(range = c(2, 10), name = "Tokens") +
  scale_color_manual(values = arci_palette,
                     labels = c("FALSE" = "non-arci", "TRUE" = "arci-"),
                     name   = NULL) +
  guides(color = guide_legend(override.aes = list(size = 6))) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "Model fit: observed vs. predicted pair proportions",
    x = "Predicted proportion",
    y = "Observed proportion"
  ) +
  theme_slides

ggsave(file.path(plot_dir, "02_model_fit.pdf"),
       p2, width = 11, height = 10, device = cairo_pdf)

# ----------------------------------------------------------------------------
# Plot 3: Coefficient plot for the joint model
# ----------------------------------------------------------------------------
coefs <- broom.mixed::tidy(m_both, effects = "fixed", conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    term = recode(term,
                  "dICTRANS_z" = "Delta*CE",
                  "dICLOCAL_z" = "Delta*PS")
  )

p3 <- ggplot(coefs, aes(x = estimate, y = term)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high),
                 height = 0.15, linewidth = 0.8, color = "steelblue") +
  geom_point(size = 4, color = "steelblue") +
  scale_y_discrete(labels = scales::parse_format()) +
  labs(
    title    = "Fixed-effect coefficients (joint model)",
    subtitle = "Predicting P(A on left); 95% Wald confidence intervals",
    x        = "Estimate (log-odds, standardized predictors)",
    y        = NULL
  ) +
  theme_slides

ggsave(file.path(plot_dir, "03_coefficients.pdf"),
       p3, width = 7, height = 4, device = cairo_pdf)

# ----------------------------------------------------------------------------
# Plot 4: Pair-level proportions with sample sizes
# ----------------------------------------------------------------------------
pair_counts_plot <- pair_counts %>%
  mutate(
    pair_label = paste0(prefix_A, "-", prefix_B),
    has_arci   = grepl("arci", pair),
    pair_label = reorder(pair_label, prop_A_left)
  )

p4 <- ggplot(pair_counts_plot,
             aes(x = pair_label, y = prop_A_left, fill = has_arci)) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50") +
  geom_col(alpha = 0.85, width = 0.7) +
  geom_text(aes(label = paste0("n=", n_total)),
            hjust = -0.15, size = 3.3) +
  scale_fill_manual(values = arci_palette,
                    labels = c("FALSE" = "non-arci", "TRUE" = "arci-"),
                    name   = NULL) +
  coord_flip(ylim = c(0, 1.08)) +
  scale_y_continuous(breaks = seq(0, 1, 0.25)) +
  labs(
    title    = "Ordering proportions by prefix pair",
    subtitle = "Proportion of tokens where the alphabetically-first prefix appears leftward",
    x        = NULL,
    y        = "P(A on left)"
  ) +
  theme_slides

ggsave(file.path(plot_dir, "04_pair_proportions.pdf"),
       p4, width = 8, height = 6, device = cairo_pdf)

# ----------------------------------------------------------------------------
# Plot 5: Adjacency comparison
# ----------------------------------------------------------------------------
p5 <- ggplot(stacked_model,
             aes(x = dICTRANS_z, y = observed, color = adjacency)) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"),
              se = TRUE, alpha = 0.15, linewidth = 1) +
  geom_jitter(width = 0, height = 0.04, alpha = 0.25, size = 1.5) +
  scale_color_manual(values = c("adjacent"     = "steelblue",
                                "non_adjacent" = "darkorange"),
                     labels = c("adjacent"     = "Adjacent",
                                "non_adjacent" = "Non-adjacent (3-1)"),
                     name   = NULL) +
  labs(
    title    = "Informational gradient: adjacent vs. non-adjacent pairs",
    subtitle = expression(paste(
      "No significant interaction: ",
      chi^2, "(2) = 1.52, ", italic(p), " = .467")),
    x = expression(paste(Delta, "CE (z-scored)")),
    y = "P(A on left)"
  ) +
  theme_slides

ggsave(file.path(plot_dir, "05_adjacency.pdf"),
       p5, width = 7, height = 5, device = cairo_pdf)

# ----------------------------------------------------------------------------
# Plot 6: Total order visualization
# ----------------------------------------------------------------------------
# Displays the six prefixes arranged along a horizontal axis according to
# their empirical ranking (left = most-leftward, right = most-rightward),
# with point size proportional to mean dominance and color indicating
# whether the prefix's empirical rank matches its CE rank.
# arci- is highlighted as the structural exception.

total_order_data <- ranking_comparison %>%
  mutate(
    rank_label   = paste0(prefix, "-"),
    CE_rank_diff = rank_empirical - rank_CE,
    is_arci      = prefix == "arci"
  )

# Place prefixes along x-axis by empirical rank, CE rank on y-axis
# so misalignment from the diagonal reveals divergence

p6 <- ggplot(total_order_data,
             aes(x = rank_CE, y = rank_empirical)) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "grey50") +
  geom_point(aes(size = mean_dominance, color = is_arci),
             alpha = 0.9) +
  geom_text_repel(aes(label = paste0("italic('", prefix, "-')")),
                  parse        = TRUE,
                  size         = 5,
                  box.padding  = 0.6,
                  show.legend  = FALSE) +
  scale_size_continuous(range = c(4, 10),
                        name  = "Mean dominance") +
  scale_color_manual(values = c("FALSE" = "steelblue",
                                "TRUE"  = "firebrick4"),
                     labels = c("FALSE" = "other prefixes",
                                "TRUE"  = "arci-"),
                     name   = NULL) +
  scale_x_continuous(breaks = 1:6,
                     labels = c("1st\n(most leftward)",
                                "2nd", "3rd", "4th", "5th",
                                "6th\n(most rightward)")) +
  scale_y_continuous(breaks = 1:6,
                     labels = c("1st", "2nd", "3rd",
                                "4th", "5th", "6th")) +
  labs(
    title    = "Empirical vs. CE-derived ranking",
    subtitle = expression(paste(
      rho, " = 0.54 (all prefixes); ",
      rho, " = 0.90 (excluding ", italic("arci-"), ")")),
    x = "CE rank (1 = highest CE)",
    y = "Empirical rank (1 = most leftward)"
  ) +
  theme_slides +
  theme(legend.position = "bottom")

ggsave(file.path(plot_dir, "06_total_order.pdf"),
       p6, width = 7, height = 6, device = cairo_pdf)

# Display all plots
print(p1); print(p2); print(p3)
print(p4); print(p5); print(p6)

cat(sprintf("\nAll plots saved to: %s\n", plot_dir))

# ----------------------------------------------------------------------------
# END
# ----------------------------------------------------------------------------