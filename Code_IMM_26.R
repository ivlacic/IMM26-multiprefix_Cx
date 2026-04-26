# ============================================================================
# Informational Predictors of Prefix Linearization in Italian Intensifiers
# Author: Ivan Lacić
# Date: April, 2026
# ============================================================================
#
# This script analyzes the linearization of stacked intensifying prefixes
# (arci-, extra-, iper-, stra-, super-, ultra-) in Italian multi-prefix
# constructions. It tests whether two information-theoretic measures --
# contextual entropy (ICTRANS) and paradigmatic surprisal (ICLOCAL) --
# predict the order in which prefixes appear when stacked.
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
#
# Inputs (on ~/Desktop by default):
#   - <prefix>_intensifiers.csv for each of the six prefixes
#   - prefix_stackings.xlsx     (multi-prefix construction tokens)
#
# Outputs:
#   - prefix_information_values_MillerMadow.csv
#   - pair_counts.csv
# ============================================================================

# ----------------------------------------------------------------------------
# 0. Libraries
# ----------------------------------------------------------------------------
library(dplyr)
library(tidyr)
library(readr)
library(readxl)
library(purrr)
library(lme4)
library(broom.mixed)

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
# lexicalized and would distort the stra- base distribution.

all_data <- prefixes %>%
  map_df(function(pref) {
    file <- file.path(folder_path, paste0(pref, "_intensifiers.csv"))
    df <- read_csv(file, show_col_types = FALSE)
    df$Prefix <- pref
    df
  }) %>%
  filter(!(Prefix == "stra" & Word == "grande"))   # drop lexicalized stragrande

head(all_data)   # expected columns: Word | Prefix


# ----------------------------------------------------------------------------
# 3. Compute ICTRANS (contextual entropy of bases given prefix)
# ----------------------------------------------------------------------------
# For each prefix, we treat the distribution of bases as a probability
# distribution and compute its Shannon entropy. High ICTRANS means the
# prefix attaches to many different bases relatively evenly -- a marker
# of morphological productivity (Baayen 2009).
#
# The Miller-Madow correction (k-1)/(2*N) nats, converted here to bits
# by dividing by ln(2), corrects the well-known downward bias of the
# plug-in entropy estimator on finite samples.

freq_table <- all_data %>%
  group_by(Prefix, Word) %>%
  summarise(Freq = n(), .groups = "drop")

ICTRANS <- freq_table %>%
  group_by(Prefix) %>%
  mutate(P_base_given_prefix = Freq / sum(Freq)) %>%
  summarise(
    ICTRANS_raw  = -sum(P_base_given_prefix * log2(P_base_given_prefix)),
    prefix_total = sum(Freq),
    k            = n(),                     # number of distinct bases
    .groups = "drop"
  ) %>%
  mutate(
    MM_correction = (k - 1) / (2 * prefix_total * log(2)),   # bits
    ICTRANS_MM    = ICTRANS_raw + MM_correction
  )


# ----------------------------------------------------------------------------
# 4. Compute ICLOCAL (Shannon surprisal of each prefix overall)
# ----------------------------------------------------------------------------
# ICLOCAL = -log2(P(prefix)), where P(prefix) is the prefix's share of the
# pooled prefix-token corpus. Rarer prefixes have higher surprisal and
# are interpretable as more expressively marked.

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
# The surprisal-memory trade-off predicts a negative correlation:
# more productive (high-ICTRANS) prefixes should be less marked (low ICLOCAL).

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


# ----------------------------------------------------------------------------
# 7. Build pairwise contrasts
# ----------------------------------------------------------------------------
# Two kinds of pairs:
#   - Adjacent pairs: consecutive positions (prefix_1 - prefix_2,
#                     prefix_2 - prefix_3, etc.)
#   - Non-adjacent 3-1 pairs: outermost vs. innermost when there are
#                             at least 3 stacked prefixes.
#
# Each pair is tagged with an `adjacency` column so that its role can
# be tested later.

stacking_pairs_adjacent <- stackings %>%
  mutate(across(starts_with("prefix_"), as.character)) %>%
  pivot_longer(
    cols      = starts_with("prefix_"),
    names_to  = "pos",
    values_to = "prefix"
  ) %>%
  arrange(TYPE, pos) %>%
  group_by(TYPE) %>%
  mutate(next_prefix = lead(prefix)) %>%
  ungroup() %>%
  filter(!is.na(prefix), !is.na(next_prefix)) %>%
  rename(prefix_left = prefix, prefix_right = next_prefix) %>%
  mutate(adjacency = "adjacent") %>%
  select(TYPE, base, prefix_left, prefix_right, adjacency)

stacking_pairs_nonadjacent <- stackings %>%
  mutate(across(starts_with("prefix_"), as.character)) %>%
  filter(!is.na(prefix_3)) %>%           # only stacks with >= 3 prefixes
  transmute(
    TYPE,
    base,
    prefix_left  = prefix_1,
    prefix_right = prefix_3,
    adjacency    = "non_adjacent"
  )

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
    dICLOCAL = ICLOCAL_left   - ICLOCAL_right
  )


# ----------------------------------------------------------------------------
# 9. Descriptive tests on raw left-right differences
# ----------------------------------------------------------------------------
# Simple sanity checks: what proportion of pairs show the predicted
# direction, and are the two differences correlated across pairs?

mean(stacked_full$dICTRANS > 0)   # share of pairs where left has higher ICTRANS
mean(stacked_full$dICLOCAL < 0)   # share of pairs where right has higher ICLOCAL
cor.test(stacked_full$dICTRANS, stacked_full$dICLOCAL, method = "spearman")


# ============================================================================
# MIXED-EFFECTS PIPELINE
# ============================================================================

# ----------------------------------------------------------------------------
# 10. Direction-consistent recoding
# ----------------------------------------------------------------------------
# The descriptive d-variables above are coded relative to the OBSERVED
# order (left minus right), which would make them trivially predict the
# outcome. For inferential modeling, we recode every pair into a fixed,
# order-independent reference frame:
#
#   For each unordered pair {A, B}:
#     A = alphabetically-first prefix
#     B = alphabetically-second prefix
#     observed = 1 if the attested left prefix is A, else 0
#     dICTRANS = ICTRANS(A) - ICTRANS(B)    (always A minus B)
#     dICLOCAL = ICLOCAL(A) - ICLOCAL(B)
#
# The model then asks: does the informational difference between A and B
# (measured in this fixed frame) predict whether A ends up on the left?

stacked_model <- stacked_full %>%
  filter(prefix_left != prefix_right) %>%    # drop same-prefix pairs
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
# For each unordered pair, report total tokens, the split between the two
# possible orderings, and the fixed informational differences. This also
# shows how concentrated the data is in a few well-attested pairs.

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
# Random intercepts for:
#   - base (repeated adjectival bases like 'bello')
#   - pair (repeated unordered prefix pairs)
#
# Predictors are z-standardized to aid convergence and make coefficients
# comparable in magnitude.

stacked_model <- stacked_model %>%
  mutate(
    dICTRANS_z = as.numeric(scale(dICTRANS)),
    dICLOCAL_z = as.numeric(scale(dICLOCAL))
  )

ctrl <- glmerControl(optimizer = "bobyqa",
                     optCtrl   = list(maxfun = 2e5))

# Null model: intercept + random effects only
m_null <- glmer(observed ~ 1 + (1 | base) + (1 | pair),
                data = stacked_model, family = binomial, control = ctrl)

# Univariate models (each predictor alone)
m_ce <- glmer(observed ~ dICTRANS_z + (1 | base) + (1 | pair),
              data = stacked_model, family = binomial, control = ctrl)

m_ps <- glmer(observed ~ dICLOCAL_z + (1 | base) + (1 | pair),
              data = stacked_model, family = binomial, control = ctrl)

# Joint model (both predictors)
# NOTE: a singular-fit warning is expected here because the fixed effects
# absorb virtually all pair-level variance -- the (1|pair) term therefore
# has nothing left to explain. We keep the random effect for transparency.
m_both <- glmer(observed ~ dICTRANS_z + dICLOCAL_z + (1 | base) + (1 | pair),
                data = stacked_model, family = binomial, control = ctrl)


# ----------------------------------------------------------------------------
# 13. Model summaries and likelihood-ratio tests
# ----------------------------------------------------------------------------
summary(m_ce)
summary(m_ps)
summary(m_both)

cat("\n--- LRT: null vs. dCE-only ---\n")
print(anova(m_null, m_ce))

cat("\n--- LRT: null vs. dPS-only ---\n")
print(anova(m_null, m_ps))

cat("\n--- LRT: dCE-only vs. joint (does dPS add anything?) ---\n")
print(anova(m_ce, m_both))

cat("\n--- LRT: dPS-only vs. joint (does dCE add anything?) ---\n")
print(anova(m_ps, m_both))

cat("\n--- Joint model coefficients (with 95% Wald CIs) ---\n")
print(broom.mixed::tidy(m_both, effects = "fixed", conf.int = TRUE))


# ----------------------------------------------------------------------------
# 14. Adjacency interaction (adjacent vs. non-adjacent 3-1 pairs)
# ----------------------------------------------------------------------------
# Tests whether the informational gradient operates differently for
# locally adjacent vs. distant (outermost-innermost) prefix relationships.
# Only runs if both adjacency levels are present in the data.

if (length(unique(stacked_model$adjacency)) > 1) {
  m_adj <- glmer(
    observed ~ (dICTRANS_z + dICLOCAL_z) * adjacency +
      (1 | base) + (1 | pair),
    data = stacked_model, family = binomial, control = ctrl
  )
  cat("\n--- Adjacency interaction model ---\n")
  print(summary(m_adj))
  cat("\n--- LRT: joint vs. joint + adjacency interaction ---\n")
  print(anova(m_both, m_adj))
}


# ----------------------------------------------------------------------------
# 15. Per-pair sanity check (aggregated, one row per pair)
# ----------------------------------------------------------------------------
# Correlates prop_A_left with dICTRANS and dICLOCAL across the ~15
# unordered pairs. This analysis is immune to within-pair repeated
# observations because each pair contributes a single data point, but
# it is also dominated by a few extreme sparse pairs -- interpret as a
# complement to, not a substitute for, the mixed model above.

cat("\n--- Per-pair Spearman correlations ---\n")
print(cor.test(pair_counts$prop_A_left, pair_counts$dICTRANS,
               method = "spearman"))
print(cor.test(pair_counts$prop_A_left, pair_counts$dICLOCAL,
               method = "spearman"))


# ----------------------------------------------------------------------------
# 16. Residual diagnostics
# ----------------------------------------------------------------------------
# Aggregate predicted vs. observed proportions by pair and inspect
# residuals. Large residuals flag pairs that the informational model
# fails to capture; we then check whether such residuals are driven by
# sample size (log token count) or by a specific prefix.

stacked_model$pred <- predict(m_both, type = "response")

pair_residuals <- stacked_model %>%
  group_by(pair) %>%
  summarise(
    n_total   = n(),
    observed  = mean(observed),
    predicted = mean(pred),
    residual  = mean(observed) - mean(pred),
    .groups   = "drop"
  ) %>%
  arrange(desc(abs(residual)))

print(pair_residuals)

pair_residuals <- pair_residuals %>%
  mutate(
    has_arci = grepl("arci", pair),
    log_n    = log(n_total)
  )

# Raw (marginal) association between residual size and sample size
cor.test(abs(pair_residuals$residual), pair_residuals$log_n)

# Fit-quality summary for non-arci pairs
pair_residuals %>%
  filter(!has_arci) %>%
  summarise(max_abs_resid  = max(abs(residual)),
            mean_abs_resid = mean(abs(residual)))

# Does the arci- effect survive after controlling for sample size?
#   - has_arci on |residual|: tests whether arci- pairs are more poorly fit
#   - has_arci on signed residual: tests whether arci- pairs are
#     systematically biased (positive = arci- goes left more than predicted)
cat("\n--- Absolute residual ~ log_n + has_arci ---\n")
print(summary(lm(abs(residual) ~ log_n + has_arci, data = pair_residuals)))

cat("\n--- Signed residual ~ log_n + has_arci ---\n")
print(summary(lm(residual ~ log_n + has_arci, data = pair_residuals)))

# ----------------------------------------------------------------------------
# 17. Bootstrap CI for the non-arci mean absolute residual
# ----------------------------------------------------------------------------
# The headline fit statistic (mean |residual| = 0.045 for non-arci pairs)
# is a point estimate. To quantify uncertainty, we bootstrap at the
# TOKEN level within non-arci pairs: each replicate resamples tokens
# with replacement, recomputes each pair's observed proportion and
# residual, then averages |residual| across pairs.
#
# Bootstrapping tokens (not pairs) is appropriate here because the
# uncertainty we want to capture is sampling variability in the
# observed proportions, not in which pairs exist.

set.seed(123)
n_boot <- 2000

# Restrict to non-arci pair-tokens; keep the predicted probability already
# attached from the joint model in section 16.
boot_data <- stacked_model %>%
  mutate(has_arci = grepl("arci", pair)) %>%
  filter(!has_arci)

# One bootstrap replicate: resample tokens within each pair, recompute
# observed proportion and residual per pair, return mean |residual|.
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

boot_ci <- quantile(boot_estimates, probs = c(0.025, 0.5, 0.975))

cat(sprintf(
  "\nNon-arci mean |residual|: point estimate = %.4f\n  95%% bootstrap CI: [%.4f, %.4f] (median = %.4f)\n  Based on %d bootstrap replicates\n",
  mean(abs(pair_residuals$residual[!pair_residuals$has_arci])),
  boot_ci[1], boot_ci[3], boot_ci[2], n_boot
))

# Optional: full bootstrap distribution for reporting/plotting
boot_summary <- data.frame(
  estimate      = mean(abs(pair_residuals$residual[!pair_residuals$has_arci])),
  ci_lower_2.5  = boot_ci[1],
  median_50     = boot_ci[2],
  ci_upper_97.5 = boot_ci[3],
  n_boot        = n_boot
)
print(boot_summary)

# ============================================================================
# PLOTS
# ============================================================================
# Fiveplots covering the main findings:
#   1. Prefix informational profile (CE vs. PS scatter)
#   2. Observed vs. predicted pair proportions (model-fit plot)
#   3. Coefficient plot for the joint mixed-effects model
#   4. Pair-level proportions with sample sizes (ordered by prop_A_left)
#   5. Adjacency comparison (effect of dCE split by adjacency)
#
# ----------------------------------------------------------------------------

library(ggplot2)
library(ggrepel)

# Shared theme
theme_slides <- theme_minimal(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title       = element_text(face = "bold", size = 15),
    plot.subtitle    = element_text(size = 12, color = "grey30"),
    legend.position  = "bottom",
    axis.title       = element_text(size = 13)
  )

# Consistent arci vs. non-arci palette
arci_palette <- c("FALSE" = "darkgoldenrod", "TRUE" = "firebrick4")

# Output directory for plots
plot_dir <- file.path(folder_path, "plots")
dir.create(plot_dir, showWarnings = FALSE)


# ----------------------------------------------------------------------------
# Plot 1: Prefix informational profile
# ----------------------------------------------------------------------------
# Each prefix positioned by its CE (x) and PS (y) values, showing the
# negative correlation (surprisal-memory trade-off) and situating
# arci- and extra- as the two "marked, low-productivity" prefixes.

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
    x = "Contextual entropy (CE, bits)",
    y = "Paradigmatic surprisal (PS, bits)"
  ) +
  theme_slides

ggsave(file.path(plot_dir, "01_prefix_profile.png"),
       p1, width = 7, height = 5.2, dpi = 300)


# ----------------------------------------------------------------------------
# Plot 2: Model-fit plot (observed vs. predicted pair proportions)
# ----------------------------------------------------------------------------
# The headline diagnostic. Points on the diagonal = perfect fit.
# arci- pairs (red) cluster above the line; all other pairs sit tightly
# on it. Point size reflects number of tokens.

p2 <- ggplot(pair_residuals,
             aes(x = predicted, y = observed,
                 size = n_total, color = has_arci)) +
  
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "grey50") +
  
  geom_point(alpha = 1) +
  
  geom_text_repel(aes(label = pair),
                  size = 9, show.legend = FALSE,
                  box.padding = 0.8, max.overlaps = 20,
                  family = "Libertine") +
  
  scale_size_continuous(range = c(2, 10), name = "Tokens") +
  
  scale_color_manual(values = arci_palette,
                     labels = c("FALSE" = "non-arci", "TRUE" = "arci-"),
                     name = NULL) +
  
  guides(
    color = guide_legend(override.aes = list(size = 6))
  ) +
  
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  
  labs(
    x = "Predicted proportion",
    y = "Observed proportion"
  ) +
  
  theme_slides +
  
  theme(
    text = element_text(family = "Libertine"),
    
    axis.text.x  = element_text(color = "black", size = 25),
    axis.text.y  = element_text(color = "black", size = 25),
    axis.title.x = element_text(color = "black", size = 23),
    axis.title.y = element_text(color = "black", size = 23),
    
    legend.text  = element_text(size = 22),
    legend.title = element_text(size = 22)
  )

ggsave("hdplot1.pdf", p2, height = 10, width = 11, device = cairo_pdf)

# ----------------------------------------------------------------------------
# Plot 3: Coefficient plot for the joint model
# ----------------------------------------------------------------------------
# A clean visualization of the key inferential result: both dCE and dPS
# are significant, with dCE being the stronger predictor.

coefs <- broom.mixed::tidy(m_both, effects = "fixed", conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    term = recode(term,
                  "dICTRANS_z" = "Delta*CE",
                  "dICLOCAL_z" = "Delta*PS")
  )

p3 <- ggplot(coefs, aes(x = estimate, y = term)) 
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high),
                 height = 0.15, linewidth = 0.8, color = "steelblue") +
  geom_point(size = 4, color = "steelblue") +
  scale_y_discrete(labels = scales::parse_format()) +
  labs(
    title    = "Fixed-effect coefficients (joint model)",
    subtitle = "Predicting P(A on left); 95% Wald confidence intervals",
    x = "Estimate (log-odds, standardized predictors)",
    y = NULL
  ) +
  theme_slides

ggsave(file.path(plot_dir, "03_coefficients.png"),
       p3, width = 7, height = 4, dpi = 300)


# ----------------------------------------------------------------------------
# Plot 4: Pair-level proportions with sample sizes
# ----------------------------------------------------------------------------
# Bar chart showing prop_A_left per pair, ordered by magnitude, with
# n_total as a text label. Gives an at-a-glance view of which pairs are
# strongly ordered (near 0 or 1) and which are balanced (near 0.5).

pair_counts_plot <- pair_counts %>%
  mutate(
    pair_label = paste0(prefix_A, "-", prefix_B),
    has_arci   = grepl("arci", pair),
    pair_label = reorder(pair_label, prop_A_left)
  )

p4 <- ggplot(pair_counts_plot,
             aes(x = pair_label, y = prop_A_left,
                 fill = has_arci)) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50") +
  geom_col(alpha = 0.85, width = 0.7) +
  geom_text(aes(label = paste0("n=", n_total)),
            hjust = -0.15, size = 3.3) +
  scale_fill_manual(values = arci_palette,
                    labels = c("FALSE" = "non-arci", "TRUE" = "arci-"),
                    name = NULL) +
  coord_flip(ylim = c(0, 1.08)) +
  scale_y_continuous(breaks = seq(0, 1, 0.25)) +
  labs(
    title    = "Ordering proportions by prefix pair",
    subtitle = "Proportion of tokens where the alphabetically-first prefix appears leftward",
    x = NULL,
    y = "P(A on left)"
  ) +
  theme_slides

ggsave(file.path(plot_dir, "04_pair_proportions.png"),
       p4, width = 8, height = 6, dpi = 300)


# ----------------------------------------------------------------------------
# Plot 5: Adjacency comparison
# ----------------------------------------------------------------------------
# Shows the relationship between dCE and P(A on left) separately for
# adjacent and non-adjacent pairs. The two regression lines being
# roughly parallel visualizes the null interaction.

# Aggregate proportions by adjacency and dCE bin
adjacency_data <- stacked_model %>%
  group_by(adjacency) %>%
  mutate(
    dICTRANS_bin = cut(dICTRANS_z,
                       breaks = quantile(dICTRANS_z, probs = seq(0, 1, 0.25),
                                         na.rm = TRUE),
                       include.lowest = TRUE)
  ) %>%
  ungroup()

p5 <- ggplot(stacked_model,
             aes(x = dICTRANS_z, y = observed, color = adjacency)) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"),
              se = TRUE, alpha = 0.15, linewidth = 1) +
  geom_jitter(width = 0, height = 0.04, alpha = 0.25, size = 1.5) +
  scale_color_manual(values = c("adjacent"     = "steelblue",
                                "non_adjacent" = "darkorange"),
                     labels = c("adjacent"     = "Adjacent",
                                "non_adjacent" = "Non-adjacent (3-1)"),
                     name = NULL) +
  labs(
    title    = "Informational gradient: adjacent vs. non-adjacent pairs",
    subtitle = expression(paste(
      "No significant interaction: ",
      chi^2, "(3) = 3.62, ", italic(p), " = 0.31")),
    x = expression(paste(Delta, "CE (z-scored)")),
    y = "P(A on left)"
  ) +
  theme_slides

ggsave(file.path(plot_dir, "05_adjacency.png"),
       p5, width = 7, height = 5, dpi = 300)


# Quick sanity check: show the plots in R
print(p1); print(p2); print(p3); print(p4); print(p5)

cat(sprintf("\nPlots saved to: %s\n", plot_dir))