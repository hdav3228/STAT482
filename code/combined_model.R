#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(lme4)
})

args <- commandArgs(trailingOnly = FALSE)
file_flag <- "--file="
script_path <- sub(file_flag, "", args[grep(file_flag, args)])
script_dir <- dirname(normalizePath(script_path))
project_root <- normalizePath(file.path(script_dir, ".."))

data_dir <- file.path(project_root, "data")
fig_dir <- file.path(project_root, "figures")
result_dir <- file.path(project_root, "results")

dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(result_dir, showWarnings = FALSE, recursive = TRUE)

input_path <- file.path(data_dir, "report2_model_input.csv")
if (!file.exists(input_path)) {
  stop("Missing data/report2_model_input.csv. Run python3 code/eda_analysis_regression.py first.")
}

cat("[INFO] Loading cleaned modeling data from", input_path, "\n")
df <- read.csv(input_path, stringsAsFactors = FALSE)

seq_df <- subset(df, prev_pitch_label != "None")
seq_df <- seq_df[complete.cases(
  seq_df[, c(
    "whiff",
    "release_speed",
    "ivb_inches",
    "hb_inches",
    "release_extension",
    "pitch_label",
    "prev_pitch_label",
    "prev_zone_cat",
    "prev_outcome",
    "pitch_seq",
    "batter"
  )]
), ]

seq_counts <- table(seq_df$pitch_seq)
common_seqs <- names(seq_counts[seq_counts >= 100])
seq_df$pitch_seq_filtered <- ifelse(
  seq_df$pitch_seq %in% common_seqs,
  seq_df$pitch_seq,
  "Other__Other"
)

seq_df$whiff <- as.numeric(seq_df$whiff)
seq_df$batter <- factor(seq_df$batter)
seq_df$prev_pitch_label <- factor(seq_df$prev_pitch_label)
seq_df$prev_zone_cat <- factor(seq_df$prev_zone_cat)
seq_df$prev_outcome <- factor(seq_df$prev_outcome)
seq_df$pitch_seq_filtered <- factor(seq_df$pitch_seq_filtered)

if ("4-Seam FB" %in% levels(seq_df$prev_pitch_label)) {
  seq_df$prev_pitch_label <- relevel(seq_df$prev_pitch_label, ref = "4-Seam FB")
}
if ("Strike" %in% levels(seq_df$prev_zone_cat)) {
  seq_df$prev_zone_cat <- relevel(seq_df$prev_zone_cat, ref = "Strike")
}
if ("Ball" %in% levels(seq_df$prev_outcome)) {
  seq_df$prev_outcome <- relevel(seq_df$prev_outcome, ref = "Ball")
}
if ("4-Seam FB__4-Seam FB" %in% levels(seq_df$pitch_seq_filtered)) {
  seq_df$pitch_seq_filtered <- relevel(seq_df$pitch_seq_filtered, ref = "4-Seam FB__4-Seam FB")
}

cat("[MODEL] Combined-model dataset:", format(nrow(seq_df), big.mark = ","), "pitches\n")
cat("[MODEL] Batters with random intercepts:", nlevels(seq_df$batter), "\n")
cat("[MODEL] Retained pitch-type sequences:", length(common_seqs), "\n")

formula_text <- paste(
  "whiff ~ scale(release_speed) + scale(ivb_inches) +",
  "scale(hb_inches) + scale(release_extension) +",
  "prev_pitch_label + prev_zone_cat + prev_outcome +",
  "pitch_seq_filtered + (1 | batter)"
)

cat("[MODEL] Fitting combined model with glmer...\n")
combined_fit <- glmer(
  as.formula(formula_text),
  data = seq_df,
  family = binomial(link = "logit"),
  nAGQ = 0,
  control = glmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 2e5)
  )
)

model_loglik <- as.numeric(logLik(combined_fit))
model_bic <- as.numeric(BIC(combined_fit))
model_aic <- as.numeric(AIC(combined_fit))
model_df <- attr(logLik(combined_fit), "df")
varcorr_df <- as.data.frame(VarCorr(combined_fit))
random_var <- varcorr_df$vcov[1]
random_sd <- sqrt(random_var)

cat("[MODEL] Log-likelihood:", format(round(model_loglik, 2), nsmall = 2, big.mark = ","), "\n")
cat("[MODEL] BIC:", format(round(model_bic, 2), nsmall = 2, big.mark = ","), "\n")
cat("[MODEL] Random-intercept SD:", round(random_sd, 4), "\n")

combined_summary <- data.frame(
  model = "Combined",
  logLik = model_loglik,
  AIC = model_aic,
  BIC = model_bic,
  n_params = model_df,
  random_intercept_sd = random_sd,
  random_intercept_var = random_var
)
write.csv(
  combined_summary,
  file.path(result_dir, "combined_model_summary.csv"),
  row.names = FALSE
)

fit_summary_path <- file.path(result_dir, "model_fit_summary.csv")
if (file.exists(fit_summary_path)) {
  fit_summary <- read.csv(fit_summary_path, stringsAsFactors = FALSE)
  fit_summary <- subset(fit_summary, model != "Combined")
  fit_summary <- rbind(
    fit_summary,
    data.frame(
      model = "Combined",
      logLik = model_loglik,
      BIC = model_bic,
      n_params = model_df
    )
  )
  write.csv(fit_summary, fit_summary_path, row.names = FALSE)
} else {
  fit_summary <- data.frame(
    model = "Combined",
    logLik = model_loglik,
    BIC = model_bic,
    n_params = model_df
  )
  write.csv(fit_summary, fit_summary_path, row.names = FALSE)
}

random_effects <- ranef(combined_fit)$batter
random_df <- data.frame(
  batter = rownames(random_effects),
  intercept = random_effects[, 1]
)
random_df <- random_df[order(random_df$intercept), ]
write.csv(
  random_df,
  file.path(result_dir, "combined_batter_effects.csv"),
  row.names = FALSE
)

fixed_effects <- data.frame(
  term = names(fixef(combined_fit)),
  estimate = unname(fixef(combined_fit))
)
write.csv(
  fixed_effects,
  file.path(result_dir, "combined_fixed_effects.csv"),
  row.names = FALSE
)

fit_summary <- fit_summary[match(c("Baseline", "Sequencing", "Combined"), fit_summary$model), ]
fit_summary <- fit_summary[!is.na(fit_summary$model), ]
bic_gap <- fit_summary$BIC - min(fit_summary$BIC)
ll_gain <- fit_summary$logLik - fit_summary$logLik[fit_summary$model == "Baseline"]

comparison_fig <- file.path(fig_dir, "fig_model_comparison.png")
png(comparison_fig, width = 1600, height = 700, res = 170)
par(mfrow = c(1, 2), mar = c(6, 5, 4, 2) + 0.1)

barplot(
  bic_gap,
  names.arg = fit_summary$model,
  col = c("#4C72B0", "#DD8452", "#55A868")[seq_len(nrow(fit_summary))],
  main = "BIC Gap from Best Model",
  ylab = "BIC Difference (lower is better)",
  las = 2
)

barplot(
  ll_gain,
  names.arg = fit_summary$model,
  col = c("#4C72B0", "#DD8452", "#55A868")[seq_len(nrow(fit_summary))],
  main = "Log-Likelihood Gain over Baseline",
  ylab = "Log-Likelihood Improvement",
  las = 2
)
dev.off()
cat("[FIG] Saved", comparison_fig, "\n")

random_fig <- file.path(fig_dir, "fig_combined_batter_effects.png")
png(random_fig, width = 1100, height = 700, res = 170)
hist(
  random_df$intercept,
  breaks = 35,
  col = "#55A868",
  border = "white",
  main = "Combined Model Batter Random Intercepts",
  xlab = "Estimated Batter Random Intercept"
)
abline(v = 0, lty = 2, lwd = 2, col = "#C44E52")
mtext(
  paste0("Estimated SD = ", sprintf("%.3f", random_sd)),
  side = 3,
  line = 0.25,
  cex = 1
)
dev.off()
cat("[FIG] Saved", random_fig, "\n")
