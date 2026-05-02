# STAT482: Modeling MLB Swing-and-Miss Outcomes

This repository contains the code, figures, and LaTeX sources for the STAT 482 final project report. The project studies pitch-level Statcast data from the full 2025 MLB regular season to understand what drives a swing-and-miss, or whiff.

The central question is:

> How do pitch sequencing and batter-specific tendencies influence whiff probability beyond the physical characteristics of the pitch itself?

The final report is available at [FinalReport.pdf](./FinalReport.pdf). The report source, bibliography, figures, and LaTeX build script live in [build/](./build). The modeling code and notebook live in [code/](./code).

## Dataset

The analysis begins with 711,897 raw pitches from the 2025 MLB regular season. After removing non-competitive events, dropping pitches with missing physical measurements, filtering unrealistic release speeds, and restricting to the nine main pitch types, the cleaned dataset contains 702,321 pitches.

Key descriptive facts from the cleaned sample:

- Overall whiff rate: 12.0%
- Batters with at least 200 pitches seen: 531
- Unique pitchers: 811
- Batter whiff-rate range: 2.4% to 23.2%

These summary patterns already suggest that neither raw pitch quality nor batter identity can be ignored.

## Repository Layout

- [FinalReport.pdf](./FinalReport.pdf): compiled final project report
- [build/](./build): LaTeX source, bibliography, compile script, and report figures
- [code/eda_analysis_regression.py](./code/eda_analysis_regression.py): EDA, baseline model, sequencing model, and report-ready figure generation
- [code/eda_regression.ipynb](./code/eda_regression.ipynb): organized notebook version of the EDA, baseline, sequencing, and standalone GLMM workflow
- [code/combined_model.R](./code/combined_model.R): combined mixed model with sequencing terms and batter random intercepts

## Exploratory Analysis

The exploratory analysis motivates all four model choices.

- Pitch type matters. Splitters, sliders, and knuckle-curves generate the highest whiff rates, while sinkers generate the lowest.
- Physical metrics alone are not enough. Marginal correlations between whiff and individual physical variables are small.
- Sequencing matters. Certain previous-to-current pitch combinations produce much higher whiff rates than either pitch type would suggest in isolation.
- Batter heterogeneity is real. Batter-level whiff rates vary widely, which supports random-effects modeling.

These findings imply that a useful whiff model must go beyond isolated pitch physics.

## Model 1: Baseline Logistic Regression

The baseline model uses only physical pitch characteristics: release speed, induced vertical break, horizontal break, and release extension.

Main insight:

- Physical features do move the log-odds of a whiff, but only modestly.
- On the common non-first-pitch sample used for direct comparison, the baseline model has log-likelihood `-203,264.99` and BIC `406,595.79`.
- Standardized coefficients show negative effects for release speed and induced vertical break, and positive effects for horizontal break and extension, but the magnitudes are not large enough to explain most of the observed variation in whiffs.

Interpretation:

This model gives a useful benchmark, but it treats each pitch as a context-free event. It cannot account for what the batter saw on the previous pitch, and it cannot separate pitch effects from hitter-specific contact skill.

## Model 2: Sequencing Model

The sequencing model extends the baseline by adding:

- previous pitch type
- previous pitch zone category
- previous pitch outcome
- current-by-previous pitch-type interaction terms

Main insight:

- Sequencing materially improves model fit.
- Relative to the baseline on the same observations, the sequencing model increases the log-likelihood to `-200,685.21`.
- The likelihood-ratio statistic is `5,159.56` on `91` degrees of freedom with p-value `< 10^-16`.
- BIC improves to `402,634.04`, a change of `-3,961.76` versus baseline.

Baseball interpretation:

- Four-seam fastball to knuckle-curve, splitter, sweeper, and slider transitions all have strong positive effects on whiff probability.
- Four-seam fastball to sinker has a strong negative interaction, which is consistent with sinkers being contact-management pitches rather than strikeout pitches.

Takeaway:

Whiff probability depends on pitch context, not just the current pitch.

## Model 3: GLMM with Batter Random Intercepts

The standalone GLMM focuses on batter heterogeneity by adding a batter-level random intercept to a logistic model with sequencing-style fixed effects.

Because of computational cost, the notebook implementation fits this model on a random subsample of 30,000 pitches covering 649 batters and 96 fixed-effect terms using `statsmodels.BinomialBayesMixedGLM`.

Main insight:

- The estimated batter random-intercept standard deviation is `0.301`.
- The corresponding variance is `0.091`.
- The implied intra-class correlation is `2.69%`.

Interpretation:

Even after controlling for pitch physics and pitch context, batter-level heterogeneity is real but modest. Batter identity is not noise, but the corrected ICC of 2.7% suggests only a small share of latent whiff variation lies between batters.

This model is used mainly to quantify batter-level heterogeneity rather than to serve as the main full-sample comparison model, since it is fit with a different estimator on a subsample.

## Model 4: Combined Model

The combined model is the most complete specification in the project. It includes:

- physical pitch characteristics
- sequencing variables
- pitch-sequence interaction terms
- batter random intercepts

This model is fit in [code/combined_model.R](./code/combined_model.R) using `lme4::glmer` on the full non-first-pitch modeling sample.

Main insight:

- The combined model achieves log-likelihood `-198,638.43`.
- Its BIC is `398,487.82`, the best among the directly comparable full-sample models.
- Relative to the sequencing model, it lowers BIC by `4,146.22` and improves log-likelihood by about `2,047`.
- The estimated batter random-intercept standard deviation is `0.327`.

Interpretation:

The combined model shows that sequencing and batter heterogeneity are complementary, not redundant. Pitchers generate whiffs not only through pitch quality and sequencing, but also by exploiting batter-specific weaknesses. This is the strongest overall model in the project.

## Overall Conclusions

The project produces four consistent conclusions:

- Physical pitch characteristics matter, but they are not sufficient on their own.
- Sequencing has a large and statistically meaningful effect on whiff probability.
- Batter-level contact skill remains important even after pitch context is included.
- The best explanation of whiffs combines pitch quality, sequence effects, and batter heterogeneity in one model.

In baseball terms, pitchers do not create whiffs only by throwing harder or creating more movement. They also create whiffs by choosing effective pitch orders and by targeting hitters whose baseline contact abilities differ in systematic ways.

## Reproducing the Analysis

Typical workflow:

1. Run `python3 code/eda_analysis_regression.py`
2. Run `Rscript code/combined_model.R`
3. Open and run `code/eda_regression.ipynb` for the notebook-based GLMM workflow
4. Compile the report with `./build/compile.sh`

The compiled final report PDF is copied back to the repository root as `FinalReport.pdf`.
