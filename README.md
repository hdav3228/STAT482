# STAT482
Capstone for STATISTICS 482

## Project Overview

This project analyzes 702,384 pitches from the full 2025 MLB season to understand the factors that lead to a swing-and-miss (whiff). The overall whiff rate in the dataset is 12.0%. Notably, we calculate whiff rate out of *all* pitches (including balls, called strikes, fouls, and balls in-play) rather than just swings. This approach allows our models to predict whether any given pitch will result in a whiff, capturing both the batter's decision to swing and the ultimate outcome of that swing.

## Key Findings

### 1. Pitch Type Matters
Different pitch types yield drastically different whiff rates. Velocity alone does not guarantee a whiff:
* **High Whiff Rates:** Splitters are the most effective pitch at generating whiffs (17.8%), followed closely by knuckle-curves (16.4%) and sliders (16.2%).
* **Low Whiff Rates:** Fastball variants (4-seam, sinker, cutter) populate the bottom half of the leaderboards. Sinkers are the easiest to make contact with (6.3% whiff rate). Sinkers are thrown with high velocity (mid-90s) and late downward, arm-side movement, making them explicitly designed to induce ground balls rather than strikeouts.

### 2. The Importance of Sequencing
Pitches cannot be analyzed in a vacuum; the preceding pitch significantly impacts a batter's success. This is why sequencing is a core component of our model.
* The highest whiff rates emerge from breaking-ball-to-breaking-ball sequences (e.g., a curveball followed by a splitter results in a 20.1% whiff rate), challenging the traditional wisdom that a fastball followed by a breaking ball is always optimal.
* Fastball to splitter transitions remain highly effective (16.0–19.2% whiff rate) because the pitches look identical out of the pitcher's hand (tunneling) before diverging late.
* Sinkers remain a "contact pitch," generating low whiff rates (5.7–8.5%) regardless of the preceding pitch.
* *Methodology Note:* For the first pitch of an at-bat, previous-pitch lag variables are coded as "None" rather than being dropped. This preserves the dataset and allows the model to estimate a baseline for first pitches.

### 3. Batter Heterogeneity
There is massive variance in predicting whiffs based on who is at the plate. Batter whiff rates form a roughly normal distribution centered at 12%, but range from as low as 2.4% to as high as 23.2% (a nearly 10x difference).
* *Methodology Note:* This extreme variance necessitates the use of Generalized Linear Mixed Models (GLMM) with random effects. Attempting to add "Batter" as a fixed effect across 531+ batters would consume over 530 degrees of freedom. By using random effects, the GLMM estimates batter heterogeneity with just one parameter ($\sigma^2$). This is vastly more computationally efficient and ensures that the "noise" from a highly disciplined hitter versus a free-swinger doesn't drown out the true underlying effect of pitch quality.

### 4. Limitations of Physical Metrics
Standard predictive models that rely solely on physical characteristics fail because they treat each pitch in isolation. Evaluating individual physical metrics against whiff rates yields near-zero correlations:
* Speed: $r = -0.05$
* Induced Vertical Break (IVB): $r = -0.04$
* Horizontal Break (HB): $r = 0.02$
* Extension: $r = 0.01$

While individual metrics don't predict whiffs linearly, they do interact with one another. For instance, the correlation between Speed and IVB is relatively strong ($r = 0.72$). To account for this, the model employs Variance Inflation Factor (VIF) diagnostics to check for multicollinearity, alongside regular monitoring of coefficient stability in our logistic regression.
