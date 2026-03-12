# STAT482
Capstone for STATISTICS 482

## Project overview

This project analyzes 702,384 pitches from the full 2025 MLB season to understand what causes a swing-and-miss (whiff). The dataset's overall whiff rate is 12.0%, calculated across *all* pitches — balls, called strikes, fouls, and balls in play — not just swings. That choice is intentional. Restricting the denominator to swings-only would assume the batter already decided to swing, which throws out half the variation worth studying. Calculating off all pitches means the model has to predict both whether the batter swings and whether they miss.

## Key findings

### 1. Pitch type matters

Pitch type has a bigger effect on whiff rate than any single physical metric. Velocity alone doesn't explain it:

* Splitters lead the dataset at 17.8%, followed by knuckle-curves (16.4%) and sliders (16.2%).
* Fastball variants sit at the bottom — 4-seamers, sinkers, cutters. Sinkers are the easiest pitch to make contact with (6.3%), which seems counterintuitive for a pitch thrown in the mid-90s. But sinkers aren't strikeout pitches. The late downward, arm-side movement is designed to produce ground balls. Low whiff rate is the feature, not the bug.

### 2. The importance of sequencing

The pitch before the current one matters. A lot. This is why sequencing is built into the model rather than treated as noise.

* Breaking-ball-to-breaking-ball sequences generate some of the highest whiff rates in the dataset. A curveball followed by a splitter produces a 20.1% whiff rate — which runs against the conventional wisdom that breaking balls need a fastball setup to be effective.
* Fastball-to-splitter transitions are also high (16.0–19.2%). The reason is tunneling: both pitches look the same out of the hand before they diverge late, which disrupts the batter's timing even when they're sitting on the fastball.
* Sinkers stay contact pitches regardless of sequence, ranging from 5.7–8.5% whiff rate even in the most favorable setups.
* *Methodology note:* For the first pitch of an at-bat, previous-pitch lag variables are coded as "None" rather than dropped. Dropping first pitches would introduce selection effects and shrink the dataset; coding them as "None" lets the model estimate a first-pitch baseline directly.

### 3. Batter heterogeneity

Batter whiff rates range from 2.4% to 23.2% across the 531 batters in the dataset. The distribution is roughly normal and centered near 12%, but that spread is too wide to treat as random noise. A free-swinger and a contact hitter are not interchangeable observations.

* *Methodology note:* This is why the model uses Generalized Linear Mixed Models (GLMM) with random effects rather than treating batter as a fixed effect. Adding 531 batters as fixed predictors would consume 530+ degrees of freedom and make the model unusable. Random effects capture that variation with a single parameter ($\sigma^2$), which is more efficient and prevents individual batter tendencies from masking the underlying effect of pitch quality.

### 4. Limitations of physical metrics

Models built only on physical pitch characteristics tend to fail because they treat each pitch as an isolated event. Looking at individual metrics against whiff rates, the linear correlations are effectively zero:

* Speed: $r = -0.05$
* Induced Vertical Break (IVB): $r = -0.04$
* Horizontal Break (HB): $r = 0.02$
* Extension: $r = 0.01$

A 98 mph fastball grooved down the middle isn't a whiff. The physical metrics matter, but through interactions, not individually. Speed and IVB are moderately correlated with each other ($r = 0.72$), which creates multicollinearity risk. The model uses Variance Inflation Factor (VIF) diagnostics throughout and monitors coefficient stability in the logistic regression to catch problems when they show up.
