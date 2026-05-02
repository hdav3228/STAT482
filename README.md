# STAT482: Modeling MLB Swing-and-Miss Outcomes

This repository contains the STAT 482 final project report and the supporting code, figures, and LaTeX source. We study swing-and-miss probability using 2025 MLB regular-season Statcast pitch data.

The headline is that stuff, sequencing, and batter skill all matter, and the combined model wins clearly by BIC.

## Research Question

How do pitch sequencing and batter-specific tendencies jointly influence the probability of a swing-and-miss, beyond the physical characteristics of the pitch itself?

The report asks whether whiff probability depends on more than pitch quality alone. Our answer is yes: the previous pitch and the batter's own contact baseline both add real signal.

## Files

- [FinalReport.pdf](./FinalReport.pdf): final project report
- [build/](./build): LaTeX source, bibliography, and report figures
- [code/](./code): exploratory analysis, logistic models, GLMM workflow, and combined mixed-effects model

## Data

The analysis starts with 711,897 raw pitches and keeps 702,321 after cleaning.

- Overall whiff rate: 12.0%
- Batters with at least 200 pitches seen: 531
- Pitchers: 811
- Batter whiff-rate range: 2.4% to 23.2%

Whiff means swinging strike, blocked swinging strike, or foul tip. Predictors are grouped into physical variables, sequence context, and batter effects.

## EDA Story

The exploratory analysis already points toward the final model story: pitch type matters, single physical effects are modest alone, and hitter context is real.

- Splitters, sliders, and knuckle-curves create the clearest pitch-type whiff spread.
- Whiff pitches skew toward lower IVB and heavier horizontal-break tails, but no single physical variable separates outcomes by itself.
- Specific pitch-pair combinations, especially into splitters and breaking balls, produce materially higher whiff rates.
- Batter whiff tendencies range from 2.4% to 23.2%, enough to justify random intercepts.

## Model Progression

The baseline logistic regression uses only physical pitch traits. It finds real effects, but the fit is limited and leaves a lot unexplained.

Adding sequence context produces the biggest single jump in fit. Whiff probability depends on what the batter just saw, with fastball-to-secondary sequences showing especially strong effects.

The standalone GLMM quantifies batter-to-batter differences. The ICC is real but modest, so hitter identity matters without dominating the whole process.

The combined model layers physical variables, sequencing variables, and batter random intercepts. It is the best directly comparable model in the report by a large margin, with BIC 398,487.82 and log-likelihood -198,638.43.

## Final Takeaway

The best explanation of a swing-and-miss combines pitch quality, sequence context, and batter-specific contact skill.

What you throw, when you throw it, and who is hitting all matter, and the combined model captures those three layers best.
