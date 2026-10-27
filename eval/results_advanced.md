**In-sample (frozen thresholds):** exact 7/10, within one 9/10, quadratic weighted kappa **0.786**, mean abs error 0.4 levels

| actual \ predicted | Poor | Average | Good | Excellent |
|---|---|---|---|---|
| Poor | 3 | 1 | 0 | 0 |
| Average | 1 | 1 | 0 | 0 |
| Good | 1 | 0 | 1 | 0 |
| Excellent | 0 | 0 | 0 | 2 |

**Leave-one-out thresholds:** exact 6/10, within one 9/10, quadratic weighted kappa **0.762**, mean abs error 0.5 levels

| actual \ predicted | Poor | Average | Good | Excellent |
|---|---|---|---|---|
| Poor | 3 | 1 | 0 | 0 |
| Average | 2 | 0 | 0 | 0 |
| Good | 1 | 0 | 1 | 0 |
| Excellent | 0 | 0 | 0 | 2 |

Model's holistic label alone: exact 2/10, within one 7/10, kappa 0.066.

Quotes: 135 exact, 6 approximate, 3 unverified, 31 non-verbal; verified rate 0.979.

| dimension | Spearman vs label | Spearman vs holistic | mean (Poor/Avg/Good/Exc) |
|---|---|---|---|
| rapport | 0.8 | 0.35 | 1.5/2/2/3 |
| engagement | 0.33 | 0.17 | 3.25/2.5/3/4 |
| clarity | 0.25 | 0.6 | 3.25/2/3/4 |
| energy | 0.29 | 0.65 | 2.5/2/2.5/3 |
| closure | 0.51 | 0.56 | 2/2/2.5/3 |
| setup | 0.34 | 0.75 | 3.5/4/3.5/4.5 |

Cross-model dimension consistency (v4-l35 vs v4lite, same prompt, n=12):

| dimension | mean abs diff | exact | within one |
|---|---|---|---|
| rapport | 0.42 | 7/12 | 12/12 |
| engagement | 0.5 | 8/12 | 11/12 |
| clarity | 0.67 | 5/12 | 11/12 |
| energy | 0.33 | 8/12 | 12/12 |
| closure | 0.25 | 9/12 | 12/12 |
| setup | 1.08 | 2/12 | 9/12 |

Stability, 3 scoring repeats per class with fixed production seed: pairwise flip rate 0.0, classes with any flip 0/3

| id | label | ratings | score mean | score sd |
|---|---|---|---|---|
| EX-1 | Excellent | Average, Average, Average | 3.1 | 0.0 |
| AV-1 | Average | Average, Average, Average | 2.85 | 0.0 |
| PO-2 | Poor | Average, Average, Average | 3.15 | 0.0 |

Mean per-dimension SD across repeats: rapport 0.0, engagement 0.0, clarity 0.0, energy 0.0, closure 0.0, setup 0.0

Stability, 3 scoring repeats per class with a different seed per repeat: pairwise flip rate 0.667, classes with any flip 3/3

| id | label | ratings | score mean | score sd |
|---|---|---|---|---|
| EX-1 | Excellent | Average, Excellent, Average | 3.267 | 0.413 |
| AV-1 | Average | Poor, Poor, Average | 2.417 | 0.409 |
| PO-2 | Poor | Average, Good, Average | 3.15 | 0.141 |

Mean per-dimension SD across repeats: rapport 0.471, engagement 0.314, clarity 0.471, energy 0.314, closure 0.573, setup 0.157
