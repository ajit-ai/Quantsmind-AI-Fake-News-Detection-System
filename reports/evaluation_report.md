# Evaluation Report

> Held-out test set evaluated exactly once. No hyperparameter tuning against this split.

## Metrics (held-out test)

| Metric | Value |
|---|---|
| Accuracy | 1.0000 |
| Precision (macro) | 1.0000 |
| Recall (macro) | 1.0000 |
| F1 (macro) | 1.0000 |
| Precision (real) | 1.0000 |
| Recall (real) | 1.0000 |
| F1 (real) | 1.0000 |
| Precision (fake) | 1.0000 |
| Recall (fake) | 1.0000 |
| F1 (fake) | 1.0000 |
| ROC-AUC | 1.0 |

## Confusion Matrix

|  | Pred real | Pred fake |
|---|---|---|
| True real | 151 | 0 |
| True fake | 0 | 150 |

## Notes

- Target is >=0.90 accuracy on held-out test for curated benchmarks (WELFake/ISOT). If not met, retry with hyperparameter adjustments and document config.
- This is a style/pattern triage filter, not a ground-truth fact checker; open-domain adversarial misinformation will score lower.
- Source-credibility feature, if used, must be evaluated separately to avoid domain leakage.

## Failure cases (manual review guidance)

- Review FP/FN examples: sensational but true vs. subtly false claims.
- Check input length / vocab shift drift monitoring.
