# Model Training Summary

## Selected model

**random_forest (tuned)**

## Dataset

- Training rows: 712
- Holdout rows: 179

## Cross-validation

- Selected-model Macro F1: unavailable

## Holdout evaluation

| Metric | Score |
|---|---:|
| Accuracy | 88.83% |
| Balanced accuracy | 86.00% |
| Macro F1 | 86.41% |

## Most influential original features

1. Fare
2. FareLog
3. CabinCount
4. CabinDeck
5. HasCabin

## Candidate-model comparison

| model | cv_accuracy_mean | cv_balanced_accuracy_mean | cv_f1_macro_mean | cv_f1_macro_std |
| --- | --- | --- | --- | --- |
| random_forest | 0.9185 | 0.9278 | 0.9088 | 0.0258 |
| logistic_regression | 0.8342 | 0.8358 | 0.8221 | 0.0223 |
| dummy | 0.5506 | 0.3333 | 0.2367 | 0.0004 |
