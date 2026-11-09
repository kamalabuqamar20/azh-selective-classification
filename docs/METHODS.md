# Methods and reasoning

## The scientific question

Let `C(x)` be the classifier's maximum temperature-scaled probability and `Q(x)` a focus proxy. The working question is whether `Q` helps rank prediction correctness beyond `C`, especially after accounting for differences among wound classes.

There is a useful theoretical boundary. Under 0–1 classification loss, if the **true posterior given all available information** were known, predictions with the highest maximum posterior would minimise expected classification error at a fixed coverage, apart from ties. A separate feature derived from the same information could not improve that ideal rule. Our model score is not that ideal posterior: marginal calibration alone does not show that correctness is independent of focus at a fixed score. The experiment investigates that gap; it does not prove it exists.

For example, two equally sized image groups might both be assigned confidence 0.8 while their actual accuracies are 0.95 and 0.65. Their combined accuracy is 0.8, yet a feature that identifies the groups could change the ranking. This is an illustration, not an observation from AZH. Calibration is discussed by [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html).

## Data and exact-match audit

The run uses the upstream repository at commit `e7dd90de113dcbd39665f2bd0e5613adeffff5fa` and verifies the archives' SHA-256 checksums. The archive loader visits test images first, then training images. It hashes both file bytes and decoded RGB pixels after EXIF orientation. Matching images with conflicting class labels raise an error. The single byte-identical train/test duplicate was kept in test and removed from training.

The 930 original images include four wound types plus normal skin and background. After the single removal, 929 images remain across all six categories. The model task uses **729 wound images**: 545 in the deduplicated original training pool and 184 in the original test pool. No patient-level or acquisition-source identifier was used.

## Fixed features and focus proxy

Images are converted to RGB and resized to 128×128. The classifier receives 512 local gradient-orientation histogram values and 192 local RGB histogram values. Features are standardised using model-fitting data only. The classifier is L2 multinomial logistic regression (`C=0.1`).

The focus proxy is the variance of the four-neighbour grayscale Laplacian on interior pixels of the resized image. It is a simple edge-variation measure, not a verified clinical quality score. Texture, colour conversion, resizing, compression and noise can influence it. Because the base classifier also sees gradient-derived features, the proxy may partly duplicate information already in its inputs. [Zhang et al.](https://doi.org/10.2196/26149) used a focus measure for wound-image feedback and highlighted image-content dependence.

## Original test-split pilot

The 545 training-pool wound images were divided by class into:

| Purpose | Images |
|---|---:|
| Fit the base classifier | 381 |
| Fit one temperature by validation log loss | 82 |
| Fit the confidence-plus-focus correctness selector | 82 |
| Original test split | 184 |

The fixed classifier's predicted class remains unchanged by a positive temperature. For multiclass problems, the **ranking** of maximum probabilities across images can change with temperature; the synthetic tests include such a counterexample.

Three scores rank the same test predictions:

1. Maximum temperature-scaled class probability.
2. Laplacian variance.
3. A logistic correctness score using the logit of maximum probability and `log(1 + Laplacian variance)`, fitted on the separate selector subset.

No true test class is supplied to a selection score. For a requested coverage `c`, the top `ceil(c × n)` scores are retained. At `c=0.8`, this retains 148 of 184 images. This is a retrospective **matched-coverage ranking** on the test pool; it does not estimate what coverage a threshold fixed in advance on validation data would achieve in deployment.

## Class composition and its limits

The retained error for rule `g` can be expressed as a weighted sum of within-class retained errors, where the weights are the retained class proportions. The class-standardised estimate replaces those weights with the full test-set class proportions:

`R_standardised(g) = Σ_c P_test(Y=c) × P(error | retained by g, Y=c)`.

If a class has no retained observations, its standardised risk is undefined; the validated helper returns `None`. Even when all classes are represented, reweighting does **not** force two rules to retain the same fraction within each class. It cannot, alone, prove that a difference is free of class-related selection effects.

The code also reports a sensitivity diagnostic that retains the same *fraction* within each true class, subject to integer rounding. It uses ground-truth class to choose the images and is therefore an **oracle audit only**, not a usable prediction-time selection policy. Small class counts make this diagnostic noisy.

The paired interval resamples images within each true class 2,000 times and recomputes the rankings at matched coverage. It conditions on the trained model, fitted selectors and test class counts. It does not include uncertainty from retraining, patient clustering or other datasets.

## Follow-up within the training pool

After the original test results were visible, a five-fold stratified image-level analysis was designed on the 545-image original training pool. Each outer fold is held out once. Within the other four folds, separate subsets fit the model, temperature and selectors.

To better isolate the added feature, this follow-up fits **both** a confidence-only logistic correctness selector and a confidence-plus-focus selector on the same inner data. It also reports direct confidence and focus-only rankings. Each outer fold retains its own top 80%; the five retained error counts are then pooled. Scores are not pooled across separately fitted folds.

The original test archive is still read so that the earlier cross-split duplicate is removed consistently, but this follow-up does not score its test images. Folds share training data, no patient identifiers were available, and the protocol was designed after observing the first result. This is sensitivity analysis, not independent or preregistered confirmation.
