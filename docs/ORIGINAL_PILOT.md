# Original pilot note — 8 November 2026

This is the preserved note from the earlier standalone bundle. Its output paths describe that bundle; use the repository's [current reproduction instructions](../README.md#reproduce) and [results](RESULTS.md) for present paths and the later sensitivity analysis.

# AZH selective classification pilot

8 November 2026. This is an AI-assisted exploratory analysis prepared to inform a master's research proposal.

The pilot asked whether a focus proxy adds information to confidence when a fixed classifier can defer predictions. It used four AZH wound-type labels (diabetic, pressure, surgical and venous), which differ from the infection/ischaemia labels in DFUC2021.

## Result at matched coverage

All three rules retained **148 of 184 held-out images (80.43%)**. They ranked the same classifier's predictions and could retain different images.

| Selection rule | Retained errors | Error among retained | Class-standardised error |
|---|---:|---:|---:|
| Temperature-scaled confidence | 55/148 | 37.16% | 37.35% |
| Laplacian variance | 61/148 | 41.22% | 41.62% |
| Confidence + Laplacian variance | 57/148 | 38.51% | 38.76% |

The joint rule's error was **+1.35 percentage points** higher than confidence alone. The paired image-level bootstrap interval was **[-0.69, +4.07] percentage points**. The pilot therefore did not establish an improvement from adding Laplacian variance. Reweighting to the same true-class mix led to the same conclusion. The curves in `risk_coverage.png` show the trade-off over 50% to 100% coverage; the CSV contains the full curve.

## How the comparison was run

- Downloaded the two original archives from the authors' [AZH release](https://github.com/uwm-bigdata/wound-classification-using-images-and-locations/tree/e7dd90de113dcbd39665f2bd0e5613adeffff5fa). The archives contained 930 image files including normal skin and background. An exact-byte and decoded-pixel audit found one byte-identical surgical image across the official train and test splits. The training copy was excluded, leaving 929 unique images. The four wound classes contributed 729 images to the pilot.
- The official test split supplied 184 wound images. The remaining wound images were split by class into 381 for model fitting, 82 for calibration, and 82 for fitting the joint selector. The fixed model was L2 multinomial logistic regression using local gradient and RGB histograms. Its unfiltered test accuracy was 56.52%.
- A single temperature was fitted by minimising validation log loss. Test negative log likelihood moved from 1.306 to 1.078. The three rankings were maximum temperature-scaled class probability, Laplacian variance on grayscale images resized to 128 by 128, and a logistic correctness score using both features. The model and validation fits were complete before test scoring.
- The primary comparison ranks test scores at a fixed nominal 80% coverage without using test labels for ranking. It estimates error at equal coverage, not the coverage of a fixed deployment threshold. The reweighted column applies the full test-set class proportions to each class's retained error rate. The separate matched-within-class diagnostic in the JSON uses true labels only for evaluation.
- Paired intervals use 2,000 class-stratified image bootstrap draws and re-rank each resample. They condition on the fitted model and test class mix.

## What this does and does not show

Laplacian variance responds to image texture as well as focus. This run involved one lightweight classifier, one wound dataset and one split. Patient and camera/source identifiers were not used, so the image-level intervals cannot establish patient-level independence and the experiment cannot test transfer to a new acquisition source. Exact deduplication does not exclude near-duplicate images or repeated patients. The estimates are exploratory and do not establish clinical validity or show that focus is useless in other settings.

## Reproduce

Use Python 3.12.14 (or a compatible Python 3.12 installation). From the extracted folder:

```sh
python -m pip install -r requirements.txt
python run_azh_pilot.py --download --data-dir azh_input --output reproduced
```

If you already have the pinned `Train.zip` and `Test.zip`, place them in `azh_input` and omit `--download`. The script checks each archive's SHA-256 hash before use. The bundle contains aggregate results, the exact run code and a plot; it contains no clinical images or individual records.

## References

- Anisuzzaman et al. (2022), [Multi-modal wound classification using wound image and location by deep neural network](https://doi.org/10.1038/s41598-022-21813-0). Public AZH source: [University of Wisconsin-Milwaukee Big Data Lab release](https://github.com/uwm-bigdata/wound-classification-using-images-and-locations).
- Guo et al. (2017), [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).
- Geifman and El-Yaniv (2017), [Selective Classification for Deep Neural Networks](https://papers.nips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html).
- Zhang et al. (2021), [Wound Image Quality From a Mobile Health Tool for Home-Based Chronic Wound Management With Real-Time Quality Feedback](https://doi.org/10.2196/26149).
