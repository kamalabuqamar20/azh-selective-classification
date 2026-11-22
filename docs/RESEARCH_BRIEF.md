# Research brief: confidence, image quality and selective prediction

## Motivation and question

When a wound-image classifier is uncertain, a selective system may defer the
prediction for review. Image quality might provide information beyond the
model's confidence, but a sharpness score can also respond to wound texture,
resolution or capture conditions. The research question is whether adding a
focus proxy changes retained-prediction error at matched coverage, compared
with a confidence-only selector.

The intended future question concerns images from previously unseen patients.
**The present AZH pilot cannot answer that patient-independent question**:
the available release does not provide patient identifiers for grouping.
Using a Laplacian-based focus measure on wound images is not itself a novel
invention; [Zhang et al. (2021)](https://doi.org/10.2196/26149) used a related
measure for image-quality feedback. [Lott et al. (2026)](https://pubmed.ncbi.nlm.nih.gov/42314450/)
studied calibrated selective prediction for pressure-injury staging, while
[Choi et al. (2026)](https://pubmed.ncbi.nlm.nih.gov/42870149/) reported
cross-dataset differences for the same clinical task. These papers do not by
themselves establish the novelty of the proposed comparison. A broader and
updated literature review is needed.

The proposed next question is: **For wound-type predictions on images from
patients excluded from classifier training, probability calibration and
selector development, how does adding a prespecified image-sharpness measure
to an otherwise equivalent selector based on calibrated confidence change
retained-prediction error at approximately 80% image coverage?** The second
question is whether that difference persists across relevant acquisition
conditions. The design, feasibility conditions and distinction between
retrospective coverage and a prospective threshold are detailed in the
[proposed next study](RESEARCH_PROPOSAL.md). This is a future plan, not a
preregistered protocol or a result established by the current pilots.

## Work completed

- Tested four wound-type labels using image features and an optional frozen
  ImageNet-pretrained ResNet-18 encoder with a fitted classifier head.
- Compared confidence-only with confidence-plus-focus selection at matched
  coverage, reported per-class counts and checked fixed-threshold behaviour
  under synthetic blur.
- Evaluated base-classifier calibration before and after temperature scaling
  using held-out image folds, ECE, NLL and multiclass Brier score.
- Checked how the paired comparison changes across five image-level split
  seeds and audited a potential patient-grouped dataset's metadata.
- Ran a separate five-fold patient-grouped pilot on 141 LUTSeg images from 39
  patients, using a binary label derived from tissue-mask metadata. This is a
  different task, not an external evaluation of the four-class AZH model.
- Prepared code, aggregate results, data hashes and offline tests for the
  research prototype. The experimental implementation and analysis were
  developed with AI assistance.

## What the results support

On the original 184-image test split, confidence alone retained 55 errors out
of 148 selected images; the joint rule retained 57/148. On the 545-image
training pool, results across five repeated image-level splits did not show a
consistent benefit from focus. The pretrained pipeline's post-scaling top-label
ECE was 0.0650 on clean images and 0.2272 at the strongest synthetic blur.
These are exploratory findings on a small dataset, not clinical validation or
evidence of a patient-independent effect.

In the separate LUTSeg experiment, the unweighted ResNet-feature model
correctly identified only 5 of 32 positive images before deferral. Learned
confidence and confidence plus focus each retained 33 errors among 115 images.
This patient-grouped pilot demonstrates a different evaluation design, while
revealing substantial class-specific failure and limited sample size.

## Next study for patient-independent AZH wound-type predictions

Secure relevant labels and adequate numbers of independently identified
patients; verify label definitions and data rights. Before the final
evaluation, specify a paired comparison, model and calibration procedures,
coverage target, subgroup analyses, uncertainty method and sample-size plan.
Keep every patient's images in one partition, fit all choices without final
test labels, and report both matched-coverage analysis and the actual coverage
of thresholds fixed on development data. If an independent patient-level test
cohort is available, use patient-aware uncertainty methods and state what
sources of variability the interval does and does not include. External-site
and prospective clinical evaluation are separate future steps.

Read the [methods](METHODS.md), [split sensitivity](SPLIT_SENSITIVITY.md),
[calibration audit](CALIBRATION_AUDIT.md) and
[patient-data feasibility audit](PATIENT_DATA_FEASIBILITY.md), and the
[separate LUTSeg pilot](LUTSEG_PATIENT_PILOT.md), for evidence and limitations.
