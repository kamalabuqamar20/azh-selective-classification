# Proposed study: when does image quality add to calibrated confidence?

**Status:** research proposal for development with a supervisor. It is not a
preregistration, and no new patient-independent AZH result, external-site
evaluation, or clinical study is claimed here.

## Question and rationale

For wound-type predictions on images from patients excluded from classifier
training, probability calibration and selector development, how does adding
a prespecified image-sharpness measure to an otherwise equivalent selector
based on calibrated confidence change the error rate among retained
predictions at approximately 80% image coverage?

A secondary question is whether the paired difference remains similar under
changes in image acquisition. The 80% target is a research operating point,
not a clinically validated review capacity. Retaining a prediction means it is
included in the evaluation; in a hypothetical future workflow the remainder
could be sent for human review. No human review or patient outcome was measured
in the present project.

Marginal probability calibration does not imply that all images assigned the
same confidence have the same error rate in different quality conditions. A
quality feature might help rank correctness, but it may also reflect wound
texture, image resolution or acquisition practices, and may add no value.
All outcomes, including zero or adverse effects, are scientifically relevant.

## What is already known

- [Zhang et al. (2021)](https://doi.org/10.2196/26149) used variance of the
  Laplacian for quality feedback in a wound-imaging tool. The present project
  does not invent that focus measure.
- [Lott et al. (2026)](https://pubmed.ncbi.nlm.nih.gov/42314450/) studied
  calibrated confidence and selective prediction for pressure-injury staging,
  including an external dataset. The current proposal must not be described
  as the first selective wound-imaging system.
- [Choi et al. (2026)](https://pubmed.ncbi.nlm.nih.gov/42870149/) studied
  cross-dataset pressure-injury staging and found substantial variation in
  performance and error patterns. A change of site can alter more than image
  sharpness, including patients, labels, devices and workflows.

The candidate contribution is a carefully controlled *incremental-value*
comparison for a specified wound-type classification task. This brief review
is not exhaustive and cannot establish that the precise comparison is novel.
The target, baselines and scope must be revisited after a full review.

## Connection to the current work

The current AZH experiments compare selection policies, image-quality
sensitivity and calibration, but patient identifiers were unavailable for
grouping. The separate LUTSeg experiment uses five patient-grouped folds on a
binary label derived from necrotic-tissue annotations; it is a different task,
not external validation of the AZH classifier. In the unweighted frozen
ResNet-feature LUTSeg pipeline, the learned confidence-only and joint
selectors each retained 33 errors out of 115 selected images, while the base
classifier detected 5 of 32 positive images before selection. The result does
not establish a benefit from adding focus. See [the pilot](LUTSEG_PATIENT_PILOT.md)
for class-specific counts, the reference comparator and limitations.

## Candidate design

1. **Data and reference standard.** Identify wound-type data with adequate
   patient identifiers and well-defined reference labels; document provenance,
   permissions, class sizes and repeated images. Assess whether an image-only
   task is appropriate when reference labels use clinical information that is
   not present in the photograph. A patient-grouped dataset for this exact
   four-class task has not been secured. If unavailable, revise the task with a
   supervisor instead of labelling a different outcome as the same task.
2. **Partitions and fitting.** Separate patients before preprocessing or
   fitting. Use disjoint development subsets, or a prespecified grouped
   cross-fitting approach, for the base classifier, probability calibration,
   correctness selector and decision threshold. Keep a final test cohort out
   of all model choices. Document pretraining provenance and any inability to
   audit overlap with pretrained datasets.
3. **Paired comparison.** Keep the classifier and its calibrated probabilities
   fixed. Fit comparable confidence-only and confidence-plus-sharpness
   correctness selectors using the same development patients and fitting
   procedure; only the prespecified sharpness input differs. Report maximum
   calibrated confidence and sharpness-only selection as additional reference
   rules. Selectors use no true test labels as inputs.
4. **Primary estimand.** On the same held-out cohort, rank each policy's
   predictions and retain approximately 80% of images, with a prespecified
   tie rule. Report error among retained images for each policy and the paired
   difference in percentage points: joint minus confidence-only. Negative
   values favour the joint rule. The selected images may differ between
   policies; equal image coverage need not mean equal class or patient coverage.
5. **Operational evaluation.** Ranking a completed test batch describes a
   matched-coverage comparison; it does not provide a threshold fixed before
   individual cases arrive. Separately choose each policy's threshold on
   development data, freeze it, and report achieved test coverage, retained
   error and review load. Do not assume the target coverage will transfer.
6. **Uncertainty and safeguards.** Before final evaluation, specify sample
   size and the smallest effect worth detecting. For a suitable independent
   test cohort, use a paired patient-cluster bootstrap or another justified
   patient-aware method, keeping all images from a sampled patient together.
   State whether the interval conditions on fixed trained models and selectors
   or also reflects refitting. Report per-class errors, deferral rates and
   sensitivity where meaningful, with denominators and uncertainty; rare
   classes may yield imprecise estimates.
7. **Shift and transfer.** Prespecify controlled blur and, if available,
   naturally varying acquisition conditions. An independent external site is
   a further test only if the same target and label definitions are compatible
   and patients do not overlap. Do not attribute all changes between sites to
   image quality. A clinical workflow or prospective benefit requires a
   separate study.

## Feasibility and interpretation

Availability of relevant patient-linked labels and enough patients per class
is the main dependency; no such new cohort is claimed here. Small groups,
uncertain labels and overlapping train sets in cross-validation can make
intervals unstable or incomplete. If the added feature is unhelpful, that is a
valid finding; a small sample cannot establish that the true effect is zero.
Quality-based deferral could also redistribute missed cases or review burden
across groups, so an overall error reduction alone would not show safety or
clinical value.

The evaluation and code are intended as a research foundation, not an
autonomous diagnosis, triage or treatment system.
