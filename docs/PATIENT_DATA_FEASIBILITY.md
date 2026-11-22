# Patient-grouped data feasibility audit

## Why inspect another dataset?

AZH lacks accessible patient identifiers for the current pilot. Its authors
state that some patients contributed multiple images, so exact-image
deduplication cannot establish patient independence. A patient-grouped test
requires reliable grouping, relevant labels and enough independent patients
per class; merely finding a larger number of image files is not enough.

## WoundsDB: metadata-only audit

We inspected the publicly downloadable `description.tab` file in
[Harvard Dataverse, DOI 10.7910/DVN/6A1HAG](https://doi.org/10.7910/DVN/6A1HAG),
file ID `14118414`, SHA-256
`ed5f27bcd21400168e45af85a018559ef75e51cf55cacb6700a74bfce480c31a`.
The provider's [data-organisation description](https://chronicwounddatabase.eu/DataOrganization)
describes `case_X` as a patient identifier and `day_Y` as a recording day.
The [dataset paper](https://doi.org/10.1016/j.compmedimag.2020.101844) reports
188 image sets from 79 patient visits and describes 47 patients in its cohort.

Our audit of that particular table found **188 photo records, 47 distinct case
identifiers and 79 case-day pairs**. One case group appears under more than one
source aetiology label; label-specific group counts therefore need not sum to
47. These are counts in the pinned table, not a guarantee about later releases.
The database [website overview](https://chronicwounddatabase.eu/) also states
"188 cases of 79 patients", which conflicts with the cited paper's 47 patients
and the 47 case identifiers in this table. We report the pinned table and paper
separately and do not assume the overview describes the same release.

| Source label as recorded | Photo records | Case groups |
|---|---:|---:|
| Venous ulcers | 74 | 17 |
| Venous insufficiency | 50 | 14 |
| Ischaemia | 24 | 6 |
| Diabetic foot | 8 | 2 |
| Wound infection | 7 | 4 |

Other source labels have only one case group each. Eight of the 11 labels occur
in fewer than five case groups. These are dataset-supplied aetiology or clinical
descriptors and do not form a verified mapping to AZH's four categories. For
example, amputation and wound infection should not be silently treated as AZH
surgical or diabetic labels. A case-group split cannot place an aetiology with
only one group in both training and testing; two diabetic-foot groups cannot
support a convincing patient-independent five-fold study.

**Decision:** WoundsDB is promising for investigating patient grouping and
future study design, but this metadata does not support the proposed four-way
patient-independent wound-type evaluation. We did not train or test a model on
WoundsDB, merge the clinical labels, or claim external validation. Independent
clinical guidance, suitable labels and a larger number of patient groups remain
necessary for the proposed evaluation.

For data with verified patient identifiers, `patient_splits.py` provides
group-disjoint outer folds and inner model/calibration/selector partitions. It
rejects missing identifiers, too few groups per class, and folds omitting a
class. Its offline tests use synthetic groups. It is now also used in a
[separate exploratory LUTSeg pilot](LUTSEG_PATIENT_PILOT.md), which has a
different derived tissue-presence target and is not a clinical model or an
external validation of AZH. Sufficient group counts are necessary but do not
by themselves guarantee representativeness or validity.

Harvard Dataverse describes its release as CC0 1.0, while the original paper
describes the source database as CC BY 4.0. Consult each source's terms before
reuse and cite the original creators. This repository redistributes no original
case-description table or clinical images, only aggregate counts.

## Reproduce the aggregate audit

```bash
python audit_woundsdb.py --description data/woundsdb/description.tab --download --output reproduced/woundsdb_audit
python verify_results.py --woundsdb reproduced/woundsdb_audit
```

Alternatively, obtain the file from the dataset provider and omit `--download`.
The script checks the pinned hash and emits only aggregated counts. The checked
results are in [`results/woundsdb_audit`](../results/woundsdb_audit/).
