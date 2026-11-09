# Data provenance and limitations

## Source

The image archives originate from the University of Wisconsin–Milwaukee Big Data Lab's [wound classification release](https://github.com/uwm-bigdata/wound-classification-using-images-and-locations), pinned at commit `e7dd90de113dcbd39665f2bd0e5613adeffff5fa`. The authors describe four wound types and two capture devices; some patients contributed more than one image. The code does not have patient- or device-level grouping.

| Archive | SHA-256 |
|---|---|
| `Train.zip` | `ebfff40b57e626ae38e7d27b163cebd86c8ddf793b4f64b3fd8a9bef0d5eeddc` |
| `Test.zip` | `55ef08b374b80bb3231d6fef106362f77c2becec1b79fb3dd2ce69a6d54c6760` |

The repository contains no images or individual-level records. It includes aggregate counts and metrics only. The download option retrieves the original archives from the pinned upstream commit and validates the hashes. Review the upstream terms and any applicable approvals before obtaining, using or redistributing the data; this code does not grant rights to the images. No licence is currently declared for this repository's own code.

## Audit observations

- Released archives contain 930 image files over six categories. One byte-identical surgical image appeared in both original splits; the training copy was excluded.
- After exact byte and decoded-pixel matching there are 929 unique images in the working copy; four wound classes account for 729.
- Before resizing, 17 images exceeded a simple dimension flag of long side >700 px or short side >525 px: 15 surgical and 2 normal-skin images. This is an observation about dimensions, not proof of different acquisition sources or bias.
- Within the 184 wound test images, median Laplacian variance after fixed resizing was approximately 134 for diabetic, 139 for pressure, 146 for surgical and 181 for venous images. This statistic can reflect texture and resolution as well as focus; it is not a clinical quality label.

## Scope

- **Different task:** AZH wound type is not the infection/ischaemia classification task proposed for DFUC2021.
- **Image-level splitting:** exact-match removal cannot rule out near-duplicates or repeated patients. Grouped validation requires reliable identifiers.
- **Single setting:** one source dataset, one handcrafted-feature classifier, one focus proxy and limited sample sizes do not establish external performance.
- **Retrospective analysis:** the original test result was inspected before designing the training-pool follow-up. Neither run is a preregistered confirmatory test.
- **Operational threshold:** the reported matched-coverage selection uses the test pool's score ordering. A deployed system would need a threshold fixed without access to the target test pool.
- **Clinical applicability:** no clinical usability, benefit, patient outcome, clinician agreement or prospective validation is established.
