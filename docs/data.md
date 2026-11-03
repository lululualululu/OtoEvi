# Data preparation and availability

## OtoInsight

The manuscript describes 20,220 examinations from 8,422 patients. Fixed partitions contain 15,450 train, 1,583 validation and 3,187 temporal-test reports (6,265/681/1,476 patients). The current CCL table in this release is the **validation** evaluation. Do not label historical pooled-positive 2026 results as final-CCL results.

The complete clinical dataset, its individual reports/images and trained task checkpoints are not distributed here. No public download URL or access-approval promise is supplied. The small example manifests are fictional schema examples, not data for reproducing the paper.

A manifest is a nonempty JSON list:

- `report_id`: unique examination/report identifier within the split.
- `patient_id`: de-identified identifier for patient-disjoint splitting and negative filtering.
- `filenames`: ordered paths relative to the image root; the first two real images are used.
- `caption`: nonempty original report text.

Use consistent patient identifiers across all splits. Train/validation patient, report and filename overlap is rejected. Filename checks do not replace image-similarity or patient-identity auditing. Vocabulary and clause statistics are fit on training reports only.

## Public datasets

Obtain images and source image-caption TSV files from the dataset's original/authorized distribution. A TSV row is `image_filename<TAB>caption`. The release does not redistribute the images or complete source captions. Its specifications store public image IDs, caption hashes and source-span edits sufficient to reconstruct the exact cohort from matching source files.

```bash
python scripts/prepare_public_data.py --dataset iu_xray \
  --annotations /path/to/iu/train_images.tsv /path/to/iu/test_images.tsv \
  --output data/iu_xray

python scripts/prepare_public_data.py --dataset peir_gross \
  --annotations /path/to/peir/peir_gross.tsv \
  --output data/peir_gross
```

Then place the corresponding images under `data/<dataset>/images`, or edit `image_root` in the config. Image filenames must match. Source-file layout differences require an explicit conversion to TSV, not a new random split. Preparation verifies every selected source caption and reconstructed target hash. `patient_id: null` explicitly denotes unavailable patient identity in these manifests; image/report independence must not be called verified patient independence.

| Dataset | Train | Validation | Test | Reference text |
|---|---:|---:|---:|---|
| IU X-Ray | 3,099 | 344 | 383 | Original supplied descriptions |
| PEIR Gross, study-revised | 5,489 | 610 | 745 | History-reduced descriptions |

## What the PEIR revision means

This is **our study's revised protocol**, not an official corrected PEIR release. The earlier local collection already contained explicit age/history narratives and cross-split related images. Our initial duplicate checks did not detect all near-duplicate or specimen-series relationships.

The revised protocol kept the 745 test cases/order, conservatively grouped related images/descriptions, excluded 599 development examples in total and reduced explicit history in target text. Of those 599,508 were linked to test families and 97 required annotation review, with 6 in both groups. The resulting training/validation counts are 5,489/610. Eight flagged test annotations remain in evaluation. The full-library edit ledger changed 1,155 strings, including 991 with word-content changes; 125 test strings changed, 111 in word content.

Known groups do not cross partitions, but grouping does not establish complete patient independence. History-reduced references are not physician-validated visual-only labels. Test cases had already been inspected: this was post-hoc corrective re-evaluation, not a new blind test. Both split and supervision changed, so their separate effects cannot be inferred from the old/new score difference.

The frozen reconstruction applies precomputed source-span edits and split membership, not a new image-retrieval audit. It also emits `test_original.json` for scoring the same predictions against the original references. Source hashes prevent edits from being applied to a different text version.
