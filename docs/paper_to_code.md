# Paper-to-code correspondence

Implementation map for **OtoEvi: Local Evidence-Guided Otoendoscopic Report Generation**.

| Manuscript | Public implementation | Correspondence |
|---|---|---|
| III.B; Eq. 2 | `models/memory.py::_structured_multiscale_tokens` | Spatial/depth embeddings, gated local structure residual, top24 per level |
| III.B; Eq. 3 | `models/memory.py::VisualMemoryDecoder.build_memory` | M0 has 50 tokens; eight visual evidence tokens give M=58 |
| III.C; Eq. 4 | `models/evidence.py::EvidenceSlotResampler` | 12 queries per real view, conditioned on its global feature |
| III.C; Eqs. 5–6 | `models/evidence.py::SparseEvidenceReader` | Four-head sparsemax, query-dependent gate, independent final-two-layer readers |
| III.D.1 | `data/dataset.py`, `data/clauses.py` | Up to 10 clauses × 32 tokens; full-report negative filtering |
| III.D.2; Eq. 7 | `training/alignment.py::slot_clause_loss` | Log-mean-exp similarity over valid slots, temperature .07 |
| III.D.2; Eqs. 8–9 | `training/alignment.py::unique_positive_mask`, `coverage_loss` | Distinct-positive forward average; pooled reverse term |
| III.E; Eq. 10 | `training/objective.py`, `training/generation.py` | Unweighted CE + .05 report contrast + .015 clause loss |
| III.E; Eq. 11 | `training/gradients.py` | Conflict projection and norm cap at slots; weighted before accumulation |
| III.F | `training/engine.py`, `data/transforms.py` | Fixed final epoch 15, AdamW, BF16 training, FP32 generation |
| S1.1 | `models/vision.py` | Per-view FOV fallback, no batch-wide unmasking |
| S1.4 | `evaluation/ten_concept_evaluator.py` | Frozen ten-concept F1, explicit negatives and laterality |
| S6 | `models/ablations.py` | Complete 2×2×2 memory / reading / CCL factorial |

Paths in the table are relative to `src/otoevi/`.

## Naming and ablations

A = Multi-depth Visual Memory, B = View-preserving Sparse Reading, CCL = Clause Coverage Learning. The tensor C in the memory equations denotes the eight-token visual branch within A.

Removing A removes both M0 and the eight visual C tokens and uses one projected global token. Removing B removes sparse reading at inference. If CCL remains enabled, the local slot carrier still runs during training: the main gradient at the slots is explicitly zero, so the zero-main-gradient rule applies. Shared view-global projection parameters connect that auxiliary route to the generator. This is not the same as removing all local machinery during training.

The result tables cover all eight memory / reading / CCL configurations. The `no_gradient_coordination` option keeps the full model and CCL objective but uses ordinary backpropagation; scores for this additional control are not included.

## Initialization and checkpoints

Parameter initialization uses a fixed constructor order to make seeded experiments reproducible. The `initialization_bank_size=384` setting controls temporary parameter allocation during initialization; these temporary parameters are removed before training and are not part of the forward computation or saved model. Keep this setting fixed when reproducing the reported results.

Checkpoint loading validates all active model parameters with `strict=True`. Resume training with a checkpoint produced by `otoevi-train`, using the same data and configuration.
