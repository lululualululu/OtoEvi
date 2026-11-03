# Paper-to-code correspondence

Target: **OtoEvi: Local Evidence-Guided Otoendoscopic Report Generation**, November 3, 2026 reading copy. Its hash and the frozen research commit are recorded in [provenance](../results/provenance.json).

| Manuscript | Public implementation | Correspondence |
|---|---|---|
| III.B; Eq. 2 | `models/memory.py::_structured_multiscale_tokens` | Spatial/depth embeddings, gated local structure residual, top24 per level |
| III.B; Eq. 3 | `models/memory.py::VisualMemoryDecoder.build_memory` | M0 has 50 tokens; eight visual evidence tokens give M=58 |
| III.C; Eq. 4 | `models/evidence.py::EvidenceSlotResampler` | 12 queries per real view, conditioned on its global feature |
| III.C; Eqs. 5–6 | `models/evidence.py::SparseEvidenceReader` | Four-head sparsemax, query-dependent gate, independent final-two-layer readers |
| III.D.1 | `data/dataset.py`, `data/clauses.py` | Up to 10 clauses × 32 tokens; full-report negative filtering |
| III.D.2; Eq. 7 | `training/alignment.py::slot_clause_loss` | Log-mean-exp similarity over valid slots, temperature .07 |
| III.D.2; Eqs. 8–9 | `training/alignment.py::unique_positive_mask`, `coverage_loss` | Distinct-positive forward average; original pooled reverse term |
| III.E; Eq. 10 | `training/objective.py`, `training/generation.py` | Unweighted CE + .05 report contrast + .015 clause loss |
| III.E; Eq. 11 | `training/gradients.py` | Conflict projection and norm cap at slots; weighted before accumulation |
| III.F | `training/engine.py`, `data/transforms.py` | Fixed final epoch 15, AdamW, BF16 training, FP32 generation |
| S1.1 | `models/vision.py` | Per-view FOV fallback, no batch-wide unmasking |
| S1.4 | `evaluation/ten_concept_evaluator.py` | Frozen ten-concept F1, explicit negatives and laterality |
| S6 | `models/ablations.py` | Complete 2×2×2 memory / reading / CCL factorial |

Paths in the table are relative to `src/otoevi/`.

## Naming and ablations

A = Multi-depth Visual Memory, B = View-preserving Sparse Reading, CCL = Clause Coverage Learning (historical D1). The tensor C in the memory equations is the eight-token visual branch within A, not the retired clause objective.

Removing A removes both M0 and the eight visual C tokens and uses one projected global token. Removing B removes sparse reading at inference. If CCL remains enabled, the existing slot carrier still runs during training: the main gradient at the slots is explicitly zero, so the published zero-main-gradient exception applies. Shared view-global projection parameters connect that auxiliary route to the generator. This is not the same as removing all local machinery during training.

The final factorial scores include all eight configurations. The `no_gradient_coordination` option is supplied as a mechanism control, without claiming a completed final-CCL score. Historical pooled-positive controls in the manuscript are separate archived experiments; this package defaults to CCL.

## Initialization and checkpoint compatibility

The paper experiments instantiated unused historical parameters. To preserve their exact random initialization of active parameters, `_initialization.py` reproduces the constructor's RNG draws. `compatibility.py` then deletes those inactive modules before training. No dynamic prefix, semantic anchor injection or anchor loss is executed or stored in a new checkpoint. The scalar `initialization_bank_size=384` remains only as the historical initialization-bank size; there is no need to mine a semantic dictionary.

Historical D1 / H-series checkpoints are loaded by discarding only an explicit allowlist of retired keys, then applying `strict=True` to every remaining parameter. Other old variants are rejected. Optimizer state from the old full-storage layout is not automatically migrated; resume applies to checkpoints produced by this package.

## Manuscript availability update

The supplied PDF describes the old access-controlled repository, and S1.6 warns that its packaged CLI uses an earlier architecture. Those statements describe that archived repository; they no longer describe this public package. Suggested replacement:

> The public implementation is available at https://github.com/lululualululu/OtoEvi under the MIT License. The packaged training, inference and evaluation entry points implement the final Multi-depth Visual Memory, View-preserving Sparse Reading and Clause Coverage Learning framework. The release includes fixed configurations, public-dataset split reconstruction, evaluation code and aggregate experiment records. The complete OtoInsight dataset and trained report-generation checkpoints are not distributed. The original experimental implementation is archived at research commit e4d0f8e.

中文建议：

> 最终实现公开于 https://github.com/lululualululu/OtoEvi，采用 MIT 许可证。打包的训练、推理和评估入口实现最终的多深度视觉记忆、保留视图的稀疏读取及子句覆盖学习框架。发布内容包括固定配置、公开数据集划分重建工具、评估代码及实验汇总记录。完整 OtoInsight 数据集和报告生成检查点不分发。原始实验实现归档于研究提交 e4d0f8e。

No changes have been made to the supplied manuscript PDF. This repository does not assert publication or acceptance status.
