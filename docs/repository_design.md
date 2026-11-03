# Repository organization

Reviewed on 2026-11-03 using official repositories:

- [DINOv2](https://github.com/facebookresearch/dinov2): a named method package, configuration separation, distinct training/evaluation guidance, explicit model availability.
- [MAE](https://github.com/facebookresearch/mae): concise project landing page and separate task-specific reproduction instructions.
- [R2Gen](https://github.com/cuhksz-nlp/R2Gen): direct dataset preparation, environment requirements, run commands and citation.

OtoEvi adopts these presentation principles with a `src` package so installed imports do not depend on the research working directory. Modules use paper names, not dated experiment filenames. Configs describe reproducible runs; scripts prepare external data; tests check the mechanisms; results contain compact, non-patient-level evidence. The old research history and abandoned architectures are not copied into the new Git history.

`_initialization.py` is the one compatibility boundary: it reproduces the original constructor's random draws and immediately discards inactive heads. This preserves experimental initialization while exposing only the final method in the forward/training graph.
