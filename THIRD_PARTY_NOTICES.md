# Third-party components

The MIT license in this repository covers the original OtoEvi source release. It does not relicense separately downloaded dependencies, datasets or pretrained weights.

| Component | Use | Upstream |
|---|---|---|
| PyTorch / torchvision | Tensor operations, transformers, optimization, image transforms | https://github.com/pytorch/pytorch ; https://github.com/pytorch/vision |
| timm | DINOv3 model implementation and weight loading | https://github.com/huggingface/pytorch-image-models |
| DINOv3 weights | Frozen generic visual encoder, obtained separately | https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m |
| pycocoevalcap | Corpus BLEU, ROUGE-L and CIDEr | https://github.com/salaniz/pycocoevalcap |
| NumPy / Pillow / safetensors | Numerical utilities, images, weight serialization | Their installed distributions |

The DINOv3 model card identifies a separate **DINOv3 license**. Follow the upstream access and use conditions; its weights are not included in this repository. Source dataset images/captions retain their applicable distribution terms. The overview figure reproduces Figure 1 from the OtoEvi paper, including its illustrative otoendoscopic images; the OtoInsight dataset and individual clinical records are currently withheld from public release to protect patient privacy and confidential clinical information. Public-dataset images and captions should be obtained from their original or authorized providers.
