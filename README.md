<div align="center">

<h1>FUSER</h1>

<h3>Factor-Utilization-Guided Selective Evidence Refinement<br>for Reliable Multimodal Sentiment Analysis</h3>

<p><strong>Discover factors · Audit predictive dependence · Learn selective adjustments</strong></p>

<p>Xiaoran Kou · Dingkang Yang · Jingyi Wu · Yang Liu · Jiangchuan Liu · Hong Chen</p>

<p>
  <a href="https://github.com/ma-tubu/FUSER"><img src="https://img.shields.io/badge/Code-PyTorch%20implementation-253858?style=flat-square&logo=github&logoColor=white" alt="PyTorch implementation"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.5 or newer">
  <img src="https://img.shields.io/badge/Modalities-Text%20%C2%B7%20Audio%20%C2%B7%20Vision-477C6C?style=flat-square" alt="Text, audio and vision">
</p>

<p>
  <a href="#overview">Overview</a> ·
  <a href="#method">Method</a> ·
  <a href="#visualization">Visualization</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#training">Training</a> ·
  <a href="#project-structure">Project structure</a> ·
  <a href="#citation">Citation</a>
</p>

</div>

<p align="center">
  <img src="assets/images/fig2.png" width="1100" alt="FUSER: factor discovery, utilization auditing, selective refinement and reliability-calibrated fusion">
  <br>
  <sub>From localized factors to selective refinement and post-refinement reliability.</sub>
</p>

---

## Overview

Reliable multimodal sentiment prediction depends on the evidence behind a decision.
A confident modality can still rely on incidental patterns, while a highly influential factor
can carry useful affective information.

**FUSER connects factor-level utilization auditing with task-guided selective refinement.**
It discovers recurring patterns, measures how a fixed predictor responds to localized
interventions, and learns bounded representation adjustments. Reliability is then recalibrated
on the refined representations before directed cross-modal fusion.

> **Core idea**
>
> Predictive reliability estimates how trustworthy a modality is. Factor utilization measures
> how strongly a particular factor affects its prediction. The sentiment objective guides
> whether, and how much, that evidence should be adjusted.

<p align="center">
  <img src="assets/images/fig1.png" width="480" alt="Motivation: reliability and evidence dependence provide different information">
  <br>
  <sub>Reliability alone does not reveal which local factors drive a prediction.</sub>
</p>

### Highlights

<table>
  <tr>
    <td width="50%" valign="top">
      <strong>🔎 Structure-aware discovery</strong><br>
      Multiscale local evidence and global summaries retain their input and hidden supports.
      Observed training descriptors form cosine K-medoids factor banks.
    </td>
    <td width="50%" valign="top">
      <strong>🧪 Factor-specific auditing</strong><br>
      Localized, structurally admissible interventions probe a frozen encoder and predictor,
      producing utilization scores without inference-time labels.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <strong>🎚️ Selective refinement</strong><br>
      A shared gate combines four audit diagnostics with a modality embedding.
      The downstream sentiment loss learns a bounded adjustment strength.
    </td>
    <td width="50%" valign="top">
      <strong>🔀 Recalibrated fusion</strong><br>
      Error estimators operate on refined representations. Source reliability weights
      six directed interactions between text, audio and vision.
    </td>
  </tr>
</table>

## Method

FUSER consists of four components:

| Step | Component | What it contributes |
|:---:|---|---|
| **01** | **Structure-aware factor discovery** | Extract local spans/windows and global summaries, fit training-only factor banks, and localize matching evidence. |
| **02** | **Factor-specific utilization auditing** | Retrieve a context-compatible training reference, apply a local interpolation, and measure the fixed predictor's response after structural checks. |
| **03** | **Utilization-informed selective refinement** | Form a support-masked counterfactual direction and learn its bounded coefficient from sentiment prediction. |
| **04** | **Reliability-calibrated multimodal fusion** | Re-estimate reliability on refined features, weight six directed attention branches by source reliability, and regress sentiment. |

### From dependence to a learned adjustment

The audit measures a factor's predictive influence:

$$
U_m = \tanh\!\left(\frac{|f_m(H_m^{\mathrm{cf}})-f_m(H_m)|}{\tau_u}\right).
$$

The gate uses utilization, matching strength, edit coverage and off-target drift to determine
a bounded coefficient. Only the selected hidden support contributes to the residual update:

$$
\widetilde H_m = H_m + \alpha_m M_m^H \odot (H_m^{\mathrm{cf}}-H_m),
\qquad 0 \leq \alpha_m \leq \alpha_{\max}.
$$

Strong utilization can arise from useful affective cues. The coefficient is therefore learned
through the sentiment objective, with a regularizer that penalizes unnecessary refinement.
When no admissible factor is available, the representation remains unchanged.

<details>
<summary><strong>View the reliability and evidence-dependence perspective</strong></summary>

<p align="center">
  <img src="assets/images/fig3.png" width="820" alt="Conceptual reliability-guidance, shortcut and affective-evidence pathways">
</p>

This conceptual view distinguishes modality-level reliability from factor-level evidence
dependence. The interventions audit model responses in feature space; the resulting patterns
are candidate factors rather than independently verified causal variables.

</details>

## Visualization

### How factor banks organize evidence

<p align="center">
  <img src="assets/images/fig4.png" width="1050" alt="Two-dimensional visualization of factor assignments across the three modalities">
  <br>
  <sub>Local evidence grouped by factor assignment, shown separately for text, audio and vision.</sub>
</p>

Factor discovery provides recurring patterns and their supports. Utilization auditing then
examines how a predictor responds to interventions on a matched pattern. This separation
allows the gate to account for both structural admissibility and downstream correction utility.

## Quick start

### Install

```bash
git clone https://github.com/ma-tubu/FUSER.git
cd FUSER
pip install -e .
```

Python **3.10+** and PyTorch **2.5+** are required. The reference software configuration uses
PyTorch **2.6.0**. Install the PyTorch build appropriate for your hardware before installing
the package if you need CUDA support.

### Check the model interface

The following example uses small random feature tensors to check the forward interface.
It constructs the fusion-initialization phase; input feature dimensions and temporal lengths
are supplied explicitly.

```python
import torch

from fuser.utils import build_backbone, build_model, load_config

config = load_config("configs/model.json")
feature_dims = {"text": 8, "audio": 6, "vision": 4}
backbone = build_backbone(feature_dims, config)
model = build_model(backbone, config)
model.set_phase("initialization")
model.eval()

inputs = {
    "text_features": torch.randn(2, 5, 8),
    "attention_mask": torch.ones(2, 5, dtype=torch.bool),
    "audio": torch.randn(2, 7, 6),
    "audio_mask": torch.ones(2, 7, dtype=torch.bool),
    "vision": torch.randn(2, 4, 4),
    "vision_mask": torch.ones(2, 4, dtype=torch.bool),
}

with torch.no_grad():
    prediction = model(inputs)["prediction"]

print(prediction.shape)  # torch.Size([2])
```

For factor auditing, fit `FactorBank` and `ReferencePool` on training evidence and attach
the resulting `FactorAuditor`. The configuration factory
`build_auditor(training_evidence, config, prototype_budget)` constructs these components
with an explicit prototype budget.

### Configuration guide

| Configuration | Controls |
|---|---|
| [`configs/model.json`](configs/model.json) | Encoders, evidence scales, factor-bank support, reference retrieval, structural checks, refinement and fusion. |
| [`configs/optimization.json`](configs/optimization.json) | Phase-specific learning rates, weight decay, gradient accumulation, gradient clipping and objective parameters. |

The shared architecture uses **128-dimensional representations**, **two** acoustic/visual
Transformer layers, **four** attention heads, and a **16-unit** refinement gate with a
**four-dimensional** modality embedding. Input dimensions, prototype budgets and numerical
settings remain configurable. Each modality preserves its own valid-position mask.

## Training

### Optimization schedule

| Model | Phase | Trainable components | Reliability used by fusion |
|---|---|---|---|
| Unimodal backbone | `base` | Encoders and unimodal sentiment heads | — |
| Unimodal backbone | `error` | Raw error estimators | — |
| FUSER | `initialization` | Fusion; refinement disabled | Raw |
| FUSER | `gate_warmup` | Refinement gate; fusion frozen | Raw |
| FUSER | `refinement` | Refinement gate and fusion | Raw |
| FUSER | `calibration` | Post-refinement error estimators; gate frozen | Error-estimation outputs only |
| FUSER | `fusion` | Fusion; gate and error estimators frozen | Calibrated post-refinement |

Factor-bank construction and discrete auditing take place after unimodal fitting.
After the gate converges, fit the post-refinement error heads on the resulting representations
and call `complete_calibration()`. Final fusion requires fitted post-refinement heads.

### Training with supplied feature batches

Given a fitted backbone, training factor bank and corresponding audits, select the phase
before constructing its optimizer and trainer:

```python
from fuser.train import Trainer, build_optimizer
from fuser.utils import load_config

optimization = load_config("configs/optimization.json")
model.set_phase("refinement")
optimizer = build_optimizer(model, optimization)
trainer = Trainer(model, optimizer, optimization)

trainer.train_batch(inputs, labels, audits=audits)
trainer.flush()
```

- **Task learning:** Smooth-L1 sentiment loss plus a mean-squared refinement-coefficient penalty.
- **Error estimation:** Mean-squared loss against `tanh(abs(prediction - label) / tau_e)` targets.
- **Accumulation:** Sample-weighted gradients with support for a partial final window through `flush()`.
- **Phase transitions:** Create a new optimizer and trainer after changing the phase.
- **Training-only supervision:** Externally normalized out-of-fold or held-out error targets can be passed through `error_targets`.

The detailed [method and API reference](docs/method.md) describes tensor shapes, masks,
audit normalization, cached-audit checks, calibration and checkpoint restoration.

## Project structure

```text
FUSER/
├── fuser/
│   ├── models/
│   │   ├── backbone.py        # Unimodal encoders and prediction/error heads
│   │   ├── encoders.py        # Cached text projection and temporal encoders
│   │   ├── evidence.py        # Structured evidence extraction
│   │   ├── factors.py         # Cosine K-medoids and factor localization
│   │   ├── audit.py           # Reference retrieval, intervention and utilization
│   │   ├── refinement.py      # Bounded, utilization-informed refinement
│   │   ├── reliability.py     # Post-refinement error estimation
│   │   ├── attention.py       # Six directed cross-modal interactions
│   │   ├── fusion.py          # Reliability-weighted sentiment regression
│   │   ├── records.py         # Evidence and factor records
│   │   └── model.py           # FUSER composition and phase control
│   ├── train/
│   │   ├── losses.py          # Unimodal, error-estimation and refinement losses
│   │   ├── optim.py           # Phase-specific AdamW construction
│   │   └── trainer.py         # Training steps and gradient accumulation
│   └── utils/
│       ├── config.py          # Configuration loading and component factories
│       └── tensors.py         # Nested tensor device transfer
├── configs/                  # Shared model and optimization settings
├── assets/
│   ├── images/               # PNG figures for documentation
│   └── figures/              # Original vector figures
├── docs/
│   └── method.md             # Method and API reference
├── pyproject.toml
└── requirements.txt
```

## Citation

If this work is useful to your research, please cite:

```bibtex
@misc{kou2026fuser,
  title  = {FUSER: Factor-Utilization-Guided Selective Evidence Refinement
            for Reliable Multimodal Sentiment Analysis},
  author = {Kou, Xiaoran and Yang, Dingkang and Wu, Jingyi and
            Liu, Yang and Liu, Jiangchuan and Chen, Hong},
  year   = {2026},
  url    = {https://github.com/ma-tubu/FUSER}
}
```

## Contact

For questions about the method or implementation:

- **Xiaoran Kou:** [2451806@tongji.edu.cn](mailto:2451806@tongji.edu.cn)
- **Yang Liu:** [yangliu25@tongji.edu.cn](mailto:yangliu25@tongji.edu.cn)

---

<div align="center">
  <strong>FUSER</strong><br>
  <sub>Factor-level auditing · Selective refinement · Recalibrated fusion</sub>
</div>
