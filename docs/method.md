# Model and API reference

Core implementation of **Factor-Utilization-Guided Selective Evidence Refinement**.

## Modules

| Module | Component |
| --- | --- |
| `backbone.py`, `encoders.py` | Cached linguistic projection, acoustic/visual Conv1D and Transformer encoders, unimodal prediction |
| `evidence.py`, `records.py` | Local multiscale evidence and global summaries with input/hidden supports |
| `factors.py` | Training-fitted normalization, cosine K-medoids, support screening and sample-specific localization |
| `audit.py` | Training-reference retrieval, local input interpolation, structural admissibility and utilization |
| `refinement.py` | Four audit diagnostics, modality embedding and bounded residual refinement |
| `reliability.py` | Error estimation on refined representations |
| `attention.py`, `fusion.py` | Six directed interactions weighted by source reliability and sentiment regression |
| `models/model.py`, `train/losses.py` | Model composition, optimization phases and objectives |

## Inputs

Python 3.10+ and PyTorch 2.5+ are required. The reference software configuration uses PyTorch 2.6.0.

`UnimodalBackbone(input_dims)` accepts positive input dimensions keyed by `text`, `audio` and `vision`.
Its defaults are a hidden dimension of 128, two acoustic/visual Transformer layers, four heads,
a feed-forward dimension of 256, and 64-unit unimodal/error heads. `dropout` is configurable.
`output_activation="tanh"` applies the same bounded prediction convention to unimodal and fused outputs.

The model input mapping contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `text_features` | `[B, L_t, D_t]` | Cached continuous linguistic features |
| `attention_mask` | `[B, L_t]` | Valid linguistic positions |
| `lexical_mask` | `[B, L_t]` | Optional pooling/edit mask, contained in `attention_mask` |
| `audio`, `vision` | `[B, L_m, D_m]` | Continuous modality features |
| `audio_mask`, `vision_mask` | `[B, L_m]` | Independent valid-position masks |
| `audio_available`, `vision_available` | `[B]` | Optional modality availability |
| `audio_observed_mask`, `vision_observed_mask` | `[B, L_m]` | Optional observed-position masks for interventions |
| `sample_ids`, `group_ids` | Length `B` | Identifiers used to exclude the same sample and source from reference retrieval |

The three temporal lengths may differ. Position preservation is checked within each modality.
Mask values are Boolean, and features must be finite at valid positions.
Labels are supplied to objective functions separately from model inputs.

## Factor discovery and auditing

`EvidenceExtractor` takes `scales`, a mapping from modality to positive window widths.
Optional `strides` configure each width. Set `global.enabled=True` and `global.modalities`
to enable global summaries. Local supports never cross invalid positions.

Fit `FactorBank` and `ReferencePool` on training evidence, then construct
`FactorAuditor(extractor, bank, references, config)`. Bank configuration includes
the required `clusters_per_group`, `feature_source` (`X`, `early` or `H`), support limits and a fitting seed.
The prototype budget applies per modality and scope, across the selected scales.
`selected_scales` can restrict local scales per modality. `presence_threshold` accepts a scalar
or a modality mapping; otherwise it is fitted using the configured training similarity quantile.

Auditing uses one highest-scoring eligible local factor per sample and modality.
The default interpolation strength is 0.2. Structural checks require target change,
unchanged off-target inputs, coverage at most 0.5 and off-target hidden drift at most 0.25.
Coverage and hidden drift are measured on valid positions. Utilization is
`tanh(abs(counterfactual_prediction - prediction) / tau_u)`.

The refinement gate receives `[utilization, matching_score, coverage, off_target_drift]`
and a four-dimensional modality embedding. Its default hidden size is 16.
The coefficient is `alpha_max * sigmoid(logit)`, with `alpha_max=0.3`, and becomes zero
for inadmissible or unavailable evidence. Only the selected hidden support is updated.
Diagnostic normalization is fitted from eligible training audits.

## Optimization phases

Train the backbone using `set_phase("base")` and `unimodal_loss`, followed by
`set_phase("error")` and `reliability_loss`. `normalized_error_targets` accepts detached
training-only out-of-fold or held-out predictions as well as frozen-head predictions.
The normalized target is `tanh(abs(prediction - label) / tau_e)`.

Construct `FUSER(backbone, auditor)` after fitting the unimodal model and raw error heads.
`set_phase` controls trainable components:

| Phase | Trainable components | Reliability used by fusion |
| --- | --- | --- |
| `initialization` | Fusion | Raw reliability; refinement disabled |
| `gate_warmup` | Refinement gate | Raw reliability; fusion frozen |
| `refinement` | Refinement gate and fusion | Raw reliability |
| `calibration` | Post-refinement error heads | Error-estimation outputs only |
| `fusion` | Fusion | Calibrated post-refinement reliability; gate frozen |

Use `refinement_loss` for task learning and coefficient regularization.
During calibration, form targets for `post_unimodal`, and apply `reliability_loss`
to `post_error`. Fit against training-only targets with the converged gate fixed.
After calibration optimization, call `complete_calibration()`, select `fusion`,
and fit the final fusion layers. Final fusion rejects unfitted post-refinement heads.
Re-entering an earlier phase invalidates the calibration flag.

`audit_inputs(inputs)` computes detached audits. Reuse them through `forward(inputs, audits)`
while the backbone remains fixed. `reference_h` is checked to reject stale representations.
`fit_diagnostic_normalization(audits)` fits the four diagnostic features on training evidence.
Refit calibration if diagnostic normalization changes.

Model `state_dict()` includes fitted factor banks and reference pools when an auditor is attached.
Restore the same architecture, load the state, select the intended phase and call `eval()` for inference.

## Repository interfaces

Core classes live in `fuser.models`; the public imports `from fuser import FUSER`
and `from fuser import UnimodalBackbone` remain available. Losses are in
`fuser.train.losses`, phase-specific AdamW construction is in `fuser.train.optim`,
and `fuser.train.Trainer` handles training batches and gradient accumulation.
`fuser.utils` provides JSON configuration loading, model factories and nested tensor transfer.

`configs/model.json` contains shared model settings. Feature dimensions and prototype budgets
remain explicit construction arguments. Override numerical model and objective parameters
for the intended experimental setting before model construction.

`Trainer.train_batch` accepts `inputs`, `labels`, optional precomputed `audits`, and optional
normalized `error_targets`. `flush()` applies a remaining partial accumulation window.
Call it at the end of each epoch and before phase changes. Error-estimation targets may
come from caller-provided training-only out-of-fold or held-out predictions; constructing
those partitions remains separate from the model's optimization step.
