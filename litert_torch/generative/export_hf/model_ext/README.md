# Pluggable Model Extensions (`model_ext`)

This directory contains the pluggable **`ModelExtension`** framework and all
built-in architecture extensions for `litert_torch.generative.export_hf`.

A `ModelExtension` encapsulates every architecture-specific customization needed
to export a Hugging Face (or custom PyTorch/Diffusers) checkpoint to LiteRT
(`.tflite`) and LiteRT-LM (`.litertlm`) across text generation, multimodal
(vision/audio), speech recognition (ASR), text-to-speech (TTS), text-to-image
diffusion, and Any2Any multimodal tasks—**without modifying `litert_torch` core
code**.

---

## Directory Structure

```text
model_ext/
├── __init__.py            # Public API exports & registration of built-ins
├── builtin_extensions.py  # Imports & registers built-in <arch>/extension.py
├── extension.py           # ModelExtension contract, composition, and registry
├── exportables.py         # Exportable dispatchers delegating to ModelExtension
├── patches.py             # Context managers for class & instance patching
├── metadata_builder.py    # LlmMetadata & ExecutorMetadata builder dispatchers
├── testing.py             # Reusable ModelExtensionTestMixin contract harness
├── extension_test.py      # Unit tests for registry, composition, and built-ins
└── <arch>/                # Self-contained per-architecture packages
    ├── extension.py       # @register_model_extension(override=False) subclass
    ├── patch.py           # Class/instance runtime patches for export
    ├── exportable_module.py / vision_exportable.py / audio_exportable.py
    └── metadata_builder.py
```

### Built-in Architecture Extensions

All built-in extensions are registered **by default** when `model_ext` or
`export_lib` is imported (with `@register_model_extension(override=False)` so
explicit user registrations take precedence), and `get_model_extension()` also
supports lazy fallback imports for `model_ext.<model_type>.extension`:

| Category | Subdirectory | Extension Class(es) | Registered `model_types` |
| :--- | :--- | :--- | :--- |
| **LLM / VLM** | `gemma3/` | `Gemma3Extension` | `gemma3`, `gemma3_text` |
| **LLM / VLM** | `gemma3n/` | `Gemma3nExtension` | `gemma3n` |
| **LLM / VLM / Audio** | `gemma4/` | `Gemma4Extension` | `gemma4`, `gemma4_text` |
| **VLM** | `gemma4_unified/` | `Gemma4UnifiedExtension` | `gemma4_unified` |
| **LLM** | `lfm2/` | `Lfm2Extension` | `lfm2` |
| **VLM** | `lfm2_vl/` | `Lfm2VlExtension` | `lfm2_vl` |
| **LLM** | `qwen3/` | `Qwen3Extension` | `qwen3`, `qwen3_text` |
| **LLM** | `qwen3_5/` | `Qwen3_5Extension` | `qwen3_5`, `qwen3_5_text` |
| **ASR (Speech)** | `moonshine/` | `MoonshineExtension` | `moonshine` |
| **ASR (Speech)** | `parakeet/` | `ParakeetCtcExtension`, `ParakeetTdtExtension` | `parakeet_ctc`, `parakeet_tdt` |
| **ASR (Speech)** | `qwen3/` | `Qwen3AsrExtension` | `qwen3_asr` |
| **ASR (Speech)** | `whisper/` | `WhisperExtension` | `whisper` |
| **TTS** | `kokoro_tts/` | `KokoroTtsExtension` | `kokoro`, `kokoro_tts`, `kokoro-82m`, `kokoro_82m` |
| **TTS** | `qwen3_tts/` | `Qwen3TtsExtension` | `qwen3_tts`, `qwen3_tts_talker` |
| **Text-to-Image** | `bonsai_flux2/` | `BonsaiFlux2Extension` | `bonsai_flux2` |
| **Text-to-Image** | `flux2_klein/` | `Flux2KleinExtension` | `flux2_klein`, `Flux2KleinPipeline` |
| **Any2Any** | `minicpm_o/` | `MiniCpmOExtension` | `minicpmo`, `minicpm_o`, `minicpm-o`, `minicpm-o-4.5`, `minicpm-o-4_5`, `MiniCPM-o-4_5`, `MiniCPMO` |

---

## Quick Start: Writing an Out-of-Tree `ModelExtension`

Subclass `ModelExtension` in a standalone `.py` file (or Python package), set
`model_types` to match `model_type` in the checkpoint's `config.json`, and
override only the hooks your architecture needs:

```python
# my_arch_extension.py
import contextlib
import dataclasses
from litert_torch.generative.export_hf.model_ext import extension
from litert_torch.generative.layers import normalization
import torch


@extension.register_model_extension()
class MyArchExtension(extension.ModelExtension):
  model_types = ('my_arch',)
  is_causal_text_model = True
  extra_kwargs_keys = ('key_ts_idx',)

  def update_export_config(self, export_config, model_config):
    del model_config
    if 'key_ts_idx' in export_config.extra_kwargs:
      export_config = dataclasses.replace(
          export_config, k_ts_idx=int(export_config.extra_kwargs['key_ts_idx'])
      )
    return export_config

  @contextlib.contextmanager
  def patch_model(self, model, export_config):
    del model, export_config
    yield
```

Then run `export_main` pointing `--extension_modules` at your `.py` file (or
importable dotted module name):

```bash
python -m litert_torch.generative.export_hf.export_main \
  --model=/path/to/checkpoint \
  --output_dir=$HOME/export_out \
  --extension_modules=./my_arch_extension.py \
  --trust_remote_code \
  --split_cache \
  --key_ts_idx=3 \
  --v_ts_idx=2
```

---

## Authoring Guide & Examples by Feature

### 1. Custom CLI Flags (`extra_kwargs_keys`) & `update_export_config`

By default, `ExportableModuleConfig` rejects unrecognized CLI flags with a
`ValueError` so mistyped flags are caught immediately. To add custom CLI flags
or override export settings for your model:

* **`extra_kwargs_keys: Sequence[str]`** — Declares custom CLI / `export(...)`
  keyword argument names accepted in `export_config.extra_kwargs`.
* **`update_export_config(self, export_config, model_config)`** — Inspects
  `export_config` (including `export_config.extra_kwargs`) and `model_config`
  after config loading and returns an updated `ExportableModuleConfig` via
  `dataclasses.replace(...)`. If `update_export_config` toggles pipeline stage
  flags (`externalize_embedder`, `split_cache`, `export_vision_encoder`,
  `export_audio_encoder`, etc.), `run_export_tasks` automatically rebuilds the
  export stage list.

**Built-in examples:** `model_ext/gemma4/extension.py`,
`model_ext/lfm2_vl/extension.py`.

```python
@extension.register_model_extension()
class MyVlmExtension(extension.ModelExtension):
  model_types = ('my_vlm',)
  # Allows passing `--my_vision_tokens=256` on the CLI or to `export.export()`.
  extra_kwargs_keys = ('my_vision_tokens',)

  def update_export_config(self, export_config, model_config):
    has_vision = getattr(model_config, 'vision_config', None) is not None
    # Automatically enable externalized embedder when exporting vision encoder.
    if has_vision and export_config.export_vision_encoder:
      export_config = dataclasses.replace(
          export_config,
          externalize_embedder=True,
          single_token_embedder=True,
      )
    return export_config
```

### 2. Declarative Config, `AutoModel` & Causal SDPA Settings

For architectures with non-standard Hugging Face config layouts or model
classes, you can configure them declaratively on the `ModelExtension` subclass:

* **`text_config_attr: str | None`** — Name of the attribute on a composite
  `PretrainedConfig` that holds the text backbone config when it is not named
  `text_config` (for example, `text_config_attr = 'llm_config'`). You can also
  override `get_text_config(self, model_config)` for custom logic.
* **`auto_model_cls: type | str | None`** — Overrides the Hugging Face
  `AutoModel*` class used by `load_model` for LLM/VLM tasks. Can be a
  `transformers` attribute name (e.g. `'AutoModelForImageTextToText'`) or a
  direct `PreTrainedModel` / `torch.nn.Module` class. For task-dependent logic,
  override `resolve_auto_model_class(self, model_config, export_config, task)`.
* **`is_causal_text_model: bool | None`** — Controls whether text-generation
  prefill/decode exportables set `sdpa_is_causal = True` when GPU SDPA
  composites (`use_sdpa_composite` / `apply_gpu_composites`) are enabled:
  * `True`: Always treat this architecture as a causal decoder.
  * `False`: Disable `sdpa_is_causal` (e.g. for bidirectional or custom mask
    architectures).
  * `None` *(default)*: Defer to `core/exportable_module.py`'s default check.

```python
@extension.register_model_extension()
class CustomMultimodalExtension(extension.ModelExtension):
  model_types = ('custom_vlm',)
  text_config_attr = 'language_config'
  auto_model_cls = 'AutoModelForImageTextToText'
  is_causal_text_model = True
```

### 3. Runtime Model Patching (`patch_context` vs. `patch_model`)

`ModelExtension` provides two context-manager hooks so you never need to modify
modeling files inside a checkpoint directory:

* **`patch_context(self)`** — Entered **before** `AutoModel.from_pretrained` /
  `from_config`. Use this to monkey-patch classes or functions in `transformers`
  that run during model construction (`__init__`).
* **`patch_model(self, model, export_config)`** — Entered **after** the model is
  instantiated and **around** tracing/conversion (`export_embedder_model` and
  `export_text_prefill_decode_model`). Use this to replace submodule methods,
  emit LiteRT HLFB composite ops, or adapt rotary embeddings on `model`
  (including `trust_remote_code` classes loaded dynamically from the
  checkpoint).

**Built-in examples:** `model_ext/qwen3_5/patch.py`,
`model_ext/gemma3/patch.py`, `model_ext/lfm2/patch.py`.

#### Example: Patching `RMSNorm` Composite & Dual RoPE for `split_cache`

Many `trust_remote_code` checkpoints define their own `RMSNorm` class (without
`@use_kernel_forward_from_hub("RMSNorm")`) and separate 2-argument
`rotary_emb_full` / `rotary_emb_sliding` modules, whereas LiteRT's `split_cache`
(`externalize_rope`) pipeline expects `model.model.rotary_emb(x, pos, type)` to
accept 3 arguments (`'full_attention'` and `'sliding_attention'`). Both can be
adapted in `patch_model` without editing the checkpoint:

```python
def _rms_norm_with_hlfb_forward(self, hidden_states: torch.Tensor):
  dtype = hidden_states.dtype
  hidden_states = hidden_states.to(torch.float32)
  hidden_states = normalization.rms_norm_with_hlfb(
      hidden_states,
      self.weight,
      self.variance_epsilon,
      torch.ones(
          (self.weight.shape[0],),
          dtype=torch.float32,
          device=hidden_states.device,
      ),
  )
  return hidden_states.to(dtype)


class _DualRotaryWrapper(torch.nn.Module):

  def __init__(self, full_rope, sliding_rope=None):
    super().__init__()
    self.rotary_emb_full = full_rope
    self.rotary_emb_sliding = sliding_rope

  def forward(self, x, position_ids, attention_type=None):
    if (
        attention_type == 'sliding_attention'
        and self.rotary_emb_sliding is not None
    ):
      return self.rotary_emb_sliding(x, position_ids)
    return self.rotary_emb_full(x, position_ids)


@extension.register_model_extension()
class Kanana2Extension(extension.ModelExtension):
  model_types = ('kanana2_tiny',)
  is_causal_text_model = True

  @contextlib.contextmanager
  def patch_model(self, model, export_config):
    del export_config
    # 1. Patch custom RMSNorm to emit the `odml.rms_norm` HLFB composite.
    patched_cls = {}
    for module in model.modules():
      cls = type(module)
      if cls.__name__ == 'Kanana2TinyRMSNorm' and cls not in patched_cls:
        patched_cls[cls] = cls.forward
        cls.forward = _rms_norm_with_hlfb_forward

    # 2. Wrap `rotary_emb_full` and `rotary_emb_sliding` into a 3-arg wrapper
    # for `inject_rotary_position_embedding` and `RoPEEmbedder`.
    inner = getattr(model, 'model', model)
    orig_full = getattr(inner, 'rotary_emb_full', None)
    orig_sliding = getattr(inner, 'rotary_emb_sliding', None)
    if isinstance(orig_full, torch.nn.Module):
      delattr(inner, 'rotary_emb_full')
      if hasattr(inner, 'rotary_emb_sliding'):
        delattr(inner, 'rotary_emb_sliding')
      inner.rotary_emb = _DualRotaryWrapper(orig_full, orig_sliding)
      inner.__dict__['rotary_emb_full'] = (
          lambda x, pos: inner.rotary_emb(x, pos, 'full_attention')
      )
      inner.__dict__['rotary_emb_sliding'] = (
          (lambda x, pos: inner.rotary_emb(x, pos, 'sliding_attention'))
          if orig_sliding is not None
          else None
      )
    try:
      yield
    finally:
      for cls, orig_fn in patched_cls.items():
        cls.forward = orig_fn
```

### 4. Converter V2 Weights (`checkpoint_key_prefixes`)

When exporting with Converter V2 and meta-device weight streaming (`--use_v2`
and `--delete_in_memory_params`), `HFCheckpointWeightsLoader`
(`core/weights_loader.py`) maps PyTorch parameter names on the traced module to
tensor names inside the checkpoint's `.safetensors` files.

If your model uses non-standard checkpoint wrapper prefixes or custom fused
projections, use these two hooks:

* **`checkpoint_key_prefixes: Sequence[str]`** — Additional leading module/key
  prefixes stripped and prepended when `HFCheckpointWeightsLoader` generates
  candidate safetensors keys (on top of the built-in `'model.'`,
  `'model.language_model.'`, and `'language_model.'` prefixes).
* **`resolve_checkpoint_tensor(self, param_name, try_load) -> Tensor | None`** —
  Called by `HFCheckpointWeightsLoader` when a parameter is not directly present
  in the checkpoint (or needs custom fusion, splitting, or transposition).
  `try_load(candidate_name)` looks up any candidate parameter name through the
  loader's prefix-matching table and returns a `torch.Tensor` or `None`.

```python
@extension.register_model_extension()
class CustomWeightsExtension(extension.ModelExtension):
  model_types = ('my_custom_llm',)
  # Strips/adds 'backbone.' or 'decoder.' when matching safetensors keys.
  checkpoint_key_prefixes = ('backbone.', 'decoder.')

  def resolve_checkpoint_tensor(self, param_name, try_load):
    # Reconstruct a fused `.w13.weight` parameter from separate `.w1.weight`
    # and `.w3.weight` tensors stored in the checkpoint:
    if param_name.endswith('.w13.weight'):
      prefix = param_name[: -len('.w13.weight')]
      w1 = try_load(f'{prefix}.w1.weight')
      w3 = try_load(f'{prefix}.w3.weight')
      if w1 is not None and w3 is not None:
        return torch.cat([w1, w3], dim=0)
    return None
```

### 5. Custom Exportables & `.litertlm` Packaging Metadata

To export custom subgraphs (vision/audio encoders, per-layer embedders, or
specialized prefill/decode signatures) and package them into `.litertlm`:

* **`get_prefill_decode_exportables(self, model_config, export_config)`** —
  Returns `(prefill_cls, decode_cls)` or `None` to use the standard decoder-only
  exportables.
* **`get_vision_exportables(self, model_config, export_config=None)`** — Returns
  `VisionExportablesSpec(encoder_cls, adapter_cls=..., eoi_cls=...)` for
  `export_vision_encoder_models`.
* **`get_audio_exportables(self, model_config, export_config=None)`** — Returns
  `AudioExportablesSpec(encoder_cls, adapter_cls=..., eoa_cls=...)` for
  `export_audio_encoder_models_for_llm`.
* **`get_additional_exportables(self, model_config, export_config=None)`** —
  Returns `{submodel_name: exportable_cls}` (e.g. `{'per_layer_embedder': ...}`)
  exported as `<submodel_name>.tflite` into
  `exported_model_artifacts.additional_model_paths`.
* **`get_litert_lm_section_types(self)`** — Maps keys from
  `additional_model_paths` to `.litertlm` section types (such as
  `litert_lm_builder.TFLiteModelType.TF_LITE_PER_LAYER_EMBEDDER`).
* **`get_llm_model_type_proto(self, model_config, export_config)`** — Returns a
  custom `llm_model_type_pb2.LlmModelType` protobuf message (or `None` to fall
  back to `GenericModel`).
* **`build_llm_metadata(self, ...)`** — Populates stop tokens, prompt templates,
  or modality tokens on `LlmMetadata` given `(source_model_artifacts,
  export_config, exported_model_artifacts, llm_metadata)`.
* **`update_executor_metadata(self, ...)`** — Returns an `ExecutorMetadata`
  protobuf given `(source_model_artifacts, export_config,
  exported_model_artifacts)` when packaging `.litertlm`.

**Built-in examples:** `model_ext/gemma3/extension.py`,
`model_ext/gemma3n/extension.py`, `model_ext/gemma4/extension.py`,
`model_ext/bonsai_flux2/extension.py`.

### 6. Non-Text Tasks (ASR, TTS, ImageGen, Any2Any) & `match_checkpoint`

Some checkpoints (such as Diffusers pipelines with `model_index.json` or
standalone TTS checkpoints like Kokoro) do not have a standard Hugging Face
`model_type` field in `config.json`.

* **`match_checkpoint(cls, model_path, task=None, config_dict=None) -> bool`** —
  Classmethod inspected by `infer_model_type_from_extensions` when `config.json`
  has no `model_type`. When it returns `True`, `cls.model_types[0]` is used as
  the resolved `model_type`.
* **Task model/exportable hooks:**
  * **ASR (`automatic_speech_recognition`)**: `get_speech_model_cls(model_type)`
    and `get_speech_exportables(model_config, export_config)` (see
    `model_ext/whisper/extension.py`, `model_ext/parakeet/extension.py`).
  * **TTS (`text_to_speech`)**: `get_tts_model_cls(model_type)` and
    `is_matching_tts_checkpoint(config_dict, model_path)` (see
    `model_ext/kokoro_tts/extension.py`, `model_ext/qwen3_tts/extension.py`).
  * **Text-to-Image (`text_to_image`)**: `get_image_gen_model_cls(model_type)`
    (see `model_ext/bonsai_flux2/extension.py`,
    `model_ext/flux2_klein/extension.py`).
  * **Any2Any (`any_to_any`)**: `get_any2any_model_cls(model_type)` (see
    `model_ext/minicpm_o/extension.py`).

```python
@extension.register_model_extension()
class MyDiffusionExtension(extension.ModelExtension):
  model_types = ('my_diffusion_pipeline',)

  @classmethod
  def match_checkpoint(cls, model_path, task=None, config_dict=None) -> bool:
    del model_path
    return (
        task == 'text_to_image'
        and isinstance(config_dict, dict)
        and config_dict.get('_class_name') == 'MyDiffusionPipeline'
    )

  def get_image_gen_model_cls(self, model_type: str):
    del model_type
    return MyDiffusionImageGenModel
```

---

## Using, Composing & Distributing Extensions

### 1. Pass an Extension Directly in Python (`model_extension=...`)

Prototype or override behavior inline without global registration:

```python
from litert_torch.generative.export_hf import export

export.export(
    model='/path/to/checkpoint',
    output_dir='/path/to/output',
    model_extension=MyArchExtension(),
)
```

### 2. Layer on Top of a Built-in Extension (`CompositeModelExtension`)

If `model_config.model_type` matches a registered extension (e.g. `'gemma3'`)
and you also pass `model_extension=MyOverrideExtension()`,
`resolve_model_extension` composes them as
`CompositeModelExtension((builtin_ext, user_ext))`:

* Context managers (`patch_context`, `patch_model`) and transforms
  (`update_export_config`, `build_llm_metadata`) run sequentially in order.
* Selector hooks (`get_*`, `resolve_checkpoint_tensor`) use last-non-`None`-wins
  priority, so `user_ext` overrides only the hooks it implements while
  inheriting everything else.
* Sequence attributes (`extra_kwargs_keys`, `checkpoint_key_prefixes`) and
  dictionary hooks (`get_additional_exportables`, `get_litert_lm_section_types`)
  are merged automatically.

### 3. Load via `--extension_modules` (CLI or Python)

Pass comma-separated dotted module names **or `.py` file paths** to
`--extension_modules`. Any `ModelExtension` subclass in the loaded module/file
with non-empty `model_types` is registered automatically before CLI flag
validation runs:

```bash
python -m litert_torch.generative.export_hf.export_main \
  --model=/path/to/checkpoint \
  --output_dir=/path/to/output \
  --extension_modules=/path/to/my_arch_extension.py
```

### 4. Wheel Entry-Point Discovery (`pyproject.toml`)

Third-party Python packages can register extensions automatically on import via
standard Python entry points without needing `--extension_modules`:

```toml
[project.entry-points."litert_torch.model_extensions"]
my_arch = "my_package.my_arch_extension:MyArchExtension"
```

---

## Testing a `ModelExtension`

Use `ModelExtensionTestMixin` from `model_ext/testing.py` to test the contract
of any `ModelExtension` subclass (`model_types`, `update_export_config`,
`patch_context`, `patch_model`, and `LlmModelType` / `LlmMetadata` hooks):

```python
from absl.testing import absltest
from litert_torch.generative.export_hf.model_ext import testing as ext_testing


class MyArchExtensionTest(
    ext_testing.ModelExtensionTestMixin, absltest.TestCase
):
  extension_cls = MyArchExtension


if __name__ == '__main__':
  absltest.main()
```
