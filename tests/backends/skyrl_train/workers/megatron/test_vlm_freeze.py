"""CPU tests for the Megatron VLM freeze helpers (``trainable_params``).

The model mocks mimic the layout of Megatron-Bridge's ``Qwen3VLModel``
(``language_model``, ``vision_model`` with a ``merger`` and
``decoder.deepstack_merger_list``) and its ``freeze()``, and the PEFT mocks
mimic Bridge's ``PEFT.__call__``: freeze every parameter, then add adapters,
either beside the linears or (like Bridge's ``LoRA.transform``) by wrapping a
target linear so its weight moves under ``to_wrap``. No Megatron import is needed.

uv run --isolated --extra skyrl-train --extra dev pytest tests/backends/skyrl_train/workers/megatron/test_vlm_freeze.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from skyrl.backends.skyrl_train.distributed.megatron.trainable_params import (
    VLM_FREEZE_FIELDS,
    apply_vlm_freeze_to_provider,
    lora_checkpoint_state_dict,
    make_keep_trainable_hooks,
    register_lora_pre_wrap_hooks,
    summarize_trainable_parameters,
)


class _VisionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        self.deepstack_merger_list = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])


class _VisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _VisionDecoder()
        self.merger = nn.Linear(4, 4)


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        self.output_layer = nn.Linear(4, 8)


class _Qwen3VLLike(nn.Module):
    """``freeze()`` follows Megatron-Bridge's Qwen3VLModel.freeze."""

    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()
        self.vision_model = _VisionModel()

    def projection_modules(self):
        return [self.vision_model.merger, self.vision_model.decoder.deepstack_merger_list]

    def freeze(self, freeze_language_model, freeze_vision_model, freeze_vision_projection):
        modules = []
        if freeze_language_model:
            modules.append(self.language_model)
        if freeze_vision_model:
            modules.append(self.vision_model)
        if freeze_vision_projection:
            modules.extend(self.projection_modules())
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        if freeze_vision_model and not freeze_vision_projection:
            for module in self.projection_modules():
                for param in module.parameters():
                    param.requires_grad = True


class _Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_in = nn.Linear(4, 2, bias=False)
        self.linear_out = nn.Linear(2, 4, bias=False)


def _fake_peft(model):
    """Bridge PEFT shape: freeze every parameter, then add trainable adapters to the LM linears."""
    for chunk in model:
        for param in chunk.parameters():
            param.requires_grad = False
        for layer in chunk.language_model.decoder:
            layer.adapter = _Adapter()
    return model


class _LoRAWrapped(nn.Module):
    """Bridge's LoRA wrapper shape: the original linear moves under ``to_wrap`` (same tensors, new names)."""

    def __init__(self, to_wrap):
        super().__init__()
        self.to_wrap = to_wrap
        self.adapter = _Adapter()


def _wrap_linears(module, wrap, prefix=""):
    for name, child in list(module.named_children()):
        path = f"{prefix}{name}"
        if isinstance(child, nn.Linear) and wrap(path):
            setattr(module, name, _LoRAWrapped(child))
        else:
            _wrap_linears(child, wrap, path + ".")


def _fake_wrapping_peft(wrap):
    """Bridge PEFT with real wrapping: freeze every parameter, then wrap the linears whose path ``wrap`` accepts."""

    def peft(model):
        for chunk in model:
            for param in chunk.parameters():
                param.requires_grad = False
            _wrap_linears(chunk, wrap)
        return model

    return peft


def _all_linear(path):
    return True


def _language_model_linears(path):
    return path.startswith("language_model.")


def _build(freeze_language_model, freeze_vision_model, freeze_vision_projection, n_chunks=1):
    chunks = [_Qwen3VLLike() for _ in range(n_chunks)]
    for chunk in chunks:
        chunk.freeze(freeze_language_model, freeze_vision_model, freeze_vision_projection)
    return chunks


def _trainable(chunk):
    return {name for name, p in chunk.named_parameters() if p.requires_grad}


def _projection_names(chunk):
    return {
        n for n, _ in chunk.named_parameters() if n.startswith(("vision_model.merger", "vision_model.decoder.deep"))
    }


# ---------------------------------------------------------------------------
# apply_vlm_freeze_to_provider
# ---------------------------------------------------------------------------


def _megatron_cfg(**flags):
    return SimpleNamespace(**{name: flags.get(name, False) for name in VLM_FREEZE_FIELDS})


def test_apply_is_noop_when_no_flag_is_set():
    provider = SimpleNamespace()  # no freeze fields at all: a text-only provider
    assert apply_vlm_freeze_to_provider(provider, _megatron_cfg()) is False
    assert not any(hasattr(provider, name) for name in VLM_FREEZE_FIELDS)


def test_apply_sets_every_field_on_a_vlm_provider():
    provider = SimpleNamespace(**{name: True for name in VLM_FREEZE_FIELDS})
    cfg = _megatron_cfg(freeze_language_model=True, freeze_vision_model=True)
    assert apply_vlm_freeze_to_provider(provider, cfg) is True
    assert provider.freeze_language_model is True
    assert provider.freeze_vision_model is True
    # Unset flags are written as False, overriding a provider default (Qwen3-VL MoE defaults to True).
    assert provider.freeze_vision_projection is False


def test_apply_rejects_a_provider_without_freeze_support():
    provider = SimpleNamespace(freeze_language_model=False)
    with pytest.raises(ValueError, match="has no .*freeze_vision_model"):
        apply_vlm_freeze_to_provider(provider, _megatron_cfg(freeze_vision_model=True))


# ---------------------------------------------------------------------------
# make_keep_trainable_hooks: projection at full rank next to LoRA
# ---------------------------------------------------------------------------


def test_projection_only_selection_matches_the_projection():
    (chunk,) = _build(freeze_language_model=True, freeze_vision_model=True, freeze_vision_projection=False)
    assert _trainable(chunk) == _projection_names(chunk)


def test_lora_plus_full_projection():
    model = _build(freeze_language_model=True, freeze_vision_model=True, freeze_vision_projection=False)
    restored = []
    record, restore = make_keep_trainable_hooks(on_restore=restored.append)
    model = restore(_fake_peft(record(model)))
    (chunk,) = model

    adapters = {n for n, _ in chunk.named_parameters() if ".adapter." in n}
    assert adapters and _trainable(chunk) == _projection_names(chunk) | adapters
    # Everything else (LM base weights, vision encoder blocks) stays frozen.
    assert not any(p.requires_grad for n, p in chunk.named_parameters() if n.startswith("vision_model.decoder.layers"))
    assert not any(
        p.requires_grad for n, p in chunk.named_parameters() if n.startswith("language_model") and ".adapter." not in n
    )
    assert restored == [model]


def test_lora_with_all_three_frozen_keeps_adapters_only():
    model = _build(True, True, True)
    record, restore = make_keep_trainable_hooks()
    (chunk,) = restore(_fake_peft(record(model)))
    assert _trainable(chunk) == {n for n, _ in chunk.named_parameters() if ".adapter." in n}


def test_hooks_handle_multiple_model_chunks_independently():
    model = _build(True, True, False, n_chunks=2)
    model[1].vision_model.merger.weight.requires_grad = False  # chunks can differ (e.g. pipeline stages)
    record, restore = make_keep_trainable_hooks()
    model = restore(_fake_peft(record(model)))
    assert model[0].vision_model.merger.weight.requires_grad
    assert not model[1].vision_model.merger.weight.requires_grad
    assert model[1].vision_model.merger.bias.requires_grad


def test_hooks_do_not_depend_on_chunk_order():
    model = _build(True, True, False, n_chunks=2)
    model[1].vision_model.merger.weight.requires_grad = False
    record, restore = make_keep_trainable_hooks()
    record(model)
    reordered = _fake_peft([model[1], model[0]])
    restore(reordered)
    assert model[0].vision_model.merger.weight.requires_grad
    assert not model[1].vision_model.merger.weight.requires_grad


def test_restore_rejects_a_replaced_parameter():
    (chunk,) = _build(True, True, False)
    record, restore = make_keep_trainable_hooks()
    record([chunk])
    chunk.vision_model.merger = nn.Linear(4, 4)  # new tensors: the recorded ones are gone
    with pytest.raises(RuntimeError, match="LoRA replaced"):
        restore([chunk])


def test_lora_wrapping_frozen_layers_restores_the_projection():
    # LoRA on the language model only, wrapped the way Bridge does it: the projection is restored by
    # identity, the wrapped LM base weights stay frozen, the adapters train.
    model = _build(True, True, False)
    record, restore = make_keep_trainable_hooks()
    (chunk,) = restore(_fake_wrapping_peft(_language_model_linears)(record(model)))
    adapters = {n for n, _ in chunk.named_parameters() if ".adapter." in n}
    assert adapters and _trainable(chunk) == _projection_names(chunk) | adapters
    lm_base = [n for n, _ in chunk.named_parameters() if n.startswith("language_model.decoder") and n not in adapters]
    assert lm_base and all(".to_wrap." in n for n in lm_base)


def test_all_linear_lora_over_a_trainable_projection_raises():
    # target_modules=all-linear also matches Qwen-VL's merger layers; silently leaving the projection
    # adapter-only would contradict freeze_vision_projection=false.
    model = _build(True, True, False)
    record, restore = make_keep_trainable_hooks()
    # 8 = merger weight + bias and the three DeepStack mergers' weights + biases.
    with pytest.raises(ValueError, match=r"target_modules wrap 8 parameter\(s\).*deepstack_merger_list\.0\.bias"):
        restore(_fake_wrapping_peft(_all_linear)(record(model)))


def test_lora_over_an_unfrozen_language_model_raises():
    model = _build(False, True, True)  # LM left at full rank, LoRA also targets it
    record, restore = make_keep_trainable_hooks()
    with pytest.raises(ValueError, match="restrict model.lora.target_modules"):
        restore(_fake_wrapping_peft(_language_model_linears)(record(model)))


def test_hooks_accept_a_single_module():
    (chunk,) = _build(True, True, False)
    record, restore = make_keep_trainable_hooks()
    record(chunk)
    for p in chunk.parameters():
        p.requires_grad = False
    assert restore(chunk) is chunk
    assert _trainable(chunk) == _projection_names(chunk)


# ---------------------------------------------------------------------------
# summarize_trainable_parameters
# ---------------------------------------------------------------------------


def test_summary_counts_and_states():
    (chunk,) = _build(True, True, False)
    wrapped = nn.Module()
    wrapped.module = nn.Module()
    wrapped.module.module = chunk  # DDP(Float16Module(model)) naming
    lines = summarize_trainable_parameters([wrapped])

    total = sum(p.numel() for p in chunk.parameters())
    projection = sum(p.numel() for n, p in chunk.named_parameters() if n in _projection_names(chunk))
    assert lines[0].startswith(f"trainable parameters: {projection} of {total} (")
    by_prefix = {line.strip().split(":")[0]: line for line in lines[1:]}
    assert "frozen" in by_prefix["language_model.decoder"]
    assert by_prefix["vision_model.merger"].endswith("trainable")
    assert "partly trainable" in by_prefix["vision_model.decoder"]  # blocks frozen, DeepStack mergers trainable


# ---------------------------------------------------------------------------
# lora_checkpoint_state_dict
# ---------------------------------------------------------------------------


def test_lora_checkpoint_holds_adapters_and_trainable_base_only():
    model = _build(True, True, False)
    record, restore = make_keep_trainable_hooks()
    (chunk,) = restore(_fake_peft(record(model)))
    state = lora_checkpoint_state_dict(chunk.named_parameters())
    assert set(state) == _trainable(chunk)
    assert all(torch.equal(state[n], p.data) for n, p in chunk.named_parameters() if n in state)

    # Plain LoRA: adapters only, as before; frozen adapter tensors are still saved.
    (plain,) = _fake_peft(_build(False, False, False))
    plain.language_model.decoder[0].adapter.linear_in.weight.requires_grad = False
    state = lora_checkpoint_state_dict(plain.named_parameters())
    assert set(state) == {n for n, _ in plain.named_parameters() if ".adapter." in n}


# ---------------------------------------------------------------------------
# register_lora_pre_wrap_hooks
# ---------------------------------------------------------------------------


class _Provider(SimpleNamespace):
    def register_pre_wrap_hook(self, hook):
        self.hooks.append(hook)


def _lora_hook(model):
    return model


def test_plain_lora_ignores_provider_freeze_defaults():
    # Qwen3-VL MoE's provider defaults freeze_language_model=True; with no user flag set, plain LoRA
    # must register only the LoRA hook (adapters only), not the keep-trainable pair.
    provider = _Provider(hooks=[], freeze_language_model=True, freeze_vision_model=False)
    assert register_lora_pre_wrap_hooks(provider, _megatron_cfg(), _lora_hook) is False
    assert provider.hooks == [_lora_hook]


def test_user_freeze_flags_bracket_the_lora_hook():
    provider = _Provider(hooks=[])
    cfg = _megatron_cfg(freeze_language_model=True, freeze_vision_model=True)
    assert register_lora_pre_wrap_hooks(provider, cfg, _lora_hook) is True
    record, lora, restore = provider.hooks
    assert lora is _lora_hook

    model = _build(True, True, False)
    (chunk,) = restore(_fake_peft(lora(record(model))))
    assert _projection_names(chunk) <= _trainable(chunk)
