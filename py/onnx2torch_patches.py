"""Compatibility patches for onnx2torch 1.5.15 against the Supertonic ONNX models.

The Supertonic ONNX models (`text_encoder`, `duration_predictor`,
`vector_estimator`, `vocoder`) are all exported at **opset 19**. onnx2torch
1.5.15 ships converters for many ops only up through older opset versions
(e.g. its `Constant` converters top out at v13 -- see `_OPSET_LAG_OPS`
below). A bare `onnx2torch.convert()` call on any of the four models fails
immediately with `KeyError: OperationDescription(..., version=19)`, because
the registry is keyed by exact `(domain, op_type, version)` and has no entry
for the op at the model's resolved opset.

This module carries two independent patches, both discovered and verified
against the real `.onnx` assets in `assets/onnx/`. Both are generic
version-compatibility shims -- neither depends on this project's specific
graph shapes or weights -- but they were tuned to fix concretely-observed
failures on these four models, not written from the onnx2torch changelog in
the abstract.

Patch 1 -- opset-alias the registry (`_alias_opset_converters`)
-----------------------------------------------------------------
For every op type actually used in a given model, resolve its *since_version*
at the model's declared opset (19) via `onnx.defs.get_schema`. If onnx2torch
has no converter registered for that exact version, alias it to the highest
*registered* version below it for the same op type. Concretely, for the
Supertonic models this aliases `Cast, Constant, Equal, Pad, Reshape, Shape,
Split` down to whatever version onnx2torch actually implements (e.g.
`Constant` v19 -> v13, the newest version onnx2torch registers).

This is safe only insofar as the converter implementation does not behave
differently across the skipped versions for the attributes these models
actually use (e.g. Reshape's `allowzero` attribute added after v13 defaults
to 0, matching the pre-existing behavior; the `value_string`-style Constant
variants added after v13 aren't used here). It was not safe in the abstract
for every op in the opset-19 spec -- it is safe for the op/attribute
combinations these four graphs exercise, which is what the numeric
self-test below checks.

Patch 2 -- a Clip converter that tolerates an omitted bound (`_dynamic_clip_converter`)
-----------------------------------------------------------------------------------------
`text_encoder` has 8 `Clip` nodes (in its attention layers' relative-position
bias) and `duration_predictor` has 4, in every case with a `min` input and no
`max` input. Per the ONNX spec, an omitted optional trailing input is encoded
as an *empty string* in the node's `input` list rather than being dropped
from it -- so these `Clip` nodes carry `input = [data, min, ""]`.

Investigating the failure directly (not merely from onnx2torch's exception
message) showed the actual defect: onnx2torch's stock Clip converter
(`onnx2torch/node_converters/clip.py`) takes `max_name = node.input_values[2]`
whenever the input list has 3 entries, without checking for the empty-string
sentinel, then calls `get_const_value("", graph)`. That raises `KeyError`
inside `get_const_value` (there is no tensor named `""`), which the stock
converter re-raises as `NotImplementedError('Dynamic value of min/max is not
implemented')` -- a message that (misleadingly, in this codebase's case)
suggests a genuinely graph-computed bound. In every Clip node in every
Supertonic model, `min` is in fact produced by a plain `Constant` node (a
true compile-time constant onnx2torch's own `get_const_value` already knows
how to fold) and `max` is simply absent. So the failure here is stock
onnx2torch mishandling an *omitted* optional input, not a *dynamic* one.

The patch below treats the empty string as "no bound" (matching the ONNX
spec) instead of a tensor name to resolve, which alone fixes every Clip
node actually present in these four models. It additionally keeps a live
graph edge and does `torch.clamp(x, min=tensor)` for the case where a bound
*is* wired to a non-constant producer (e.g. computed by a `Sub`), since that
is a real possibility for a Clip node in general and costs nothing to
support -- but as of this writing no Clip node in any of the four Supertonic
ONNX graphs actually exercises that branch; it is defensive generality, not
something the self-test below exercises. Read `_dynamic_clip_converter` as
"correct optional-input handling with a dynamic-bound fallback," not as
"the fix for a sequence-length-dependent bound" -- an earlier description of
this patch (before the graphs were inspected node-by-node) assumed the
`min` bound was computed from a `Sub` and depended on sequence length; that
turned out not to match any Clip node in these models. The docstring above
is corrected to what was actually verified.

Both patches are generic version-compatibility/spec-conformance shims, not
model-specific hacks: they change how onnx2torch resolves op versions and
optional inputs, not anything about Supertonic's graph structure or
weights. Faithfulness was verified numerically -- see the `__main__`
self-test below, which converts `text_encoder`, runs a forward + backward
pass, and diffs the converted-torch output against onnxruntime on the same
input (max abs diff must be < 1e-4).

Usage
-----
    from onnx2torch_patches import apply_patches
    import onnx

    model = onnx.load("assets/onnx/text_encoder.onnx")
    summary = apply_patches(model)   # patches the global onnx2torch registry
    print(summary)
    from onnx2torch import convert
    torch_model = convert(model)

`apply_patches` is idempotent: calling it again (on the same or a different
model) only adds aliases/registrations that are not already present, and
never overwrites a previously-applied alias.
"""

from dataclasses import dataclass, field

import onnx
from onnx import defs
import torch
from torch import nn

from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription
from onnx2torch.node_converters.clip import _create_torch_module as _clip_create_module
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import (
    OnnxMapping,
    OnnxToTorchModule,
    OperationConverterResult,
    get_const_value,
)

__all__ = ["apply_patches", "PatchSummary"]


@dataclass
class PatchSummary:
    """What `apply_patches` did, for logging/assertions in callers."""

    aliased: list = field(default_factory=list)  # [(op_type, from_version, to_version), ...]
    already_aliased: list = field(default_factory=list)  # same shape, skipped as already-present
    unresolvable: list = field(default_factory=list)  # [(op_type, since_version), ...]
    clip_patched: bool = False
    clip_already_patched: bool = False

    def __str__(self) -> str:
        lines = []
        if self.aliased:
            lines.append(
                "opset aliases added: "
                + ", ".join(f"{op}(v{f}->v{t})" for op, f, t in self.aliased)
            )
        if self.already_aliased:
            lines.append(f"opset aliases already present: {len(self.already_aliased)}")
        if self.unresolvable:
            lines.append(
                "UNRESOLVABLE ops (no lower converter registered): "
                + ", ".join(f"{op}(v{sv})" for op, sv in self.unresolvable)
            )
        if self.clip_patched:
            lines.append("Clip converter patched for versions 11, 12, 13")
        elif self.clip_already_patched:
            lines.append("Clip converter already patched")
        return "; ".join(lines) if lines else "no changes (already fully patched)"


# ---------------------------------------------------------------------------
# Patch 1: opset-alias the registry for whatever ops a given model uses.
# ---------------------------------------------------------------------------

_CLIP_VERSIONS_TO_PATCH = (11, 12, 13)


def _alias_opset_converters(onnx_model: "onnx.ModelProto", summary: PatchSummary) -> None:
    op_types = sorted({n.op_type for n in onnx_model.graph.node})
    opset_map = {o.domain: o.version for o in onnx_model.opset_import}
    model_opset = opset_map.get("", 1)

    for op_type in op_types:
        try:
            since_version = defs.get_schema(
                op_type, domain="", max_inclusive_version=model_opset
            ).since_version
        except Exception:
            continue
        key = OperationDescription(domain="", operation_type=op_type, version=since_version)
        if key in _CONVERTER_REGISTRY:
            continue

        candidates = [
            k
            for k in _CONVERTER_REGISTRY
            if k.domain == "" and k.operation_type == op_type and k.version <= since_version
        ]
        if not candidates:
            entry = (op_type, since_version)
            if entry not in summary.unresolvable:
                summary.unresolvable.append(entry)
            continue

        best = max(candidates, key=lambda k: k.version)
        _CONVERTER_REGISTRY[key] = _CONVERTER_REGISTRY[best]
        summary.aliased.append((op_type, best.version, since_version))


# ---------------------------------------------------------------------------
# Patch 2: a Clip converter that treats an omitted optional input (encoded
# as "" per the ONNX spec) as "no bound" instead of a tensor name to
# resolve, and falls back to a live graph edge for a genuinely
# graph-computed (non-constant) bound rather than raising.
# ---------------------------------------------------------------------------


class OnnxDynamicClip(nn.Module, OnnxToTorchModule):
    """Clip with a bound wired to a live (non-constant) graph tensor.

    Not exercised by any Clip node in the current Supertonic ONNX models --
    every bound present in those graphs resolves to a constant, so
    `_dynamic_clip_converter` never actually builds this module for them.
    Kept because a genuinely dynamic bound is a real possibility for a Clip
    node under the ONNX spec, and supporting it costs nothing.
    """

    def __init__(self, has_min: bool, has_max: bool, const_min=None, const_max=None):
        super().__init__()
        self.has_min = has_min
        self.has_max = has_max
        self.const_min = const_min
        self.const_max = const_max

    def forward(self, x, *dyn):
        dyn = list(dyn)
        min_v = dyn.pop(0) if self.has_min else self.const_min
        max_v = dyn.pop(0) if self.has_max else self.const_max
        return torch.clamp(x, min=min_v, max=max_v)


def _dynamic_clip_converter(node: "OnnxNode", graph: "OnnxGraph") -> "OperationConverterResult":
    # An omitted optional trailing input is an empty string per the ONNX
    # spec, not a real tensor name -- treat it as absent. This is the line
    # stock onnx2torch's Clip converter is missing.
    min_name = node.input_values[1] if len(node.input_values) > 1 and node.input_values[1] else None
    max_name = node.input_values[2] if len(node.input_values) > 2 and node.input_values[2] else None

    def resolve(name):
        if name is None:
            return None, None
        try:
            return "const", float(get_const_value(name, graph))
        except Exception:
            return "dyn", name

    min_kind, min_v = resolve(min_name)
    max_kind, max_v = resolve(max_name)

    if min_kind != "dyn" and max_kind != "dyn":
        # Every bound present resolves statically -- reuse upstream's own
        # module (gets the ReLU/ReLU6 fast paths) for correctness and perf.
        torch_module = _clip_create_module(min_val=min_v, max_val=max_v)
        return OperationConverterResult(
            torch_module=torch_module,
            onnx_mapping=OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values),
        )

    has_min = min_kind is not None
    has_max = max_kind is not None
    dyn_inputs = [node.input_values[0]]
    const_min = None
    const_max = None
    if has_min:
        if min_kind == "dyn":
            dyn_inputs.append(min_name)
        else:
            const_min = min_v
    if has_max:
        if max_kind == "dyn":
            dyn_inputs.append(max_name)
        else:
            const_max = max_v

    torch_module = OnnxDynamicClip(
        has_min=(has_min and min_kind == "dyn"),
        has_max=(has_max and max_kind == "dyn"),
        const_min=const_min,
        const_max=const_max,
    )
    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(inputs=tuple(dyn_inputs), outputs=node.output_values),
    )


def _patch_clip_converter(summary: PatchSummary) -> None:
    already = True
    for v in _CLIP_VERSIONS_TO_PATCH:
        key = OperationDescription(domain="", operation_type="Clip", version=v)
        if _CONVERTER_REGISTRY.get(key) is not _dynamic_clip_converter:
            already = False
        _CONVERTER_REGISTRY[key] = _dynamic_clip_converter
    summary.clip_patched = not already
    summary.clip_already_patched = already


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_patches(onnx_model: "onnx.ModelProto | None" = None) -> PatchSummary:
    """Apply both compatibility patches to onnx2torch's global registry.

    Idempotent: safe to call more than once, on the same model, a different
    model, or with `onnx_model=None`. Already-applied aliases and the Clip
    patch are detected and reported as such rather than reapplied/duplicated
    (dict assignment for the Clip patch is a no-op the second time; the
    opset-alias loop already skips any key already present in the registry).

    Pass the loaded `onnx.ModelProto` you are about to `convert()` so its
    actual op set can be aliased; the Clip patch is model-independent and
    is always (re-)applied. Call once per distinct model before converting
    it, since different Supertonic models can use different op sets.
    """
    summary = PatchSummary()
    _patch_clip_converter(summary)
    if onnx_model is not None:
        _alias_opset_converters(onnx_model, summary)
    return summary


if __name__ == "__main__":
    import os
    import sys

    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from helper import UnicodeProcessor

    ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "onnx")
    TEXT_ENCODER_PATH = os.path.join(ASSETS_DIR, "text_encoder.onnx")

    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
        checks.append(ok)

    print("=== onnx2torch_patches self-test (text_encoder only) ===")

    model = onnx.load(TEXT_ENCODER_PATH)
    summary = apply_patches(model)
    print(f"patch summary: {summary}")
    check("no unresolvable ops", len(summary.unresolvable) == 0, str(summary.unresolvable))

    from onnx2torch import convert

    try:
        torch_model = convert(model)
        torch_model.eval()
        check("onnx2torch.convert() succeeds on text_encoder", True)
    except Exception as e:  # noqa: BLE001
        check("onnx2torch.convert() succeeds on text_encoder", False, f"{type(e).__name__}: {e}")
        sys.exit(1)

    text_processor = UnicodeProcessor(os.path.join(ASSETS_DIR, "unicode_indexer.json"))
    text_ids, text_mask = text_processor(["The quick brown fox jumps."], ["en"])
    rng = np.random.default_rng(0)
    style_ttl = (rng.standard_normal((1, 50, 256)).astype(np.float32) * 0.1)

    style_ttl_t = torch.tensor(style_ttl, requires_grad=True)
    text_ids_t = torch.tensor(text_ids)
    text_mask_t = torch.tensor(text_mask)

    out = torch_model(text_ids_t, style_ttl_t, text_mask_t)
    out_t = out[0] if isinstance(out, (tuple, list)) else (list(out.values())[0] if isinstance(out, dict) else out)
    print(f"output shape: {tuple(out_t.shape)}")

    scalar = out_t.sum()
    scalar.backward()
    grad = style_ttl_t.grad

    grad_ok = grad is not None and torch.isfinite(grad).all().item() and (grad != 0).any().item()
    nonzero = int((grad != 0).sum().item()) if grad is not None else 0
    grad_max = float(grad.abs().max().item()) if grad is not None else float("nan")
    check(
        "backward: grad finite and non-zero",
        grad_ok,
        f"nonzero={nonzero}/{grad.numel() if grad is not None else 0}, max|grad|={grad_max:.6g}",
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(TEXT_ENCODER_PATH, providers=["CPUExecutionProvider"])
    ort_out, *_ = sess.run(None, {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask})
    torch_out_np = out_t.detach().numpy()

    shape_ok = ort_out.shape == torch_out_np.shape
    check("onnxruntime/torch output shapes match", shape_ok, f"ort={ort_out.shape} torch={torch_out_np.shape}")

    if shape_ok:
        diff = np.abs(ort_out - torch_out_np)
        max_diff = float(diff.max())
        check("max abs diff < 1e-4 (onnxruntime vs converted torch)", max_diff < 1e-4, f"max_abs_diff={max_diff:.3e}")
    else:
        checks.append(False)

    print("=== " + ("ALL CHECKS PASSED" if all(checks) else "SOME CHECKS FAILED") + " ===")
    sys.exit(0 if all(checks) else 1)
