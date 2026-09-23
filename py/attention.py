"""Read the attention this fork's `vector_estimator.onnx` never exports.

The frozen graph declares a single output, `denoised_latent`, but its 24
transformer blocks compute 8 Softmax tensors along the way that ONNX simply
never wires to `graph.output`. Nothing about the model needs retraining or
gradients to read them -- they're already materialized during every forward
pass. `export_attention_graph` appends them to a *copy* of the graph so
`onnxruntime` will hand them back; the copy is cached next to the source
model since it's a quarter gig and the source almost never changes.

Two node families show up, distinguished by op name (see CLAUDE.md's
"Project Notes" for how this maps to what each style tensor actually does):

  main_blocks.{3,9,15,21}/attn        (heads, streams, L, T)  -- attends over
                                       TEXT positions (T text units)
  main_blocks.{5,11,17,23}/attention  (heads, streams, L, 50) -- attends over
                                       the 50 rows of `style_ttl`

L is the number of latent frames, each `base_chunk_size *
chunk_compress_factor` samples of audio. The leading two axes of both
families are (head, stream); nothing here assumes what "stream" means beyond
"a second axis to search over", per the brief this module was built from.

Usage (from the repo root, so `assets/onnx` resolves):
    python3 py/attention.py --text "..." --voice assets/voice_styles/M1.json
"""

import argparse
import os
import re
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper as onnx_helper
from scipy.stats import ConstantInputWarning, spearmanr

from helper import Style, TextToSpeech, load_text_to_speech, load_voice_style

# Cached instrumented copy of vector_estimator.onnx, next to the source
# model. Both `*.onnx` and `assets/` are gitignored (see .gitignore), so this
# file never enters version control regardless of where it's written.
DEFAULT_ATTENTION_PATH = "assets/onnx/vector_estimator.attn.onnx"

_TEXT_NODE_RE = re.compile(r"(main_blocks\.\d+/attn)/Softmax")
_STYLE_NODE_RE = re.compile(r"(main_blocks\.\d+/attention)/Softmax")
_OPEN_TAG_RE = re.compile(r"^<[a-zA-Z]+>")
_CLOSE_TAG_RE = re.compile(r"</[a-zA-Z]+>$")

# Reused across `analyze()` calls in one process so repeated renders (e.g. in
# the test suite) don't reload a 256 MB ONNX session every time.
_SESSION_CACHE: dict[str, ort.InferenceSession] = {}


def export_attention_graph(src_path: str, dst_path: str) -> list[str]:
    """Append every Softmax node's output to `graph.output` and save.

    Idempotent: if `dst_path` already exists and its own Softmax nodes are
    already all present in its `graph.output`, nothing is re-saved -- the
    existing file's output names are returned as-is. This matters because
    `vector_estimator.onnx` is 256 MB; re-saving it on every call would waste
    seconds of runtime and a quarter gig of disk for no benefit.

    Returns the appended output names, in graph order.
    """
    if os.path.exists(dst_path):
        dst_model = onnx.load(dst_path)
        softmax_outputs = [
            n.output[0] for n in dst_model.graph.node if n.op_type == "Softmax"
        ]
        existing = {o.name for o in dst_model.graph.output}
        if softmax_outputs and all(name in existing for name in softmax_outputs):
            return softmax_outputs
        # Fall through: dst exists but is stale/incomplete -- rebuild it.

    model = onnx.load(src_path)
    graph = model.graph
    softmax_outputs = [n.output[0] for n in graph.node if n.op_type == "Softmax"]
    existing = {o.name for o in graph.output}
    for name in softmax_outputs:
        if name not in existing:
            graph.output.append(
                onnx_helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            )

    dst_dir = os.path.dirname(dst_path)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
    onnx.save(model, dst_path)
    return softmax_outputs


def _source_path_for(attention_path: str) -> str:
    """Map an instrumented-graph path back to its source model, assuming the
    `<name>.attn.onnx` next to `<name>.onnx` convention `DEFAULT_ATTENTION_PATH`
    establishes."""
    directory = os.path.dirname(attention_path)
    base = os.path.basename(attention_path)
    if base.endswith(".attn.onnx"):
        src_base = base[: -len(".attn.onnx")] + ".onnx"
    else:
        src_base = "vector_estimator.onnx"
    return os.path.join(directory, src_base)


def _short_node_name(full_output_name: str) -> tuple[str, str]:
    """Map a raw Softmax output name to (short node name, family).

    family is "text" for the `.../attn/Softmax...` family (attends over text
    units) and "style" for `.../attention/Softmax...` (attends over
    `style_ttl` rows).
    """
    m = _TEXT_NODE_RE.search(full_output_name)
    if m:
        return m.group(1), "text"
    m = _STYLE_NODE_RE.search(full_output_name)
    if m:
        return m.group(1), "style"
    raise ValueError(f"Unrecognized attention output name: {full_output_name}")


def _get_instrumented_session(attention_path: str, src_path: str) -> ort.InferenceSession:
    session = _SESSION_CACHE.get(attention_path)
    if session is not None:
        return session
    export_attention_graph(src_path, attention_path)
    opts = ort.SessionOptions()
    session = ort.InferenceSession(
        attention_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    _SESSION_CACHE[attention_path] = session
    return session


def _invert_indexer(indexer: list[int]) -> dict[int, str]:
    """Invert the codepoint -> token-id indexer so ids can be decoded back to
    characters. First codepoint that maps to a given id wins (there can be
    more than one; any of them decodes the id consistently for our purposes).
    """
    inv: dict[int, str] = {}
    for cp, i in enumerate(indexer):
        if i >= 0 and i not in inv:
            inv[i] = chr(cp)
    return inv


def _tag_char_indices(joined: str) -> set:
    """Character indices of a leading `<lang>` / trailing `</lang>` tag."""
    idx: set = set()
    m = _OPEN_TAG_RE.match(joined)
    if m:
        idx.update(range(m.start(), m.end()))
    m = _CLOSE_TAG_RE.search(joined)
    if m:
        idx.update(range(m.start(), m.end()))
    return idx


@dataclass
class Alignment:
    """The text-attending head selected as the render's alignment."""

    matrix: np.ndarray  # (L, T) the chosen head's attention map
    tokens: list  # length T, decoded characters
    frame_seconds: float
    node: str  # e.g. "main_blocks.9/attn"
    head: int
    stream: int
    spearman: float  # monotonicity score of the chosen head
    duration: float  # predicted utterance seconds from the duration predictor

    def _unit_bounds(self) -> dict:
        """{text unit index -> (first_frame, last_frame)} for units that at
        least one frame's argmax selects. Units no frame selects are absent."""
        argmax_seq = self.matrix.argmax(axis=-1)
        bounds = {}
        for t_idx in range(self.matrix.shape[1]):
            frames = np.nonzero(argmax_seq == t_idx)[0]
            if frames.size:
                bounds[t_idx] = (int(frames.min()), int(frames.max()))
        return bounds

    def token_spans(self) -> list:
        """Per text unit: (token, start_s, end_s) from the first/last frame
        whose argmax is that unit. Units no frame selects are omitted.

        A frame covers the half-open interval
        [frame * frame_seconds, (frame + 1) * frame_seconds), so a unit
        selected only by frame N spans exactly one `frame_seconds`-wide
        interval rather than a zero-length instant.
        """
        bounds = self._unit_bounds()
        return [
            (
                self.tokens[t_idx],
                bounds[t_idx][0] * self.frame_seconds,
                (bounds[t_idx][1] + 1) * self.frame_seconds,
            )
            for t_idx in sorted(bounds)
        ]

    def word_spans(self) -> list:
        """Whitespace-grouped words, language tags excluded, each spanning
        min start to max end over its constituent (alphanumeric) units.
        Punctuation-only groups are skipped, and punctuation attached to a
        word (e.g. a trailing period) is dropped from both the label and the
        span -- only alphanumeric units contribute either. Uses the same
        frame-as-interval convention as `token_spans`."""
        bounds = self._unit_bounds()
        joined = "".join(self.tokens)
        tag_idx = _tag_char_indices(joined)

        groups: list = []
        current: list = []
        for i, ch in enumerate(self.tokens):
            if i in tag_idx:
                continue  # transparent: neither breaks nor joins a word
            if ch.isspace():
                if current:
                    groups.append(current)
                    current = []
                continue
            current.append(i)
        if current:
            groups.append(current)

        spans = []
        for group in groups:
            alnum = [i for i in group if self.tokens[i].isalnum()]
            present = [i for i in alnum if i in bounds]
            if not present:
                continue
            label = "".join(self.tokens[i] for i in alnum)
            start = min(bounds[i][0] for i in present) * self.frame_seconds
            end = (max(bounds[i][1] for i in present) + 1) * self.frame_seconds
            spans.append((label, start, end))
        return spans


@dataclass
class StyleAttention:
    """The style_ttl-attending heads, averaged into one (L, 50) map."""

    matrix: np.ndarray  # (L, 50), mean over the four style nodes and their leading axes
    frame_seconds: float

    def row_weights(self) -> np.ndarray:
        """(50,) mean attention weight per style_ttl row, over frames."""
        return self.matrix.mean(axis=0)

    def frame_divergence(self) -> np.ndarray:
        """(L,) total-variation distance of each frame's row distribution
        from the utterance mean. 0 means that frame reads style_ttl exactly
        like the utterance average does (style read globally)."""
        mean_dist = self.row_weights()
        return 0.5 * np.abs(self.matrix - mean_dist[None, :]).sum(axis=-1)

    def effective_rows(self) -> np.ndarray:
        """(L,) exp(entropy) of each frame's row distribution -- how many of
        the 50 style_ttl rows that frame is effectively spreading its
        attention over."""
        p = self.matrix
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * np.log(p), 0.0)
        entropy = -terms.sum(axis=-1)
        return np.exp(entropy)


def _select_best_alignment(text_arrays: dict) -> tuple:
    """Score every (node, head, stream) in the text family by Spearman rho
    between frame index and argmax text unit, and return the best one as
    (node, head, stream, rho). NaN scores (constant argmax) are skipped."""
    best_node = best_head = best_stream = None
    best_rho = float("-inf")
    for node, arr in text_arrays.items():
        n_heads, n_streams, n_frames, _ = arr.shape
        frame_idx = np.arange(n_frames)
        for h in range(n_heads):
            for s in range(n_streams):
                argmax_seq = arr[h, s].argmax(axis=-1)
                # A constant argmax sequence (attention collapsed onto one
                # text unit for the whole utterance) makes rho undefined;
                # scipy warns and returns NaN, which we skip below.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConstantInputWarning)
                    rho, _ = spearmanr(frame_idx, argmax_seq)
                if np.isnan(rho):
                    continue
                if rho > best_rho:
                    best_node, best_head, best_stream, best_rho = node, h, s, rho
    if best_node is None:
        raise ValueError(
            "No (node, head, stream) combination produced a defined Spearman "
            "rho -- every argmax sequence was constant."
        )
    return best_node, best_head, best_stream, best_rho


def analyze(
    tts: TextToSpeech,
    text: str,
    lang: str,
    style: Style,
    total_step: int = 8,
    speed: float = 1.0,
    seed: Optional[int] = None,
    attention_path: str = DEFAULT_ATTENTION_PATH,
) -> tuple:
    """Run the full pipeline once with an instrumented `vector_estimator`
    session substituted in, and return (wav, alignment, style_attention).

    This re-implements `TextToSpeech._infer`'s denoising loop rather than
    calling it, since `_infer` has nowhere to hand the extra Softmax outputs
    back to. `test_attention.py::test_instrumented_render_matches_helper`
    exists specifically to catch drift between the two loops -- keep them in
    lockstep if you touch either.

    `tts`'s `vector_est_ort` is swapped out only for the duration of this
    call and restored in a `finally`, so the caller's object is left as it
    was found.
    """
    assert style.ttl.shape[0] == 1, "analyze() only supports a single style/text"

    src_path = _source_path_for(attention_path)
    instrumented_session = _get_instrumented_session(attention_path, src_path)
    output_names = [o.name for o in instrumented_session.get_outputs()]

    original_session = tts.vector_est_ort
    tts.vector_est_ort = instrumented_session
    try:
        text_list, lang_list = [text], [lang]
        text_ids, text_mask = tts.text_processor(text_list, lang_list)
        dur_onnx, *_ = tts.dp_ort.run(
            None,
            {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask},
        )
        dur_onnx = dur_onnx / speed
        text_emb_onnx, *_ = tts.text_enc_ort.run(
            None,
            {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask},
        )
        xt, latent_mask = tts.sample_noisy_latent(dur_onnx, seed=seed)
        bsz = len(text_list)
        total_step_np = np.array([total_step] * bsz, dtype=np.float32)

        final_outputs = None
        for step in range(total_step):
            current_step = np.array([step] * bsz, dtype=np.float32)
            outputs = instrumented_session.run(
                output_names,
                {
                    "noisy_latent": xt,
                    "text_emb": text_emb_onnx,
                    "style_ttl": style.ttl,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": current_step,
                    "total_step": total_step_np,
                },
            )
            by_name = dict(zip(output_names, outputs))
            xt = by_name["denoised_latent"]
            if step == total_step - 1:
                final_outputs = by_name
        wav, *_ = tts.vocoder_ort.run(None, {"latent": xt})
    finally:
        tts.vector_est_ort = original_session

    text_arrays = {}
    style_arrays = {}
    for name, arr in final_outputs.items():
        if name == "denoised_latent":
            continue
        short_name, family = _short_node_name(name)
        if family == "text":
            text_arrays[short_name] = arr
        else:
            style_arrays[short_name] = arr

    node, head, stream, rho = _select_best_alignment(text_arrays)
    alignment_matrix = text_arrays[node][head, stream]

    indexer = tts.text_processor.indexer
    inv = _invert_indexer(indexer)
    tokens = [inv[int(tid)] for tid in text_ids[0]]

    frame_seconds = tts.base_chunk_size * tts.chunk_compress_factor / tts.sample_rate
    duration = float(np.asarray(dur_onnx).reshape(-1)[0])

    alignment = Alignment(
        matrix=alignment_matrix,
        tokens=tokens,
        frame_seconds=frame_seconds,
        node=node,
        head=head,
        stream=stream,
        spearman=rho,
        duration=duration,
    )

    style_mats = [arr.mean(axis=(0, 1)) for arr in style_arrays.values()]
    style_matrix = np.mean(style_mats, axis=0)
    style_attention = StyleAttention(matrix=style_matrix, frame_seconds=frame_seconds)

    return wav, alignment, style_attention


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Read out the vector_estimator attention this fork's ONNX graph never exports."
    )
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--voice", type=str, default="assets/voice_styles/M1.json")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--onnx-dir", type=str, default="assets/onnx")
    return parser.parse_args()


def main():
    args = _parse_args()
    tts = load_text_to_speech(args.onnx_dir)
    style = load_voice_style([args.voice])
    attention_path = os.path.join(args.onnx_dir, "vector_estimator.attn.onnx")

    _wav, alignment, style_attention = analyze(
        tts,
        args.text,
        args.lang,
        style,
        total_step=args.steps,
        speed=args.speed,
        seed=args.seed,
        attention_path=attention_path,
    )

    print(f"Node: {alignment.node}  head={alignment.head}  stream={alignment.stream}  "
          f"spearman={alignment.spearman:.4f}  duration={alignment.duration:.3f}s")
    print()
    print(f"{'word':<12}{'start_s':>10}{'end_s':>10}")
    for word, start, end in alignment.word_spans():
        print(f"{word:<12}{start:>10.2f}{end:>10.2f}")

    fd = style_attention.frame_divergence()
    er = style_attention.effective_rows()
    print()
    print(
        f"style frame_divergence: mean={fd.mean():.3f} min={fd.min():.3f} max={fd.max():.3f}"
    )
    print(
        f"style effective_rows:   mean={er.mean():.1f} min={er.min():.1f} max={er.max():.1f} of {style_attention.matrix.shape[1]}"
    )


if __name__ == "__main__":
    main()
