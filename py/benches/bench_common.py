"""Shared helpers for the listening-bench generators in this directory.

Each generator (`phase0_linearity_gate_bench.py`, `phase0_row_locality_bench.py`,
`phase2b_prosody_correction_bench.py`, `phase2b_direction_collapse_bench.py`)
reads WAVs and PNGs from `py/results/...`, base64-embeds them, and writes a
standalone HTML file — see docs/LISTENING_BENCHES.md for what each one covers
and the verdict it produced.

This module holds only the two pieces that were byte-identical, or safely
factorable, across all four scripts when they were pulled into the repo: the
base64 read and the "playing one clip pauses the others" behavior. Everything
else — CSS tokens, page-specific JS (keyboard shortcuts, hover tooltips) —
stays duplicated per script on purpose. The CSS blocks differ in small ways
between scripts (extra tokens, font-stack fallbacks) that were not safe to
unify without re-running every generator to confirm the merged output still
renders identically, and this pass was explicitly not to run them.
"""

import base64
import pathlib


def b64(path) -> str:
    """Read a file and return its contents as a base64 string for a data: URI."""
    return base64.b64encode(pathlib.Path(path).read_bytes()).decode()


# The "playing one clip stops the others" behavior every bench page shares, so
# comparison stays A/B instead of overlapping. Verbatim across all four
# scripts before this move. Some benches append further listeners (keyboard
# shortcuts, hover tooltips) after this block; see each script's own
# <script> section.
PLAYER_PAUSE_SCRIPT = """  var players = Array.prototype.slice.call(document.querySelectorAll('audio'));
  players.forEach(function (a) {
    a.addEventListener('play', function () {
      players.forEach(function (o) { if (o !== a) { o.pause(); } });
    });
  });"""
