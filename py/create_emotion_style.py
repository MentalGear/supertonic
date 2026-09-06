"""Create a reusable emotion style-difference JSON from calibrated styles.

Example:
    python create_emotion_style.py \
        --neutral M1_neutral.json F1_neutral.json \
        --emotional M1_surprised.json F1_surprised.json \
        --emotion surprised \
        --output ../assets/emotion_styles/surprised.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def read_component(path: Path, name: str) -> tuple[np.ndarray, list[int], str]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    component = payload[name]
    dims = component["dims"]
    data = np.asarray(component["data"], dtype=np.float32)
    expected_size = int(np.prod(dims))
    if data.size != expected_size:
        raise ValueError(
            f"{path}: {name} contains {data.size} values, expected {expected_size}"
        )
    return data.reshape(dims), dims, component.get("type", "float32")


def average_component(paths: list[Path], name: str) -> tuple[np.ndarray, list[int]]:
    values = []
    dims = None
    for path in paths:
        value, current_dims, _ = read_component(path, name)
        if dims is None:
            dims = current_dims
        elif current_dims != dims:
            raise ValueError(f"{path}: {name} dimensions {current_dims} do not match {dims}")
        values.append(value)
    return np.mean(values, axis=0, dtype=np.float32), dims


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a Supertonic emotion style-difference JSON"
    )
    parser.add_argument("--neutral", nargs="+", type=Path, required=True)
    parser.add_argument("--emotional", nargs="+", type=Path, required=True)
    parser.add_argument("--emotion", choices=["surprised", "angry"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.neutral) != len(args.emotional):
        parser.error("--neutral and --emotional must contain the same number of files")

    deltas = {}
    dimensions = {}
    for name in ("style_ttl", "style_dp"):
        neutral, dims = average_component(args.neutral, name)
        emotional, emotional_dims = average_component(args.emotional, name)
        if dims != emotional_dims:
            raise ValueError(f"{name} dimensions differ between neutral and emotional styles")
        deltas[name] = (emotional - neutral).tolist()
        dimensions[name] = dims

    output = {
        name: {"data": data, "dims": dimensions[name], "type": "float32"}
        for name, data in deltas.items()
    }
    output["metadata"] = {
        "emotion": args.emotion,
        "kind": "style_difference",
        "neutral_styles": [str(path) for path in args.neutral],
        "emotional_styles": [str(path) for path in args.emotional],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output, file)
    print(f"Wrote {args.emotion} emotion style difference to {args.output}")


if __name__ == "__main__":
    main()