"""2D plotting utilities for saved LLM fingerprints."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from llm_fingerprinter import config
from llm_fingerprinter.fingerprint_store import FingerprintStore


class FingerprintPlotError(Exception):
    """Raised when a fingerprint projection plot cannot be created."""


def _label_for(data: dict[str, Any], label_field: str) -> str:
    metadata = data.get("metadata") or {}

    if label_field == "model":
        return str(metadata.get("model_name") or data.get("model") or "unknown")

    value = data.get(label_field)
    if value is None:
        value = metadata.get(label_field)

    return str(value or "unknown")


def _split_projection_features(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split one raw fingerprint into semantic and stylistic feature blocks."""
    vector = np.asarray(vector, dtype=np.float32)
    expected_dim = config.RAW_FINGERPRINT_DIM

    if vector.shape[0] != expected_dim:
        raise ValueError(f"expected {expected_dim} dims, got {vector.shape[0]}")

    semantic_parts = []
    stylistic_parts = []
    layer_dim = config.PER_PROMPT_FEATURE_DIM

    for layer_idx in range(len(config.LAYER_ORDER)):
        start = layer_idx * layer_dim
        embed_end = start + config.EMBEDDING_DIM
        layer_end = start + layer_dim

        semantic_parts.append(vector[start:embed_end])
        stylistic_parts.append(vector[embed_end:layer_end])

    return np.concatenate(semantic_parts), np.concatenate(stylistic_parts)


def _project_2d(features: np.ndarray, method: str) -> np.ndarray:
    if method != "pca":
        raise FingerprintPlotError(f"Unsupported projection method: {method}")

    if features.shape[0] < 2:
        raise FingerprintPlotError("Need at least 2 valid fingerprints to draw a plot.")

    scaled = StandardScaler().fit_transform(features)
    return PCA(n_components=2, random_state=42).fit_transform(scaled)


def _load_plot_rows(input_dir: Path, label_field: str) -> tuple[list[dict[str, Any]], list[str]]:
    store = FingerprintStore(str(input_dir))
    files = store.list_fingerprints()

    if not files:
        raise FingerprintPlotError(
            f"No fingerprints found in {input_dir}. "
            "Run 'llm-fingerprinter simulate' first."
        )

    rows = []
    skipped = []

    for filepath in files:
        data = store.load_fingerprint(str(filepath))
        if not data:
            skipped.append(f"{filepath.name}: could not load JSON")
            continue

        vector = store._get_full_vector(data)
        if vector is None:
            skipped.append(f"{filepath.name}: missing fingerprint vector")
            continue

        try:
            semantic, stylistic = _split_projection_features(vector)
        except ValueError as exc:
            skipped.append(f"{filepath.name}: {exc}")
            continue

        rows.append({
            "path": filepath,
            "label": _label_for(data, label_field),
            "semantic": semantic,
            "stylistic": stylistic,
        })

    if len(rows) < 2:
        raise FingerprintPlotError(
            f"Need at least 2 valid 1206-dim fingerprints; found {len(rows)}. "
            "Run more simulations before plotting."
        )

    return rows, skipped


def plot_fingerprint_projection(
    input_dir: str | Path,
    output_path: str | Path,
    label_field: str = "family",
    method: str = "pca",
) -> dict[str, Any]:
    """Create a two-panel semantic/stylistic PCA scatter plot."""
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    rows, skipped = _load_plot_rows(input_dir, label_field)
    labels = [row["label"] for row in rows]
    semantic = np.vstack([row["semantic"] for row in rows])
    stylistic = np.vstack([row["stylistic"] for row in rows])

    semantic_xy = _project_2d(semantic, method)
    stylistic_xy = _project_2d(stylistic, method)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise FingerprintPlotError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20" if len(unique_labels) > 10 else "tab10")
    markers = ["o", "^", "s", "D", "P", "X", "v", "<", ">", "*", "h", "p"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    panels = [
        (axes[0], semantic_xy, "Semantic fingerprint embeddings"),
        (axes[1], stylistic_xy, "Stylistic fingerprint features"),
    ]

    for ax, coords, title in panels:
        for idx, label in enumerate(unique_labels):
            mask = np.array([item == label for item in labels])
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                label=label,
                marker=markers[idx % len(markers)],
                color=cmap(idx % cmap.N),
                s=48,
                alpha=0.86,
                edgecolors="white",
                linewidths=0.45,
            )
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, color="#9ca3af", alpha=0.35, linewidth=0.7)
        ax.legend(frameon=True, fancybox=False, edgecolor="#444444")

    fig.suptitle(f"LLM Fingerprint Projection by {label_field}", y=1.02)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "output": output_path,
        "input": input_dir,
        "samples": len(rows),
        "labels": dict(Counter(labels)),
        "semantic_dim": semantic.shape[1],
        "stylistic_dim": stylistic.shape[1],
        "skipped": skipped,
        "method": method,
    }
