"""Utilities for preparing grouped fingerprint training data."""

from __future__ import annotations

from typing import TypeVar

import numpy as np


T = TypeVar("T")


def balance_grouped_samples(
    grouped_samples: dict[str, list[T]],
    seed: int = 42,
) -> tuple[dict[str, list[T]], int]:
    """Downsample every non-empty group to the smallest group size."""
    non_empty = {
        group: list(samples)
        for group, samples in grouped_samples.items()
        if samples
    }
    if not non_empty:
        return {}, 0

    target_count = min(len(samples) for samples in non_empty.values())
    rng = np.random.default_rng(seed)
    balanced: dict[str, list[T]] = {}

    for group in sorted(non_empty):
        samples = non_empty[group]
        if len(samples) == target_count:
            balanced[group] = samples
            continue

        indices = np.sort(
            rng.choice(len(samples), size=target_count, replace=False)
        )
        balanced[group] = [samples[int(index)] for index in indices]

    return balanced, target_count
