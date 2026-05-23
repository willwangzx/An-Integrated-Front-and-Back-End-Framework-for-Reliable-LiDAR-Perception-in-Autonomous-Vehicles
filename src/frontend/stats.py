"""Helpers for frontend processing statistics."""

from typing import Dict

from .constants import BASE_FRONTEND_STATS_KEYS, TEMPORAL_FRONTEND_STATS_KEYS


def make_frontend_stats(include_temporal: bool = False) -> Dict[str, int]:
    stats = {key: 0 for key in BASE_FRONTEND_STATS_KEYS}
    if include_temporal:
        stats.update({key: 0 for key in TEMPORAL_FRONTEND_STATS_KEYS})
    return stats


def merge_frontend_stats(accumulator: Dict[str, int], delta: Dict[str, int]) -> None:
    for key in accumulator:
        accumulator[key] += int(delta.get(key, 0))
