"""Veydra Model Standard package."""

from .base import (
    VeydraModelStandard,
    Submodel,
    SimulationContext,
    META_KEYS,
    DEFAULT_SUMMARY_STATS,
    compute_summary_stats,
    build_summary_table,
    build_stacked_table,
    apply_time_window,
)

__version__ = "1.1.0"
__all__ = [
    "VeydraModelStandard",
    "Submodel",
    "SimulationContext",
    "META_KEYS",
    "DEFAULT_SUMMARY_STATS",
    "compute_summary_stats",
    "build_summary_table",
    "build_stacked_table",
    "apply_time_window",
]