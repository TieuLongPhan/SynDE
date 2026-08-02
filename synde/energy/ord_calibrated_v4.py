"""Compatibility names retained for frozen benchmark reproduction only."""

from .synde_model import SynDEModelCard, SynDEScorer

OrdCalibratedV4ModelCard = SynDEModelCard
OrdCalibratedV4Scorer = SynDEScorer

__all__ = ["OrdCalibratedV4ModelCard", "OrdCalibratedV4Scorer"]
