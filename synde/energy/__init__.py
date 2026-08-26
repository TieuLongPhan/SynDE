"""Graph-derived molecular and reaction scoring, in explicit model units.

Public names resolve lazily: importing this package costs almost nothing,
and each submodule loads only when a symbol it defines is first accessed.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

# Public export name -> the submodule that defines it.
_EXPORTS: dict[str, str] = {
    "CHARGE_TOPOLOGY_FAMILY": ".interpretable_two_d_v3",
    "CYCLE_JUNCTION_FAMILY": ".interpretable_two_d_v3",
    "CalibratedPrediction": ".calibration",
    "CalibrationRecord": ".calibration",
    "ENERGY_MODEL_RESOURCE": ".energy_predictor",
    "EmpiricalTwoDEnergyConfig": ".empirical_two_d_energy",
    "EmpiricalTwoDEnergyScorer": ".empirical_two_d_energy",
    "EnergyRefinementRecord": ".refinement",
    "EnergyRefinementReport": ".refinement",
    "FROZEN_MODEL_SHA256": ".synde_model",
    "FirstOrderTwoDEnergyConfig": ".first_order_two_d_energy",
    "FirstOrderTwoDEnergyScorer": ".first_order_two_d_energy",
    "FormulaCalibrationRecord": ".two_d_calibration",
    "FormulaRelativeEnsemble": ".two_d_ensemble",
    "FormulaRelativeEnsembleCard": ".two_d_ensemble",
    "FormulaRelativeEnsemblePrediction": ".two_d_ensemble",
    "FormulaRelativeModelCard": ".two_d_calibration",
    "FormulaRelativePrediction": ".two_d_calibration",
    "FormulaRelativeRidgeCalibrator": ".two_d_calibration",
    "GFN2SinglePointConfig": ".semiempirical_energy",
    "GFN2SinglePointScorer": ".semiempirical_energy",
    "GFN2TwoCycleConfig": ".truncated_scc_energy",
    "GFN2TwoCycleScorer": ".truncated_scc_energy",
    "GFN2XTBProxyConfig": ".xtb_proxy_energy",
    "GFN2XTBProxyScorer": ".xtb_proxy_energy",
    "GFN2_XTB_VALENCE_REFERENCE_EV": ".xtb_proxy_energy",
    "GRAPH_STERIC_FAMILY": ".interpretable_two_d_v3",
    "GraphEnergy": ".graph_energy",
    "GraphTheoryEnergyConfig": ".graph_theory_energy",
    "GraphTheoryEnergyScorer": ".graph_theory_energy",
    "HUCKEL_DENSITY_FAMILY": ".interpretable_two_d_v3",
    "ITSScoreResult": ".results",
    "ITSScorer": ".its_scoring",
    "ITSScoringConfig": ".its_scoring",
    "InterpretableTwoDV2Config": ".interpretable_two_d_v2",
    "InterpretableTwoDV2Scorer": ".interpretable_two_d_v2",
    "ModelCard": ".calibration",
    "MoleculeScoreResult": ".results",
    "MoleculeScorer": ".molecule_scoring",
    "MoleculeScoringConfig": ".molecule_scoring",
    "NAMED_FEATURE_SCHEMA": ".interpretable_two_d_v2",
    "OrbitalTheoryEnergyScorer": ".orbital_theory_energy",
    "OrdCalibratedV4ModelCard": ".ord_calibrated_v4",
    "OrdCalibratedV4Scorer": ".ord_calibrated_v4",
    "QUANTUM_GRAPH_FEATURE_SCHEMA_V3": ".interpretable_two_d_v3",
    "QuantumGraphV3ModelCard": ".quantum_graph_v3",
    "QuantumGraphV3Scorer": ".quantum_graph_v3",
    "RESONANCE_TOPOLOGY_FAMILY": ".interpretable_two_d_v3",
    "ReactionScoreResult": ".results",
    "ReactionScorer": ".reaction_scoring",
    "ReactionScoringConfig": ".reaction_scoring",
    "RidgeCalibrator": ".calibration",
    "SparseInterpretableTwoDModelCard": ".sparse_interpretable_two_d",
    "SparseInterpretableTwoDScorer": ".sparse_interpretable_two_d",
    "SynDEEnergyModelCard": ".energy_predictor",
    "SynDEEnergyPrediction": ".energy_predictor",
    "SynDEEnergyPredictor": ".energy_predictor",
    "SynDEEnergyRanking": ".energy_predictor",
    "SynDEEnergyRefiner": ".refinement",
    "SynDEModelCard": ".synde_model",
    "SynDEScorer": ".synde_model",
    "SynDEValidationRecord": ".synde_model",
    "TWO_D_FEATURE_SCHEMA": ".two_d_features",
    "TheoryEnergyConfig": ".theory_energy",
    "TheoryEnergyScorer": ".theory_energy",
    "TwoDEnergyConfig": ".two_d_energy",
    "TwoDFeatureConfig": ".two_d_features",
    "V3FeatureDefinition": ".interpretable_two_d_v3",
    "V3_FEATURE_DEFINITIONS": ".interpretable_two_d_v3",
    "V3_FEATURE_FAMILIES": ".interpretable_two_d_v3",
    "V3_FEATURE_NAMES": ".interpretable_two_d_v3",
    "ValenceEnergyConfig": ".valence_energy",
    "ValenceEnergyScorer": ".valence_energy",
    "ValenceEnergyWeights": ".valence_energy",
    "deterministic_split": ".calibration",
    "extract_heuristic_features": ".two_d_features",
    "extract_named_empirical_two_d_features": ".interpretable_two_d_v2",
    "extract_quantum_graph_v3_features": ".interpretable_two_d_v3",
    "extract_two_d_features": ".two_d_features",
    "molecular_composition": ".energy_predictor",
    "molecular_formula_charge_key": ".two_d_calibration",
    "theory_energy_corrections": ".theory_energy",
    "two_d_energy_components": ".two_d_energy",
    "uncalibrated_v2_additions": ".interpretable_two_d_v2",
    "v3_feature_family": ".interpretable_two_d_v3",
    "valence_energy_terms": ".valence_energy",
}

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from .calibration import (  # noqa: F401
        CalibratedPrediction,
        CalibrationRecord,
        ModelCard,
        RidgeCalibrator,
        deterministic_split,
    )
    from .empirical_two_d_energy import (
        EmpiricalTwoDEnergyConfig,
        EmpiricalTwoDEnergyScorer,
    )  # noqa: F401
    from .energy_predictor import (  # noqa: F401
        ENERGY_MODEL_RESOURCE,
        SynDEEnergyModelCard,
        SynDEEnergyPrediction,
        SynDEEnergyPredictor,
        SynDEEnergyRanking,
        molecular_composition,
    )
    from .first_order_two_d_energy import (
        FirstOrderTwoDEnergyConfig,
        FirstOrderTwoDEnergyScorer,
    )  # noqa: F401
    from .graph_energy import GraphEnergy  # noqa: F401
    from .graph_theory_energy import (
        GraphTheoryEnergyConfig,
        GraphTheoryEnergyScorer,
    )  # noqa: F401
    from .interpretable_two_d_v2 import (  # noqa: F401
        InterpretableTwoDV2Config,
        InterpretableTwoDV2Scorer,
        NAMED_FEATURE_SCHEMA,
        extract_named_empirical_two_d_features,
        uncalibrated_v2_additions,
    )
    from .interpretable_two_d_v3 import (  # noqa: F401
        CHARGE_TOPOLOGY_FAMILY,
        CYCLE_JUNCTION_FAMILY,
        GRAPH_STERIC_FAMILY,
        HUCKEL_DENSITY_FAMILY,
        QUANTUM_GRAPH_FEATURE_SCHEMA_V3,
        RESONANCE_TOPOLOGY_FAMILY,
        V3FeatureDefinition,
        V3_FEATURE_DEFINITIONS,
        V3_FEATURE_FAMILIES,
        V3_FEATURE_NAMES,
        extract_quantum_graph_v3_features,
        v3_feature_family,
    )
    from .its_scoring import ITSScorer, ITSScoringConfig  # noqa: F401
    from .molecule_scoring import MoleculeScorer, MoleculeScoringConfig  # noqa: F401
    from .orbital_theory_energy import OrbitalTheoryEnergyScorer  # noqa: F401
    from .ord_calibrated_v4 import (
        OrdCalibratedV4ModelCard,
        OrdCalibratedV4Scorer,
    )  # noqa: F401
    from .quantum_graph_v3 import (
        QuantumGraphV3ModelCard,
        QuantumGraphV3Scorer,
    )  # noqa: F401
    from .reaction_scoring import ReactionScorer, ReactionScoringConfig  # noqa: F401
    from .refinement import (
        EnergyRefinementRecord,
        EnergyRefinementReport,
        SynDEEnergyRefiner,
    )  # noqa: F401
    from .results import (
        ITSScoreResult,
        MoleculeScoreResult,
        ReactionScoreResult,
    )  # noqa: F401
    from .semiempirical_energy import (
        GFN2SinglePointConfig,
        GFN2SinglePointScorer,
    )  # noqa: F401
    from .sparse_interpretable_two_d import (  # noqa: F401
        SparseInterpretableTwoDModelCard,
        SparseInterpretableTwoDScorer,
    )
    from .synde_model import (
        FROZEN_MODEL_SHA256,
        SynDEModelCard,
        SynDEScorer,
        SynDEValidationRecord,
    )  # noqa: F401
    from .theory_energy import (
        TheoryEnergyConfig,
        TheoryEnergyScorer,
        theory_energy_corrections,
    )  # noqa: F401
    from .truncated_scc_energy import (
        GFN2TwoCycleConfig,
        GFN2TwoCycleScorer,
    )  # noqa: F401
    from .two_d_calibration import (  # noqa: F401
        FormulaCalibrationRecord,
        FormulaRelativeModelCard,
        FormulaRelativePrediction,
        FormulaRelativeRidgeCalibrator,
        molecular_formula_charge_key,
    )
    from .two_d_energy import TwoDEnergyConfig, two_d_energy_components  # noqa: F401
    from .two_d_ensemble import (  # noqa: F401
        FormulaRelativeEnsemble,
        FormulaRelativeEnsembleCard,
        FormulaRelativeEnsemblePrediction,
    )
    from .two_d_features import (  # noqa: F401
        TWO_D_FEATURE_SCHEMA,
        TwoDFeatureConfig,
        extract_heuristic_features,
        extract_two_d_features,
    )
    from .valence_energy import (  # noqa: F401
        ValenceEnergyConfig,
        ValenceEnergyScorer,
        ValenceEnergyWeights,
        valence_energy_terms,
    )
    from .xtb_proxy_energy import (
        GFN2XTBProxyConfig,
        GFN2XTBProxyScorer,
        GFN2_XTB_VALENCE_REFERENCE_EV,
    )  # noqa: F401

__all__ = [
    "CHARGE_TOPOLOGY_FAMILY",
    "CYCLE_JUNCTION_FAMILY",
    "CalibratedPrediction",
    "CalibrationRecord",
    "ENERGY_MODEL_RESOURCE",
    "EmpiricalTwoDEnergyConfig",
    "EmpiricalTwoDEnergyScorer",
    "EnergyRefinementRecord",
    "EnergyRefinementReport",
    "FROZEN_MODEL_SHA256",
    "FirstOrderTwoDEnergyConfig",
    "FirstOrderTwoDEnergyScorer",
    "FormulaCalibrationRecord",
    "FormulaRelativeEnsemble",
    "FormulaRelativeEnsembleCard",
    "FormulaRelativeEnsemblePrediction",
    "FormulaRelativeModelCard",
    "FormulaRelativePrediction",
    "FormulaRelativeRidgeCalibrator",
    "GFN2SinglePointConfig",
    "GFN2SinglePointScorer",
    "GFN2TwoCycleConfig",
    "GFN2TwoCycleScorer",
    "GFN2XTBProxyConfig",
    "GFN2XTBProxyScorer",
    "GFN2_XTB_VALENCE_REFERENCE_EV",
    "GRAPH_STERIC_FAMILY",
    "GraphEnergy",
    "GraphTheoryEnergyConfig",
    "GraphTheoryEnergyScorer",
    "HUCKEL_DENSITY_FAMILY",
    "ITSScoreResult",
    "ITSScorer",
    "ITSScoringConfig",
    "InterpretableTwoDV2Config",
    "InterpretableTwoDV2Scorer",
    "ModelCard",
    "MoleculeScoreResult",
    "MoleculeScorer",
    "MoleculeScoringConfig",
    "NAMED_FEATURE_SCHEMA",
    "OrbitalTheoryEnergyScorer",
    "OrdCalibratedV4ModelCard",
    "OrdCalibratedV4Scorer",
    "QUANTUM_GRAPH_FEATURE_SCHEMA_V3",
    "QuantumGraphV3ModelCard",
    "QuantumGraphV3Scorer",
    "RESONANCE_TOPOLOGY_FAMILY",
    "ReactionScoreResult",
    "ReactionScorer",
    "ReactionScoringConfig",
    "RidgeCalibrator",
    "SparseInterpretableTwoDModelCard",
    "SparseInterpretableTwoDScorer",
    "SynDEEnergyModelCard",
    "SynDEEnergyPrediction",
    "SynDEEnergyPredictor",
    "SynDEEnergyRanking",
    "SynDEEnergyRefiner",
    "SynDEModelCard",
    "SynDEScorer",
    "SynDEValidationRecord",
    "TWO_D_FEATURE_SCHEMA",
    "TheoryEnergyConfig",
    "TheoryEnergyScorer",
    "TwoDEnergyConfig",
    "TwoDFeatureConfig",
    "V3FeatureDefinition",
    "V3_FEATURE_DEFINITIONS",
    "V3_FEATURE_FAMILIES",
    "V3_FEATURE_NAMES",
    "ValenceEnergyConfig",
    "ValenceEnergyScorer",
    "ValenceEnergyWeights",
    "deterministic_split",
    "extract_heuristic_features",
    "extract_named_empirical_two_d_features",
    "extract_quantum_graph_v3_features",
    "extract_two_d_features",
    "molecular_composition",
    "molecular_formula_charge_key",
    "theory_energy_corrections",
    "two_d_energy_components",
    "uncalibrated_v2_additions",
    "v3_feature_family",
    "valence_energy_terms",
]


def __getattr__(name: str):
    """Import and cache one public symbol on first access.

    :param name: Attribute requested from this package.
    :type name: str
    :return: The requested public symbol.
    :rtype: object
    :raises AttributeError: If the name is not a public SynDE export.
    """
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public names this package exports.

    :return: Sorted public export names.
    :rtype: list[str]
    """
    return sorted(__all__)
