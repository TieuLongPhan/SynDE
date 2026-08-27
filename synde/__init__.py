"""SynDE: interpretable 2D prediction of protocol-defined GFN2-xTB energies.

Public names resolve lazily, so ``import synde`` does not pull in RDKit,
NetworkX, NumPy, or the full energy stack until a symbol is actually used.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _package_version
from typing import TYPE_CHECKING

try:  # pragma: no cover - depends on installation mode
    __version__ = _package_version("synde")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.5.0"

# Public export name -> the subpackage that defines it.
_EXPORTS: dict[str, str] = {
    "CalibratedPrediction": ".energy",
    "CalibrationRecord": ".energy",
    "ComponentFrontier": ".graph",
    "DirectionalFMO": ".graph",
    "EnergyRefinementRecord": ".energy",
    "EnergyRefinementReport": ".energy",
    "FROZEN_MODEL_SHA256": ".energy",
    "FormulaCalibrationRecord": ".energy",
    "FormulaRelativeEnsemble": ".energy",
    "FormulaRelativeEnsembleCard": ".energy",
    "FormulaRelativeEnsemblePrediction": ".energy",
    "FormulaRelativeModelCard": ".energy",
    "FormulaRelativePrediction": ".energy",
    "FormulaRelativeRidgeCalibrator": ".energy",
    "GFN2SinglePointConfig": ".energy",
    "GFN2SinglePointScorer": ".energy",
    "GFN2TwoCycleConfig": ".energy",
    "GFN2TwoCycleScorer": ".energy",
    "GFN2XTBProxyConfig": ".energy",
    "GFN2XTBProxyScorer": ".energy",
    "GFN2_XTB_VALENCE_REFERENCE_EV": ".energy",
    "GeneralizedHuckel": ".graph",
    "GeneralizedHuckelResult": ".graph",
    "GeometryScorer": ".geometry",
    "GraphBuilder": ".graph",
    "GraphEnergy": ".energy",
    "GraphNormalizer": ".graph",
    "GraphPairScorer": ".graph",
    "GraphTheoryEnergyConfig": ".energy",
    "GraphTheoryEnergyScorer": ".energy",
    "GraphValidationError": ".graph",
    "GraphWarning": ".graph",
    "GraphXTBCascade": ".integration",
    "HSABDescriptor": ".graph",
    "HuckelParameters": ".graph",
    "HuckelSystemResult": ".graph",
    "ITSGraph": ".graph",
    "ITSGraphBuilder": ".graph",
    "ITSScoreResult": ".energy",
    "ITSScorer": ".energy",
    "ITSScoringConfig": ".energy",
    "ModelCard": ".energy",
    "MoleculeScoreResult": ".energy",
    "MoleculeScorer": ".energy",
    "MoleculeScoringConfig": ".energy",
    "NormalizedMolecularGraph": ".graph",
    "OrbitalNode": ".graph",
    "OrbitalTheoryEnergyScorer": ".energy",
    "PairScoreGroup": ".graph",
    "PairScoreResult": ".graph",
    "PairScoringConfig": ".graph",
    "PiAssignmentResult": ".graph",
    "PiAtom": ".graph",
    "PiSystem": ".graph",
    "PiSystemAssigner": ".graph",
    "ReactionScoreResult": ".energy",
    "ReactionScorer": ".energy",
    "ReactionScoringConfig": ".energy",
    "RidgeCalibrator": ".energy",
    "SFEnergy": ".integration",
    "SynDEDomainError": ".errors",
    "SynDEEnergyModelCard": ".energy",
    "SynDEEnergyPrediction": ".energy",
    "SynDEEnergyPredictor": ".energy",
    "SynDEEnergyRanking": ".energy",
    "SynDEEnergyRefiner": ".energy",
    "SynDEError": ".errors",
    "SynDEInputError": ".errors",
    "SynDEModelCard": ".energy",
    "SynDEScorer": ".energy",
    "SynDEValidationRecord": ".energy",
    "TWO_D_FEATURE_SCHEMA": ".energy",
    "TheoryEnergyConfig": ".energy",
    "TheoryEnergyScorer": ".energy",
    "TwoDEnergyConfig": ".energy",
    "TwoDFeatureConfig": ".energy",
    "ValenceEnergyConfig": ".energy",
    "ValenceEnergyScorer": ".energy",
    "ValenceEnergyWeights": ".energy",
    "assign_orbital_pi": ".graph",
    "assign_pi_systems": ".graph",
    "deterministic_split": ".energy",
    "directional_fmo": ".graph",
    "extract_heuristic_features": ".energy",
    "extract_two_d_features": ".energy",
    "hsab_descriptor": ".graph",
    "local_hsab_compatibility": ".graph",
    "molecular_composition": ".energy",
    "molecular_formula_charge_key": ".energy",
    "normalize_graph": ".graph",
    "solve_generalized_huckel": ".graph",
    "theory_energy_corrections": ".energy",
    "two_d_energy_components": ".energy",
    "valence_energy_terms": ".energy",
}

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from .energy import (  # noqa: F401
        CalibratedPrediction,
        CalibrationRecord,
        EnergyRefinementRecord,
        EnergyRefinementReport,
        FROZEN_MODEL_SHA256,
        FormulaCalibrationRecord,
    )
    from .energy import (  # noqa: F401
        FormulaRelativeEnsemble,
        FormulaRelativeEnsembleCard,
        FormulaRelativeEnsemblePrediction,
        FormulaRelativeModelCard,
        FormulaRelativePrediction,
        FormulaRelativeRidgeCalibrator,
    )
    from .energy import (  # noqa: F401
        GFN2SinglePointConfig,
        GFN2SinglePointScorer,
        GFN2TwoCycleConfig,
        GFN2TwoCycleScorer,
        GFN2XTBProxyConfig,
        GFN2XTBProxyScorer,
    )
    from .energy import (  # noqa: F401
        GFN2_XTB_VALENCE_REFERENCE_EV,
        GraphEnergy,
        GraphTheoryEnergyConfig,
        GraphTheoryEnergyScorer,
        ITSScoreResult,
        ITSScorer,
    )
    from .energy import (  # noqa: F401
        ITSScoringConfig,
        ModelCard,
        MoleculeScoreResult,
        MoleculeScorer,
        MoleculeScoringConfig,
        OrbitalTheoryEnergyScorer,
    )
    from .energy import (  # noqa: F401
        ReactionScoreResult,
        ReactionScorer,
        ReactionScoringConfig,
        RidgeCalibrator,
        SynDEEnergyModelCard,
        SynDEEnergyPrediction,
    )
    from .energy import (  # noqa: F401
        SynDEEnergyPredictor,
        SynDEEnergyRanking,
        SynDEEnergyRefiner,
        SynDEModelCard,
        SynDEScorer,
        SynDEValidationRecord,
    )
    from .energy import (  # noqa: F401
        TWO_D_FEATURE_SCHEMA,
        TheoryEnergyConfig,
        TheoryEnergyScorer,
        TwoDEnergyConfig,
        TwoDFeatureConfig,
        ValenceEnergyConfig,
    )
    from .energy import (  # noqa: F401
        ValenceEnergyScorer,
        ValenceEnergyWeights,
        deterministic_split,
        extract_heuristic_features,
        extract_two_d_features,
        molecular_composition,
    )
    from .energy import (  # noqa: F401
        molecular_formula_charge_key,
        theory_energy_corrections,
        two_d_energy_components,
        valence_energy_terms,
    )
    from .errors import SynDEDomainError, SynDEError, SynDEInputError  # noqa: F401
    from .geometry import GeometryScorer  # noqa: F401
    from .graph import (  # noqa: F401
        ComponentFrontier,
        DirectionalFMO,
        GeneralizedHuckel,
        GeneralizedHuckelResult,
        GraphBuilder,
        GraphNormalizer,
    )
    from .graph import (  # noqa: F401
        GraphPairScorer,
        GraphValidationError,
        GraphWarning,
        HSABDescriptor,
        HuckelParameters,
        HuckelSystemResult,
    )
    from .graph import (  # noqa: F401
        ITSGraph,
        ITSGraphBuilder,
        NormalizedMolecularGraph,
        OrbitalNode,
        PairScoreGroup,
        PairScoreResult,
    )
    from .graph import (  # noqa: F401
        PairScoringConfig,
        PiAssignmentResult,
        PiAtom,
        PiSystem,
        PiSystemAssigner,
        assign_orbital_pi,
    )
    from .graph import (  # noqa: F401
        assign_pi_systems,
        directional_fmo,
        hsab_descriptor,
        local_hsab_compatibility,
        normalize_graph,
        solve_generalized_huckel,
    )
    from .integration import GraphXTBCascade, SFEnergy  # noqa: F401

__all__ = [
    "__version__",
    "CalibratedPrediction",
    "CalibrationRecord",
    "ComponentFrontier",
    "DirectionalFMO",
    "EnergyRefinementRecord",
    "EnergyRefinementReport",
    "FROZEN_MODEL_SHA256",
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
    "GeneralizedHuckel",
    "GeneralizedHuckelResult",
    "GeometryScorer",
    "GraphBuilder",
    "GraphEnergy",
    "GraphNormalizer",
    "GraphPairScorer",
    "GraphTheoryEnergyConfig",
    "GraphTheoryEnergyScorer",
    "GraphValidationError",
    "GraphWarning",
    "GraphXTBCascade",
    "HSABDescriptor",
    "HuckelParameters",
    "HuckelSystemResult",
    "ITSGraph",
    "ITSGraphBuilder",
    "ITSScoreResult",
    "ITSScorer",
    "ITSScoringConfig",
    "ModelCard",
    "MoleculeScoreResult",
    "MoleculeScorer",
    "MoleculeScoringConfig",
    "NormalizedMolecularGraph",
    "OrbitalNode",
    "OrbitalTheoryEnergyScorer",
    "PairScoreGroup",
    "PairScoreResult",
    "PairScoringConfig",
    "PiAssignmentResult",
    "PiAtom",
    "PiSystem",
    "PiSystemAssigner",
    "ReactionScoreResult",
    "ReactionScorer",
    "ReactionScoringConfig",
    "RidgeCalibrator",
    "SFEnergy",
    "SynDEDomainError",
    "SynDEEnergyModelCard",
    "SynDEEnergyPrediction",
    "SynDEEnergyPredictor",
    "SynDEEnergyRanking",
    "SynDEEnergyRefiner",
    "SynDEError",
    "SynDEInputError",
    "SynDEModelCard",
    "SynDEScorer",
    "SynDEValidationRecord",
    "TWO_D_FEATURE_SCHEMA",
    "TheoryEnergyConfig",
    "TheoryEnergyScorer",
    "TwoDEnergyConfig",
    "TwoDFeatureConfig",
    "ValenceEnergyConfig",
    "ValenceEnergyScorer",
    "ValenceEnergyWeights",
    "assign_orbital_pi",
    "assign_pi_systems",
    "deterministic_split",
    "directional_fmo",
    "extract_heuristic_features",
    "extract_two_d_features",
    "hsab_descriptor",
    "local_hsab_compatibility",
    "molecular_composition",
    "molecular_formula_charge_key",
    "normalize_graph",
    "solve_generalized_huckel",
    "theory_energy_corrections",
    "two_d_energy_components",
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
