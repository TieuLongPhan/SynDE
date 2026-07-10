from dataclasses import dataclass


@dataclass
class SynParams:
    """
    Parameter bundle for Hückel + substituent effects and scoring.
    Default values are tuned as in your original SynParams.
    """

    beta: float = -1.0
    min_gap: float = 1e-8

    strength_plusM: float = 0.20
    strength_minusM: float = 0.22
    strength_plusH: float = 0.10
    strength_plusI: float = 0.00
    strength_minusI: float = 0.12

    att_M: float = 0.70
    att_H: float = 0.75
    att_I: float = 0.50

    use_extended: bool = True
    w_front: float = 1.0
    w_front_ext: float = 0.45

    w_coul_pi: float = 0.60
    w_coul_pc: float = 0.15
    w_steric: float = 0.25
    eps: float = 4.0
    rC: float = 1.4
    steric_k: float = 0.05

    auto_detect_effects: bool = True
    allow_sigma_fallback: bool = False
    detect_carbonyl_alpha: bool = True
    detect_halogen_I: bool = True
    hetero_plusM: bool = True
    sp3_plusH: bool = True

    enable_da_channels: bool = True
    top_pairs: int = 50
    top_da_channels: int = 4
