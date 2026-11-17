"""Configuration for the deterministic heuristic pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    """Resource bounds and declared heuristic thresholds.

    The values are engineering defaults for an MVP, not learned or statistically
    calibrated parameters. They are kept in one immutable object so reports and
    tests can state exactly which configuration was used.
    """

    max_samples: int = 32
    max_analysis_side: int = 640
    max_decode_frames_fallback: int = 20_000
    orb_features: int = 1_500
    orb_fast_threshold: int = 12
    ratio_test: float = 0.75
    ransac_reprojection_threshold_px: float = 2.0
    good_score_threshold: float = 70.0
    review_score_threshold: float = 45.0

    def validate(self) -> None:
        if not 2 <= self.max_samples <= 120:
            raise ValueError("max_samples must be between 2 and 120")
        if not 160 <= self.max_analysis_side <= 1_920:
            raise ValueError("max_analysis_side must be between 160 and 1920")
        if self.max_decode_frames_fallback < self.max_samples:
            raise ValueError("max_decode_frames_fallback must be >= max_samples")
        if self.orb_features < 100:
            raise ValueError("orb_features must be >= 100")
        if not 0.5 <= self.ratio_test < 1.0:
            raise ValueError("ratio_test must be in [0.5, 1.0)")

