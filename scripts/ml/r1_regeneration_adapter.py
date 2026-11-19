"""Adapter used by the experimental training notebook for executed R1 variants."""

from pathlib import Path
from typing import Any, Dict, Mapping

from video_suitability.ml.r1_regeneration import regenerate_r1_experiment


def generate_experiment(
    *,
    record: Mapping[str, Any],
    co3d_root: Path,
    output_dir: Path,
    variant_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """Regenerate one R1 experiment using the extracted executed implementation."""

    if variant_contract.get("fidelity_tier") != "executed_contract_reproduced":
        raise ValueError(
            "R1 has no original prepared-image hashes; its fidelity tier must remain "
            "executed_contract_reproduced"
        )
    evidence = variant_contract.get("executed_notebook_evidence")
    if not isinstance(evidence, dict) or evidence.get("implementation_verified") is not True:
        raise ValueError("R1 variant contract lacks verified executed-notebook evidence")
    return regenerate_r1_experiment(
        record=record,
        co3d_root=co3d_root,
        output_dir=output_dir,
        executed_notebook_evidence=evidence,
    )
