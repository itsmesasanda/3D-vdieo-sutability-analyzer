"""Interface template for exact 60-frame regeneration from executed labeling code.

Copy the relevant functions from the actually executed notebook/script into a new,
versioned adapter. Do not implement transformations from variant names or this
template. Hash the completed adapter into the regeneration contract.
"""

from pathlib import Path
from typing import Any, Dict, Mapping


def generate_experiment(
    *,
    record: Mapping[str, Any],
    co3d_root: Path,
    output_dir: Path,
    variant_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """Generate exactly the labeled 60 inputs and return their evidence paths.

    The completed implementation must:

    - use the authoritative CO3D frame ordering from the executed workflow;
    - apply the exact recorded temporal selection and image transformation order;
    - use recorded random seeds and exact runtime-dependent encoder settings;
    - write a 60-row manifest accepted by
      ``verify_registration_frame_manifest`` with SHA-256 for every output;
    - set ``original_hashes_verified`` true only after comparing every generated
      file with hashes recorded during labeling.

    Return keys:

    ``frames_manifest``: path to the 60-row manifest
    ``images_directory``: directory containing exactly those files
    ``variant``: the reproduced variant name
    ``source_group``: the physical source identifier
    ``original_hashes_verified``: boolean
    """

    raise NotImplementedError(
        "Replace this template only with implementation recovered from executed "
        "labeling code and configuration; never infer missing settings."
    )
