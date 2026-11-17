import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "colmap_reconstruction_colab.ipynb"
)


def test_colab_notebook_is_clean_valid_and_all_code_cells_parse():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 10
    for index, cell in enumerate(notebook["cells"]):
        assert cell["cell_type"] in {"markdown", "code"}
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]), filename=f"notebook_cell_{index}")


def test_colab_notebook_has_persistence_gates_and_objective_outputs():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    required = {
        "run_sparse_colmap",
        "PERSISTENT_RUN_ROOT",
        "frames_manifest.csv",
        "TRY_CONSERVATIVE_FALLBACK = False",
        "PROCESS_REMAINING_AFTER_PILOT = False",
        "registered_image_ratio",
        "mean_reprojection_error_pixels",
        "mean_track_length_images",
        "camera_poses.csv",
        "sparse_points_and_cameras.ply",
        "sparse_preview.png",
        "completeness_claimed",
        "gpu_execution_verified",
    }
    assert required <= {token for token in required if token in text}
    assert "download CO3D" in text
    assert "dense reconstruction" in text

