import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/r1_regeneration_pilot_colab.ipynb"
PILOT_SCRIPT = ROOT / "scripts/ml/run_r1_feature_pilot.py"
ADAPTER = ROOT / "scripts/ml/r1_regeneration_adapter.py"


def test_r1_pilot_notebook_is_clean_cpu_and_parseable():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["metadata"]["accelerator"] == "CPU"
    assert len(notebook["cells"]) >= 7
    assert sum(len(cell.get("outputs", [])) for cell in notebook["cells"]) == 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            ast.parse("".join(cell["source"]), filename=f"r1_pilot_cell_{index}")


def test_r1_pilot_uses_production_extractor_and_has_guarded_expansion():
    notebook_text = NOTEBOOK.read_text(encoding="utf-8")
    script_text = PILOT_SCRIPT.read_text(encoding="utf-8")
    adapter_text = ADAPTER.read_text(encoding="utf-8")
    assert "extract_registration_features_from_frame_manifest" in script_text
    assert "--all-r1-sparse" in script_text
    assert "EXPAND_ALL_R1 = False" in notebook_text
    assert "run_r1_feature_pilot.py" in notebook_text
    assert "regenerate_r1_experiment" in adapter_text
    assert "run_sparse_colmap" not in notebook_text + script_text + adapter_text
    assert "artifact_manifest.json" not in notebook_text + script_text + adapter_text
