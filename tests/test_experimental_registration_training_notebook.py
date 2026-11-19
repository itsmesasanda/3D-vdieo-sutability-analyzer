import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/experimental_registration_training_colab.ipynb"
CONTRACT = (
    ROOT
    / "evaluation/ml/experimental_registration_training/regeneration_contract_template.json"
)


def _notebook_text():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return notebook, "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_experimental_training_notebook_is_clean_and_code_cells_parse():
    notebook, _ = _notebook_text()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "CPU"
    assert len(notebook["cells"]) >= 12
    for index, cell in enumerate(notebook["cells"]):
        assert cell["cell_type"] in {"markdown", "code"}
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]), filename=f"training_notebook_cell_{index}")


def test_notebook_has_evidence_group_cv_holdout_and_deployment_gates():
    _, text = _notebook_text()
    required = {
        "EXPECTED_PROVISIONAL_CANDIDATES = 647",
        'classification"] == "recoverable_success_missing_evidence"',
        "verify_registration_frame_manifest",
        "extract_registration_features_from_frame_manifest",
        "GroupKFold",
        '"dummy_mean"',
        '"ridge"',
        '"random_forest"',
        '"gradient_boosting"',
        "errors_by_degradation_type.csv",
        "independent_holdout_evaluation.json",
        "holdout_used_for_selection\": False",
        "REQUEST_DEPLOYMENT_PACKAGING = False",
        "PROVENANCE_COMPLETE = False",
        "verify_training_feature_parity",
        "package_registration_rf.py",
        "experimental_training_manifest.json",
    }
    assert required <= {value for value in required if value in text}
    assert "run_sparse_colmap" not in text
    assert "COLMAP mapper" not in text
    assert "models/registration_rf_current" in text


def test_regeneration_contract_template_is_nonpermissive():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema_version"] == "registration_regeneration_contract/1.0"
    assert contract["executed_code_evidence"]
    for variant in contract["variants"].values():
        assert variant["sampling_fully_specified"] is False
        assert variant["transformation_fully_specified"] is False
        assert variant["deterministic"] is False
        assert variant["fidelity_tier"] == "insufficient_evidence"
