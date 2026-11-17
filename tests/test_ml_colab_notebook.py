import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ml_colab_notebook_is_clean_parseable_and_label_gated():
    path = ROOT / "notebooks" / "ml_training_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 10
    full_source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    assert "ALLOW_SYNTHETIC_SMOKE = False" in full_source
    assert "require_targets=True" in full_source
    assert "EmbeddingCache" in full_source
    assert "Google Drive" in full_source
    assert "heuristic" in full_source.lower()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]), filename=f"cell_{index}")

