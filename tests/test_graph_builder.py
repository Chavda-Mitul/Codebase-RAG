"""Unit tests for graph schema and builder helpers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.schema import (
    FileNode, ClassNode, FunctionNode, ModuleNode,
    ContainsRel, ImportsRel,
)
from src.graph.builder import _node_text


def test_file_node_serialization():
    node = FileNode(id="abc", path="src/main.py", language="python", repo="myrepo")
    data = node.model_dump()
    assert data["path"] == "src/main.py"
    assert data["label"] == "File"
    assert data["embedding"] is None


def test_class_node_text():
    node = ClassNode(
        id="xyz",
        name="InvoiceProcessor",
        file_path="src/processor.py",
        docstring="Handles invoice processing",
    )
    text = _node_text(node)
    assert "InvoiceProcessor" in text
    assert "Handles invoice processing" in text


def test_relationship_model():
    rel = ContainsRel(source_id="file_id", target_id="class_id")
    assert rel.type == "CONTAINS"
    assert rel.source_id == "file_id"
    assert rel.target_id == "class_id"


def test_imports_relationship():
    rel = ImportsRel(source_id="file_id", target_id="mod_id")
    assert rel.type == "IMPORTS"


def test_function_node_has_line_numbers():
    node = FunctionNode(
        id="fn1",
        name="detect",
        file_path="src/model.py",
        start_line=10,
        end_line=25,
    )
    assert node.start_line == 10
    assert node.end_line == 25
