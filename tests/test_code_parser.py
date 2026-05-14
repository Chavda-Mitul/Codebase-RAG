"""Unit tests for the tree-sitter code parser."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.parsers.code_parser import parse_python_file
from src.graph.schema import FileNode, ClassNode, FunctionNode, ModuleNode


FIXTURE = '''
import os
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Detects anomalies in invoice data using Isolation Forest."""

    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(contamination=contamination)

    def fit(self, X):
        """Fit the model on training data."""
        self.model.fit(X)
        return self

    def predict(self, X):
        """Return -1 for anomalies, 1 for normal."""
        return self.model.predict(X)


def load_data(path: str):
    """Load invoice data from CSV."""
    return pd.read_csv(path)
'''


def test_extracts_file_node():
    nodes, _ = parse_python_file(FIXTURE, "detector.py")
    file_nodes = [n for n in nodes if isinstance(n, FileNode)]
    assert len(file_nodes) == 1
    assert file_nodes[0].path == "detector.py"
    assert file_nodes[0].language == "python"


def test_extracts_class_node():
    nodes, _ = parse_python_file(FIXTURE, "detector.py")
    class_nodes = [n for n in nodes if isinstance(n, ClassNode)]
    assert len(class_nodes) == 1
    assert class_nodes[0].name == "AnomalyDetector"
    assert "Detects anomalies" in class_nodes[0].docstring


def test_extracts_functions():
    nodes, _ = parse_python_file(FIXTURE, "detector.py")
    func_nodes = [n for n in nodes if isinstance(n, FunctionNode)]
    names = {f.name for f in func_nodes}
    assert "fit" in names
    assert "predict" in names
    assert "load_data" in names
    assert "__init__" in names


def test_extracts_imports():
    nodes, _ = parse_python_file(FIXTURE, "detector.py")
    module_nodes = [n for n in nodes if isinstance(n, ModuleNode)]
    names = {m.name for m in module_nodes}
    assert "os" in names
    assert "pandas" in names or "pd" in names or any("pandas" in n for n in names)


def test_extracts_relationships():
    _, rels = parse_python_file(FIXTURE, "detector.py")
    rel_types = {r.type for r in rels}
    assert "CONTAINS" in rel_types
    assert "IMPORTS" in rel_types
    assert "DEFINES" in rel_types


def test_method_links_to_class():
    nodes, rels = parse_python_file(FIXTURE, "detector.py")
    class_nodes = [n for n in nodes if isinstance(n, ClassNode)]
    func_nodes = [n for n in nodes if isinstance(n, FunctionNode) and n.class_name == "AnomalyDetector"]
    defines_rels = [r for r in rels if r.type == "DEFINES"]

    class_id = class_nodes[0].id
    defined_ids = {r.target_id for r in defines_rels if r.source_id == class_id}
    method_ids = {f.id for f in func_nodes}
    assert defined_ids & method_ids, "Class should DEFINES its methods"
