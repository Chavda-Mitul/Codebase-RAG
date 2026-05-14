"""Parse Python source files with tree-sitter to extract structural entities."""
from __future__ import annotations
import hashlib
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

from src.graph.schema import (
    FileNode, ClassNode, FunctionNode, ModuleNode,
    ContainsRel, ImportsRel, DefinesRel, InheritsRel, CallsRel,
    BaseNode, BaseRelationship,
)

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)


def _id(kind: str, *parts: str) -> str:
    raw = f"{kind}:{'|'.join(parts)}"
    return hashlib.md5(raw.encode()).hexdigest()[:16] + f"_{kind}"


def _get_text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_docstring(body_node: Node, source: bytes) -> str:
    """Extract docstring from a class/function body if first statement is a string."""
    if not body_node:
        return ""
    for child in body_node.children:
        if child.type == "expression_statement":
            for sub in child.children:
                if sub.type == "string":
                    raw = _get_text(sub, source)
                    return raw.strip("\"'").strip()
    return ""


def _extract_imports(tree_root: Node, source: bytes, file_id: str) -> tuple[list[ModuleNode], list[ImportsRel]]:
    modules: list[ModuleNode] = []
    rels: list[ImportsRel] = []

    for node in _walk(tree_root):
        if node.type in ("import_statement", "import_from_statement"):
            for child in node.children:
                if child.type in ("dotted_name", "aliased_import"):
                    name = _get_text(child, source).split(" as ")[0].strip()
                    if not name:
                        continue
                    mod_id = _id("Module", name)
                    modules.append(ModuleNode(id=mod_id, name=name))
                    rels.append(ImportsRel(source_id=file_id, target_id=mod_id))
    return modules, rels


def _extract_classes(
    tree_root: Node,
    source: bytes,
    file_id: str,
    rel_path: str,
) -> tuple[list[ClassNode], list[ContainsRel], list[InheritsRel]]:
    classes: list[ClassNode] = []
    contains: list[ContainsRel] = []
    inherits: list[InheritsRel] = []

    for node in _walk(tree_root):
        if node.type != "class_definition":
            continue

        name_node = node.child_by_field_name("name")
        if not name_node:
            continue
        name = _get_text(name_node, source)
        class_id = _id("Class", rel_path, name)

        body = node.child_by_field_name("body")
        docstring = _get_docstring(body, source) if body else ""

        # Base classes
        base_classes: list[str] = []
        for child in node.children:
            if child.type == "argument_list":
                for arg in child.children:
                    if arg.type in ("identifier", "attribute"):
                        base_name = _get_text(arg, source)
                        base_classes.append(base_name)
                        base_id = _id("Class", base_name)
                        inherits.append(InheritsRel(source_id=class_id, target_id=base_id))

        classes.append(ClassNode(
            id=class_id,
            name=name,
            file_path=rel_path,
            docstring=docstring,
            base_classes=base_classes,
        ))
        contains.append(ContainsRel(source_id=file_id, target_id=class_id))

    return classes, contains, inherits


def _extract_functions(
    tree_root: Node,
    source: bytes,
    file_id: str,
    rel_path: str,
    class_map: dict[str, str],  # name → id
) -> tuple[list[FunctionNode], list[ContainsRel | DefinesRel]]:
    functions: list[FunctionNode] = []
    rels: list[ContainsRel | DefinesRel] = []

    for node in _walk(tree_root):
        if node.type != "function_definition":
            continue

        name_node = node.child_by_field_name("name")
        if not name_node:
            continue
        name = _get_text(name_node, source)

        params_node = node.child_by_field_name("parameters")
        signature = f"{name}{_get_text(params_node, source)}" if params_node else name

        body = node.child_by_field_name("body")
        docstring = _get_docstring(body, source) if body else ""

        func_id = _id("Function", rel_path, name, str(node.start_point[0]))

        # Determine parent: class or file
        parent = node.parent
        class_name = ""
        if parent and parent.type == "block":
            grandparent = parent.parent
            if grandparent and grandparent.type == "class_definition":
                cn_node = grandparent.child_by_field_name("name")
                if cn_node:
                    class_name = _get_text(cn_node, source)

        functions.append(FunctionNode(
            id=func_id,
            name=name,
            file_path=rel_path,
            class_name=class_name,
            docstring=docstring,
            signature=signature,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        ))

        if class_name and class_name in class_map:
            rels.append(DefinesRel(source_id=class_map[class_name], target_id=func_id))
        else:
            rels.append(ContainsRel(source_id=file_id, target_id=func_id))

    return functions, rels


def _walk(node: Node):
    yield node
    for child in node.children:
        yield from _walk(child)


def parse_python_file(
    source_code: str,
    rel_path: str,
    repo: str = "",
) -> tuple[list[BaseNode], list[BaseRelationship]]:
    """Parse a Python file and return all nodes + relationships."""
    source = source_code.encode("utf-8")
    tree = _parser.parse(source)
    root = tree.root_node

    file_id = _id("File", rel_path)
    file_node = FileNode(id=file_id, path=rel_path, language="python", repo=repo)

    modules, import_rels = _extract_imports(root, source, file_id)
    classes, class_contains, inherits_rels = _extract_classes(root, source, file_id, rel_path)
    class_map = {c.name: c.id for c in classes}
    functions, func_rels = _extract_functions(root, source, file_id, rel_path, class_map)

    nodes: list[BaseNode] = [file_node, *modules, *classes, *functions]
    rels: list[BaseRelationship] = [*import_rels, *class_contains, *inherits_rels, *func_rels]

    return nodes, rels
