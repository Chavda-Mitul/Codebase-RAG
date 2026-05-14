from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class BaseNode(BaseModel):
    id: str
    embedding: Optional[list[float]] = None


class FileNode(BaseNode):
    label: str = "File"
    path: str
    language: str
    repo: str = ""


class ClassNode(BaseNode):
    label: str = "Class"
    name: str
    file_path: str
    docstring: str = ""
    base_classes: list[str] = Field(default_factory=list)


class FunctionNode(BaseNode):
    label: str = "Function"
    name: str
    file_path: str
    class_name: str = ""
    docstring: str = ""
    signature: str = ""
    start_line: int = 0
    end_line: int = 0


class ModuleNode(BaseNode):
    label: str = "Module"
    name: str


class ConceptNode(BaseNode):
    label: str = "Concept"
    name: str
    description: str = ""


class CommunityNode(BaseNode):
    label: str = "Community"
    name: str
    members: list[str] = Field(default_factory=list)  # member node IDs
    summary: str = ""


class BaseRelationship(BaseModel):
    source_id: str
    target_id: str
    type: str
    properties: dict = Field(default_factory=dict)


class ContainsRel(BaseRelationship):
    type: str = "CONTAINS"


class ImportsRel(BaseRelationship):
    type: str = "IMPORTS"


class CallsRel(BaseRelationship):
    type: str = "CALLS"


class InheritsRel(BaseRelationship):
    type: str = "INHERITS"


class DefinesRel(BaseRelationship):
    type: str = "DEFINES"


class DependsOnRel(BaseRelationship):
    type: str = "DEPENDS_ON"


class RelatedToRel(BaseRelationship):
    type: str = "RELATED_TO"
