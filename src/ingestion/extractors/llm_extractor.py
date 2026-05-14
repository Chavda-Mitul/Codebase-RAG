"""LLM-based semantic entity and relationship extraction."""
from __future__ import annotations
import hashlib
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from src.config import settings
from src.graph.schema import ConceptNode, RelatedToRel, BaseNode, BaseRelationship


class ExtractedConcept(BaseModel):
    name: str = Field(description="Short concept name, e.g. 'AnomalyDetection', 'DataPipeline'")
    description: str = Field(description="One sentence description")
    related_to: list[str] = Field(default_factory=list, description="Names of other concepts this relates to")


class ExtractionResult(BaseModel):
    concepts: list[ExtractedConcept] = Field(default_factory=list)


_SYSTEM_PROMPT = """You are a software architecture analyst. Given a code file, extract high-level architectural concepts and patterns present in it.

Extract concepts like: design patterns, architectural components, data flows, important abstractions, and business domain concepts.
Keep concept names concise (2-4 words). Only extract concepts clearly evidenced by the code.
"""

_USER_PROMPT = """File: {path}

```python
{code}
```

Extract architectural concepts from this file."""


def _concept_id(name: str) -> str:
    return hashlib.md5(name.lower().encode()).hexdigest()[:16] + "_concept"


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )


def extract_concepts_from_file(
    doc: Document,
    llm: ChatGroq | None = None,
) -> tuple[list[BaseNode], list[BaseRelationship]]:
    """Extract semantic concepts from a single file using the LLM."""
    if llm is None:
        llm = _get_llm()

    structured_llm = llm.with_structured_output(ExtractionResult)

    path = doc.metadata.get("path", "unknown")
    # Truncate large files to avoid token limits
    code = doc.page_content[:6000]

    try:
        result: ExtractionResult = structured_llm.invoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_PROMPT.format(path=path, code=code)},
            ]
        )
    except Exception as e:
        print(f"  [LLM] Failed on {path}: {e}")
        return [], []

    nodes: list[BaseNode] = []
    rels: list[BaseRelationship] = []

    for concept in result.concepts:
        cid = _concept_id(concept.name)
        nodes.append(ConceptNode(id=cid, name=concept.name, description=concept.description))

        for related_name in concept.related_to:
            related_id = _concept_id(related_name)
            rels.append(RelatedToRel(source_id=cid, target_id=related_id))

    return nodes, rels
