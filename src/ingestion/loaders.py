import os
import tempfile
from pathlib import Path
from git import Repo
from langchain_core.documents import Document


_DEFAULT_EXTENSIONS = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml"}


def clone_repo(url: str, dest_dir: str | None = None) -> str:
    """Clone a GitHub repo and return the local path."""
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp(prefix="code-rag-")
    Repo.clone_from(url, dest_dir)
    return dest_dir


def discover_files(
    root: str,
    extensions: set[str] = _DEFAULT_EXTENSIONS,
) -> list[Document]:
    """Walk root directory and return Documents for matching files."""
    docs: list[Document] = []
    root_path = Path(root)

    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in extensions:
            continue
        # Skip hidden dirs and common noise
        parts = path.relative_to(root_path).parts
        if any(p.startswith(".") or p in ("__pycache__", "node_modules", ".git") for p in parts):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        language = _ext_to_language(path.suffix)
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "path": str(path.relative_to(root_path)),
                    "abs_path": str(path),
                    "language": language,
                    "extension": path.suffix,
                    "repo": root,
                },
            )
        )
    return docs


def _ext_to_language(ext: str) -> str:
    return {
        ".py": "python",
        ".md": "markdown",
        ".txt": "text",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
    }.get(ext, "unknown")
