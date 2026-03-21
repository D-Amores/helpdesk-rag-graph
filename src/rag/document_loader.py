import hashlib
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from config import DOCS_PATH


class DocumentLoader:
    """Handles loading and enriching .md documents from the docs folder."""

    def __init__(self, docs_path: str = DOCS_PATH):
        self.docs_path = Path(docs_path)

    def load(self):
        """Load all .md files from the docs directory."""
        print(f"📚 Loading documents from {self.docs_path}")

        loader = DirectoryLoader(
            str(self.docs_path),
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )

        documents = loader.load()
        documents = self._enrich_metadata(documents)

        print(f"✅ Loaded {len(documents)} documents")
        return documents

    def _enrich_metadata(self, documents: List[Document]) -> List[Document]:
        """Adds extra information to each document."""

        for doc in documents:
            filename = Path(doc.metadata["source"]).stem
            doc.metadata.update(
                {
                    "filename": filename,
                    "doc_type": self._get_doc_type(filename),
                    "doc_id": self._generate_doc_id(doc),
                }
            )
        return documents

    def _get_doc_type(self, filename: str) -> str:
        """Classifies the document based on its filename."""
        filename = filename.lower()
        if "faq" in filename:
            return "faq"
        elif "manual" in filename:
            return "manual"
        elif "troubleshooting" in filename:
            return "troubleshooting"
        else:
            return "general"

    def _generate_doc_id(self, content: str) -> str:
        """Generates a unique ID for the document."""
        return hashlib.md5(content.encode()).hexdigest()[:8]
