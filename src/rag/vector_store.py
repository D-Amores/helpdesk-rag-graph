# src/rag/vector_store.py
import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMADB_PATH
from rag.embedder import EmbeddingProvider

COLLECTION_NAME = "helpdesk_knowledge"


class VectorStore:
    """Handles creating and loading the ChromaDB vector store."""

    def __init__(self, chroma_path: str = CHROMADB_PATH):
        self.chroma_path = Path(chroma_path)
        self.embeddings = EmbeddingProvider().get_embeddings()

    def create(self, chunks: List[Document]) -> Chroma:
        """Creates a new vector store from chunks."""
        print("🔄 Creating vector store in ChromaDB...")

        if self.chroma_path.exists():
            shutil.rmtree(self.chroma_path)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.chroma_path),
            collection_name=COLLECTION_NAME,
        )

        print(f"✅ Vector store created at {self.chroma_path}")
        print(f"📊 Total vectors: {len(chunks)}")
        return vectorstore

    def load(self) -> Chroma:
        """Loads an existing vector store."""
        if not self.chroma_path.exists():
            raise FileNotFoundError(f"Vector store not found at {self.chroma_path}")

        print("📦 Loading existing vector store...")

        vectorstore = Chroma(
            persist_directory=str(self.chroma_path),
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME,
        )

        return vectorstore
