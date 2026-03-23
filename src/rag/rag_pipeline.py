# src/rag/rag_pipeline.py
from langchain_chroma import Chroma

from rag.document_loader import DocumentLoader
from rag.text_splitter import TextSplitter
from rag.vector_store import VectorStore
from rag.retriever import Retriever


class RAGPipeline:
    """Orchestrates the complete RAG system."""

    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = TextSplitter()
        self.vector_store = VectorStore()

    def build(self, force_rebuild: bool = False) -> Chroma:
        """Builds the complete RAG pipeline."""
        print("🚀 Setting up RAG system...")

        if not force_rebuild:
            try:
                vectorstore = self.vector_store.load()
                print("✅ Existing vector store loaded")
                return vectorstore
            except FileNotFoundError:
                print("⚠️  No existing vector store found, building...")

        documents = self.loader.load()
        if not documents:
            print("⚠️  No documents found")
            return None

        chunks = self.splitter.split(documents)

        vectorstore = self.vector_store.create(chunks)

        print("✅ RAG system ready!")
        return vectorstore

    def get_retriever(self, vectorstore: Chroma, k: int = 3) -> Retriever:
        """Returns a retriever ready to search."""
        return Retriever(vectorstore=vectorstore, k=k)
