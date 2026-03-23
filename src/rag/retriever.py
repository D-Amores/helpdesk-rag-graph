from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document


class Retriever:
    """Handles searching relevant chunks from ChromaDB."""

    def __init__(self, vectorstore: Chroma, k: int = 3):
        self.vectorstore = vectorstore
        self.k = k

    def search(self, query: str) -> List[Document]:
        """Searches for the most relevant chunks for a query."""
        print(f"🔍 Searching for: '{query}'")

        results = self.vectorstore.similarity_search(query, k=self.k)

        print(f"✅ Found {len(results)} relevant chunks")
        return results

    def format_results(self, results: List[Document]) -> str:
        """Formats results into readable text."""
        formatted = []

        for i, doc in enumerate(results, 1):
            chunk = (
                f"📄 Result {i}:\n"
                f"Type: {doc.metadata.get('doc_type', 'unknown')}\n"
                f"File: {doc.metadata.get('filename', 'unknown')}\n"
                f"Content: {doc.page_content[:200]}...\n"
            )
            formatted.append(chunk)

        return "\n".join(formatted)
