from graph.state import HelpdeskState
from rag.rag_system import VectorRAGSystem
from config import CHROMADB_PATH


class RagNode:
    """Node responsible for searching in the knowledge base."""

    def __init__(self):
        self.rag = VectorRAGSystem(chroma_path=CHROMADB_PATH)

    def process_rag(self, state: HelpdeskState):
        """Search in the knowledge base using the consultation."""
        consultation = state["consultation"]
        result = self.rag.search(consultation)

        return {
            "response_rag": result["response"],
            "confidence": result["confidence"],
            "sources": result["sources"],
            "context_rag": result["response"],
            "history": [
                "Executed RAG with MultiQueryRetriever",
                f"Confidence: {result['confidence']}",
                f"Sources: {result['sources']}",
            ],
        }
