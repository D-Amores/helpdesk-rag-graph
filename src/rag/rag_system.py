# src/rag/rag_system.py
from pathlib import Path
import logging
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

from config import EMBEDDINGS_MODEL, DEEPSEEK_API_KEY, CHROMADB_PATH


class VectorRAGSystem:
    """Advanced RAG system with ChromaDB and MultiQueryRetriever."""

    def __init__(self, chroma_path: str = CHROMADB_PATH):
        self.chroma_path = Path(chroma_path)
        self.embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
        self.llm = ChatDeepSeek(
            model="deepseek-chat", api_key=DEEPSEEK_API_KEY, temperature=0
        )
        self.vectorstore = None
        self.retriever = None

        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        self._load_vectorstore()

    def _load_vectorstore(self):
        """Loads ChromaDB and creates MultiQueryRetriever."""
        try:
            if not self.chroma_path.exists():
                print(f"⚠️ Vector store not found at {self.chroma_path}")
                return

            self.vectorstore = Chroma(
                persist_directory=str(self.chroma_path),
                embedding_function=self.embeddings,
                collection_name="helpdesk_knowledge",
            )

            self.retriever = MultiQueryRetriever.from_llm(
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4},
                ),
                llm=self.llm,
                prompt=self._get_multi_query_prompt(),
            )

            print("✅ VectorRAGSystem initialized successfully")

        except Exception as e:
            print(f"❌ Error loading vector store: {str(e)}")
            self.vectorstore = None
            self.retriever = None

    def _get_multi_query_prompt(self):
        """Custom prompt for MultiQueryRetriever."""
        return ChatPromptTemplate.from_template(
            """You are an expert helpdesk assistant. Your task is to generate multiple 
versions of the user's query to retrieve relevant documents from a 
technical support knowledge base.

Generate 3 different versions of the original query, considering:
- Technical synonyms
- Different ways to express the same problem
- Variations in helpdesk terminology

Original query: {question}

Alternative versions:"""
        )

    def search(self, query: str) -> Dict[str, Any]:
        """Searches for answers using MultiQueryRetriever."""
        if not self.retriever:
            return {
                "response": "RAG system not available. Please check configuration.",
                "confidence": 0.0,
                "sources": [],
            }

        try:
            # Search relevant documents with MultiQueryRetriever
            documents = self.retriever.invoke(query)

            if not documents:
                return {
                    "response": "No relevant information found in the knowledge base.",
                    "confidence": 0.1,
                    "sources": [],
                }

            # Extract context and sources
            context_parts = []
            sources = []

            for i, doc in enumerate(documents[:3]):
                content = doc.page_content.strip()
                if content:
                    context_parts.append(f"Document {i+1}: {content}")
                    filename = doc.metadata.get("filename", f"doc_{i+1}")
                    if filename not in sources:
                        sources.append(filename)

            if not context_parts:
                return {
                    "response": "Documents found but no useful content.",
                    "confidence": 0.2,
                    "sources": sources,
                }

            # Generate response using found context
            context = "\n\n".join(context_parts)
            response = self._generate_response(query, context)

            # Calculate confidence based on relevance
            confidence = self._calculate_confidence(query, documents)

            return {"response": response, "confidence": confidence, "sources": sources}

        except Exception as e:
            print(f"❌ Error in RAG search: {str(e)}")
            return {
                "response": f"Internal search error: {str(e)}",
                "confidence": 0.0,
                "sources": [],
            }

    def _generate_response(self, query: str, context: str) -> str:
        """Generates a response based on the found context."""
        prompt = ChatPromptTemplate.from_template(
            """You are an expert helpdesk assistant. Answer the user's query 
based ONLY on the context provided from the knowledge base.

Instructions:
- Provide a clear, direct and useful response
- If the context does not contain enough information, say so clearly
- Maintain a professional but friendly tone
- Do not invent information that is not in the context

Knowledge base context:
{context}

User query: {query}

Response:"""
        )

        try:
            response = self.llm.invoke(prompt.format(query=query, context=context))
            return response.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _calculate_confidence(self, query: str, documents: List) -> float:
        """Calculates confidence based on document relevance."""
        if not documents:
            return 0.0

        num_docs = len(documents)
        query_words = set(query.lower().split())

        relevance_score = 0
        total_content = 0

        for doc in documents[:3]:
            content = doc.page_content.lower()
            total_content += len(content.split())

            # Count how many query words appear in the chunk
            matches = sum(
                1 for word in query_words if word in content and len(word) > 2
            )
            relevance_score += matches

        if query_words and total_content > 0:
            # Base: how relevant is the content
            base_confidence = min(relevance_score / len(query_words), 1.0)

            # Bonus for number of documents found
            docs_bonus = min(num_docs / 4.0, 0.2)

            # Bonus for amount of content
            content_bonus = min(total_content / 1000.0, 0.1)

            final_confidence = min(base_confidence + docs_bonus + content_bonus, 1.0)
            return round(final_confidence, 2)

        return 0.3
