from langchain_deepseek import ChatDeepSeek
from rag.rag_system import VectorRAGSystem
from config import CHROMADB_PATH, DEEPSEEK_API_KEY
from graph.state import HelpdeskState
from langchain_core.prompts import ChatPromptTemplate
from core.models import Classification
from llm.deepseek import get_llm


class HelpdeskGraph:
    """Class to build the helpdesk graph."""

    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        self.rag = VectorRAGSystem(chroma_path=CHROMADB_PATH)
        self.graph = None

    def proccess_rag(self, state: HelpdeskState):
        """Search in the knowledge base using the consultation"""

        consultation = state["consultation"]
        result = self.rag.search(consultation)

        return {
            "response_rag": result["response"],
            "confidence": result["confidence"],
            "sources": result["sources"],
            "context_rag": result["response"],
            "history": [
                "executed RAG with MultiQueryRetriever",
                f"confidence: {result['confidence']}",
                f"sources: {result['sources']}",
            ],
        }

    def classify_with_context(self, state: HelpdeskState):
        """Classify the query with context using structured output."""

        consultation = state["consultation"]
        context_rag = state.get("context_rag", "")
        confidence = state.get("confidence", 0)

        prompt = ChatPromptTemplate.from_template(
            """
                Analyze this helpdesk query and decide if it can be resolved 
                automatically or needs escalation:

                USER QUERY: {consultation}
                KNOWLEDGE BASE INFO: {context_rag}
                SEARCH CONFIDENCE: {confidence}

                Criteria:
                - automatic: complete info, confidence > 0.6, standard topic
                - escalated: insufficient info, low confidence, complex problem
            """
        )

        try:
            structured_llm = self.llm.with_structured_output(Classification)

            result = structured_llm.invoke(
                prompt.format(
                    consultation=consultation,
                    context_rag=context_rag,
                    confidence=confidence,
                )
            )

            return {
                "category": result.category,
                "history": [
                    f"Classification: {result.category}",
                    f"Justification: {result.justification}",
                ],
            }

        except Exception as e:
            category = "automatic" if confidence > 0.6 else "escalated"
            return {
                "category": category,
                "history": [f"Error in classification, using confidence: {e}"],
            }

    def prepare_escaling(self, state: HelpdeskState):
        """Prepare the escalation message to the human"""

        return {
            "required_human": True,
            "history": ["Escalating to human - Wait for human response"],
        }

    def process_human_response(self, state: HelpdeskState):
        """Process the human response and update the state"""

        response_human = state.get("response_human", "")

        if response_human:
            return {
                "response_final": response_human,
                "history": ["Human response received and processed"],
            }

        return {"history": [" Waiting for human response"]}

    def generate_final_response(self, state: HelpdeskState):
        """Generate the final response to the user"""

        if state.get("response_final"):
            return {"history": ["Response already provided by human"]}

        respuesta_rag = state.get("response_rag", "")
        sources = state.get("sources", [])

        final_response = respuesta_rag
        if sources:
            text_sources = ", ".join(sources)
            final_response += f"\n\nSources: {text_sources}"

        return {
            "response_final": final_response,
            "history": ["Final response generated"],
        }
