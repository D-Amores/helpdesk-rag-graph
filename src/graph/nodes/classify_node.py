from langchain_core.prompts import ChatPromptTemplate
from graph.state import HelpdeskState
from core.models import Classification
from llm.deepseek import get_llm


class ClassifyNode:
    """Node responsible for classifying the query."""

    def __init__(self):
        self.llm = get_llm(temperature=0.0)

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
