# src/graph/nodes/response_node.py
from graph.state import HelpdeskState


class ResponseNode:
    """Node responsible for generating the final response."""

    def generate_final_response(self, state: HelpdeskState):
        """Generate the final response to the user."""
        if state.get("response_final"):
            return {"history": ["Response already provided by human"]}

        response_rag = state.get("response_rag", "")
        sources = state.get("sources", [])

        final_response = response_rag
        if sources:
            text_sources = ", ".join(sources)
            final_response += f"\n\nSources: {text_sources}"

        return {
            "response_final": final_response,
            "history": ["Final response generated"],
        }
