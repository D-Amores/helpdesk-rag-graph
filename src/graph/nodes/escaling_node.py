# src/graph/nodes/escaling_node.py
from graph.state import HelpdeskState


class EscalingNode:
    """Node responsible for escalation to human."""

    def prepare_escaling(self, state: HelpdeskState):
        """Prepare the escalation message to the human."""
        return {
            "required_human": True,
            "history": ["Escalating to human - Wait for human response"],
        }

    def process_human_response(self, state: HelpdeskState):
        """Process the human response and update the state."""
        response_human = state.get("response_human", "")

        if response_human:
            return {
                "response_final": response_human,
                "history": ["Human response received and processed"],
            }

        return {"history": ["Waiting for human response"]}
