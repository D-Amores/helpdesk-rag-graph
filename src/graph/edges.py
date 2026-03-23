from graph.state import HelpdeskState


def decide_from_clasification(state: HelpdeskState) -> str:
    """Decide the next node based on the classification."""
    category = state.get("category", "escalated")

    if category == "automatic":
        return "response_final"
    return "escalated"


def decide_since_human(state: HelpdeskState) -> str:
    """Decide the next node based on the human response."""
    response_human = state.get("response_human", "")

    if response_human:
        return "process_human_response"
    return "waiting_human"
