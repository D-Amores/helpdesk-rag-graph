import uuid
import streamlit as st
from datetime import datetime
from graph.state import HelpdeskState


def _create_ticket_id() -> str:
    """Generate a unique ticket ID."""
    return f"TK-{uuid.uuid4().hex[:6].upper()}"


def _process_query(consultation: str, ticket_id: str):
    """Send the query to the LangGraph graph."""

    initial_state = HelpdeskState(
        consultation=consultation,
        category="",
        response_rag=None,
        confidence=0.0,
        sources=[],
        context_rag=None,
        required_human=False,
        response_human=None,
        response_final=None,
        history=[],
    )

    config = {"configurable": {"thread_id": ticket_id}}
    processing_history = []

    try:
        for chunk in st.session_state.helpdesk.stream(
            initial_state, config=config, stream_mode="updates"
        ):
            for node, output in chunk.items():
                if "history" in output and output["history"]:
                    processing_history.extend(output["history"])

        final_state = st.session_state.helpdesk.get_state(config)
        return final_state.values, processing_history, config

    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None, [], None


def render_problem_panel():
    """Render the problem submission panel."""

    st.subheader("📝 New Query")

    # Ejemplos de consultas
    with st.expander("💡 Query Examples"):
        examples = [
            "I cannot reset my password",
            "Error 500 in the application",
            "How do I cancel my subscription?",
            "The application is very slow",
            "Billing problems",
        ]
        for example in examples:
            if st.button(f"📋 {example}", key=f"ex_{example}"):
                st.session_state.example_query = example

    with st.form("new_query"):
        user = st.text_input("👤 User", placeholder="your@email.com")

        initial_query = st.session_state.get("example_query", "")
        consultation = st.text_area(
            "💬 Problem description",
            value=initial_query,
            placeholder="Describe your query or problem here...",
            height=100,
        )

        submitted = st.form_submit_button("🚀 Submit Query")

        if submitted and consultation.strip():
            if "example_query" in st.session_state:
                del st.session_state.example_query

            ticket_id = _create_ticket_id()

            with st.spinner("🔄 Processing query..."):
                result, history, config = _process_query(consultation, ticket_id)

            if result:
                st.session_state.tickets[ticket_id] = {
                    "user": user,
                    "consultation": consultation,
                    "result": result,
                    "history": history,
                    "config": config,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }

                st.success(f"✅ Ticket {ticket_id} created")
                st.rerun()
