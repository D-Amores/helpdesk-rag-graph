import streamlit as st


def _handle_human_response(ticket_id: str, ticket_data: dict):
    """Handle human intervention for escalated tickets."""

    result = ticket_data["result"]
    config = ticket_data["config"]

    st.warning("👨‍💼 Requires human intervention")

    # Contexto para el agente
    if result.get("response_rag"):
        with st.expander("📋 Context for agent"):
            st.text(result["response_rag"])

    response_human = st.text_area(
        "✍️ Agent response:",
        key=f"response_{ticket_id}",
        height=100,
        placeholder="Write the response for the user...",
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Send Response", key=f"btn_{ticket_id}"):
            if response_human.strip():
                _resume_graph(ticket_id, ticket_data, response_human)
            else:
                st.warning("⚠️ Write a response before sending")

    with col2:
        if st.button("🔄 Use RAG Response", key=f"rag_{ticket_id}"):
            rag_response = result.get("response_rag", "")
            _resume_graph(ticket_id, ticket_data, rag_response)


def _resume_graph(ticket_id: str, ticket_data: dict, human_response: str):
    """Resume the graph with the human response."""

    config = ticket_data["config"]

    # Actualiza el estado con la respuesta humana
    st.session_state.helpdesk.update_state(config, {"response_human": human_response})

    # Retoma el grafo desde donde pausó
    for chunk in st.session_state.helpdesk.stream(
        None, config=config, stream_mode="updates"
    ):
        for node, output in chunk.items():
            if "history" in output and output["history"]:
                ticket_data["history"].extend(output["history"])

    # Obtiene el estado final actualizado
    final_state = st.session_state.helpdesk.get_state(config)
    ticket_data["result"] = final_state.values

    st.success("✅ Response processed")
    st.rerun()


def _render_resolved_ticket(result: dict):
    """Render a resolved ticket."""

    st.success("✅ Ticket Resolved")
    st.markdown("**💬 Response:**")
    st.info(result["response_final"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Confidence", f"{result.get('confidence', 0):.2f}")
    with col2:
        st.metric("🔍 Sources", len(result.get("sources", [])))
    with col3:
        resolved_by = "RAG" if not result.get("required_human") else "Human"
        st.metric("🤖 Resolved by", resolved_by)


def render_ticket_panel():
    """Render the tickets panel."""

    st.subheader("🎫 Recent Tickets")

    if not st.session_state.tickets:
        st.info("No active tickets")
        return

    for ticket_id, ticket_data in reversed(list(st.session_state.tickets.items())):
        with st.expander(f"🎫 {ticket_id} - {ticket_data['timestamp']}", expanded=True):

            st.markdown(f"**👤 User:** {ticket_data['user']}")
            st.markdown(f"**💬 Query:** {ticket_data['consultation'][:100]}...")

            # Historial de procesamiento
            st.subheader("🔄 Processing:")
            for step in ticket_data["history"]:
                st.text(step)

            result = ticket_data["result"]

            # Confianza y fuentes
            if result.get("confidence", 0) > 0:
                st.markdown(f"**🎯 RAG Confidence:** {result['confidence']:.2f}")
                st.progress(result["confidence"])

                if result.get("sources"):
                    st.markdown(f"**📚 Sources:** {', '.join(result['sources'])}")

            # ¿Necesita humano o ya está resuelto?
            if result.get("required_human") and not result.get("response_final"):
                _handle_human_response(ticket_id, ticket_data)
            elif result.get("response_final"):
                _render_resolved_ticket(result)
