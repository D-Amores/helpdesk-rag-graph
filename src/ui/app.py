import streamlit as st
from ui.state import initialize_state
from ui.components.sidebar import render_sidebar
from ui.components.problem_panel import render_problem_panel
from ui.components.ticket_panel import render_ticket_panel


def main():
    """Main application entry point."""

    st.set_page_config(
        page_title="Helpdesk 2.0 with RAG", page_icon="🎧", layout="wide"
    )

    st.title("🎧 Helpdesk 2.0 with RAG + ChromaDB")
    st.markdown("*Intelligent system with LangGraph and vector search*")

    # 1. Inicializa la sesión
    initialize_state()

    # 2. Renderiza el sidebar
    render_sidebar()

    # 3. Área principal — dos columnas
    col1, col2 = st.columns([1, 1])

    with col1:
        render_problem_panel()

    with col2:
        render_ticket_panel()

    # 4. Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <small>
                🚀 LangGraph | 🔍 ChromaDB | 
                💾 Checkpointing | 👨‍💼 Human-in-the-Loop
            </small>
        </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
