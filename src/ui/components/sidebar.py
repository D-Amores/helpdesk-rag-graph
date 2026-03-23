import streamlit as st
from pathlib import Path
from config import CHROMADB_PATH
from rag.rag_pipeline import RAGPipeline


def render_sidebar():
    """Render the sidebar with control panel."""

    with st.sidebar:
        st.header("📊 Control Panel")

        # Métricas
        st.metric("Active Tickets", len(st.session_state.tickets))

        # Estado del RAG
        st.subheader("🔍 RAG Status")
        chroma_path = Path(CHROMADB_PATH)

        if chroma_path.exists():
            st.success("✅ ChromaDB configured")
        else:
            st.warning("⚠️ RAG not configured")
            if st.button("🚀 Configure RAG"):
                _setup_rag()

        # Instrucciones
        st.subheader("🔄 System Flow")
        st.text(
            """
            1. 📝 User sends query
            2. 🔍 RAG vector search
            3. 🤖 Automatic classification
            4. 📊 Confidence evaluation
            5. 👨‍💼 Escalation if needed
            6. ✅ Final response
        """
        )

        # Acciones
        st.subheader("⚙️ Settings")
        if st.button("🔄 Rebuild RAG"):
            _setup_rag(force_rebuild=True)

        if st.button("🗑️ Clear Tickets"):
            st.session_state.tickets = {}
            st.rerun()


def _setup_rag(force_rebuild: bool = False):
    """Configure or rebuild the RAG system."""
    with st.spinner("🔧 Configuring RAG system..."):
        pipeline = RAGPipeline()
        vectorstore = pipeline.build(force_rebuild=force_rebuild)

        if vectorstore:
            st.success("✅ RAG configured successfully")
            st.rerun()
        else:
            st.error("❌ Error configuring RAG")
