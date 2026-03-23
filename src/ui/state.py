import streamlit as st
from graph.workflow import create_helpdesk


def initialize_state():
    """Initialize session state variables"""

    if "helpdesk" not in st.session_state:
        st.session_state.helpdesk = create_helpdesk()

    if "tickets" not in st.session_state:
        st.session_state.tickets = {}
