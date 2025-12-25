"""Streamlit application entry point."""
import streamlit as st
from main import TrafficIntelligenceSystem

# Page config
st.set_page_config(
    page_title="Traffic Intelligence System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize system in session state
if 'system' not in st.session_state:
    st.session_state.system = TrafficIntelligenceSystem()

# Run the app
st.session_state.system.run_streamlit_app()

