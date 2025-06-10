import streamlit as st
import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Configure Streamlit to listen on all network interfaces
st.set_page_config(page_title="DocBot - Document Research & Theme ID", layout="wide")

# Set server address to 0.0.0.0 to allow external access
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
os.environ['STREAMLIT_SERVER_PORT'] = '8501'

# Import and run the UI code
import frontend.ui 