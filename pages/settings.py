# pages/settings.py

import streamlit as st
import config
from utils import initialize_session_state

def app():
    st.title("⚙️ Settings")

    # Initialize session state variables
    initialize_session_state()

    st.header("App Configuration")
    page_title = st.text_input("Page Title", config.STREAMLIT_CONFIG["page_title"])
    layout = st.selectbox("Layout", ["centered", "wide"], index=["centered", "wide"].index(config.STREAMLIT_CONFIG["layout"]))
    sidebar_state = st.selectbox("Initial Sidebar State", ["auto", "expanded", "collapsed"],
                                 index=["auto", "expanded", "collapsed"].index(config.STREAMLIT_CONFIG["initial_sidebar_state"]))

    if st.button("Save Settings"):
        config.STREAMLIT_CONFIG["page_title"] = page_title
        config.STREAMLIT_CONFIG["layout"] = layout
        config.STREAMLIT_CONFIG["initial_sidebar_state"] = sidebar_state
        st.success("Settings saved. Please restart the app to apply changes.")

    st.header("🔄 Session Management")
    if st.button("Reset Session"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.success("Session state has been reset. Please re-upload your data.")
