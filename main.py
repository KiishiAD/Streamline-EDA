import streamlit as st
from streamlit_option_menu import option_menu
import pages.data_upload
import pages.dataset_overview
import pages.data_validation
import pages.data_cleaning
import pages.eda
import pages.settings
import pages.timeline 
from utils import initialize_session_state

def main():
    # Set Streamlit page configuration
    st.set_page_config(
        page_title=config.STREAMLIT_CONFIG["page_title"],
        layout=config.STREAMLIT_CONFIG["layout"],
        initial_sidebar_state=config.STREAMLIT_CONFIG["initial_sidebar_state"]
    )

    # Initialize session state variables
    initialize_session_state()

    # Sidebar menu for navigation
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=[
                "Upload Data",
                "Dataset Overview",
                "Data Validation",
                "Data Cleaning",
                "Exploratory Data Analysis",
                "Timeline",
                "Settings"
            ],
            icons=[
                "cloud-upload",        # Upload Data
                "bar-chart-line",      # Dataset Overview
                "shield-check",        # Data Validation
                "eraser-fill",         # Data Cleaning
                "activity",            # Exploratory Data Analysis
                "clock-history",       # Timeline
                "gear"                 # Settings
            ],
            menu_icon="cast",
            default_index=0,
        )

    # Navigation to selected page
    if selected == "Upload Data":
        pages.data_upload.app()
    elif selected == "Dataset Overview":
        pages.dataset_overview.app()
    elif selected == "Data Validation":
        pages.data_validation.app()
    elif selected == "Data Cleaning":
        pages.data_cleaning.app()
    elif selected == "Exploratory Data Analysis":
        pages.eda.app()
    elif selected == "Timeline":
        pages.timeline.app()
    elif selected == "Settings":
        pages.settings.app()

if __name__ == "__main__":
    main()
