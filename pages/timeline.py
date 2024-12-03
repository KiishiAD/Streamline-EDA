# pages/timeline.py

import streamlit as st
import pandas as pd
from utils import initialize_session_state, revert_to_state, clear_history

def app():
    st.title("🕒 Timeline")

    # Initialize session state variables
    initialize_session_state()

    if st.session_state['df'] is None:
        st.warning("No data available. Please upload and clean data first.")
        return

    st.write("### 📜 Data Modification Timeline")

    history = st.session_state.get('history', [])

    if not history:
        st.info("No timeline available. Perform some data cleaning or transformations to generate history.")
        return

    # Display the history with summaries and timestamps
    history_display = [
        {
            'index': idx,
            'summary': entry['summary'],
            'timestamp': entry['timestamp']
        }
        for idx, entry in enumerate(history, 1)
    ]

    # Convert to DataFrame for better display and interaction
    history_df = pd.DataFrame(history_display)

    # Reverse the DataFrame to show latest actions first
    history_df = history_df[::-1].reset_index(drop=True)

    # Display the timeline
    st.write("#### Action Summary and Timestamp")
    selected_state = st.selectbox(
        "Select a state to view or revert to:",
        options=history_df.apply(lambda row: f"State {row['index']}: {row['summary']}", axis=1),
        help="Choose a data state from the timeline to view or revert."
    )

    if selected_state:
        # Extract the index from the selected state string
        selected_index = int(selected_state.split(":")[0].split(" ")[1]) - 1
        entry = history[selected_index]
        df_selected = entry['df']
        summary = entry['summary']
        timestamp = entry['timestamp']

        st.write(f"### 📄 Viewing State {selected_index + 1}")
        st.write(f"**Summary:** {summary}")
        st.write(f"**Timestamp:** {timestamp}")
        st.dataframe(df_selected.head(10))
        st.write(f"**Data Types:**")
        st.write(df_selected.dtypes)

        # Option to revert to the selected state
        if st.button(f"Revert to State {selected_index + 1}"):
            revert_to_state(selected_index)
            st.success(f"Reverted to State {selected_index + 1}: {summary}")

    st.markdown("---")

    # Option to clear timeline
    if st.button("Clear Timeline"):
        clear_history()
        st.success("Timeline has been cleared.")

