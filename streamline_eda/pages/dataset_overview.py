import streamlit as st
import pandas as pd
from utils import (
    initialize_session_state,
    get_data_preview,
    get_numerical_columns, download_dataframe
)

def app():
    st.title("📊 Dataset Overview")

    # Initialize session state variables
    initialize_session_state()

    if st.session_state['df'] is None:
        st.warning("Please upload data first in the 'Upload Data' section.")
        return

    df = st.session_state['df']

    st.write("### 📈 Dataset Statistics")
    st.write(f"**Total Rows:** {df.shape[0]}")
    st.write(f"**Total Columns:** {df.shape[1]}")

    st.write("### 📝 Data Types")
    data_types = pd.DataFrame(df.dtypes, columns=["Data Type"])
    data_types.index.name = "Column"
    st.dataframe(data_types)

    st.write("### 📋 Missing Values")
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ['Column', 'Missing Values']
    missing_values['Missing Percentage (%)'] = (missing_values['Missing Values'] / df.shape[0]) * 100
    st.dataframe(missing_values)

    st.write("### 📊 Descriptive Statistics")
    st.dataframe(df.describe(include='all').transpose())

    st.markdown("---")

    st.write("### 🔍 Unique Values")

    # Dropdown to select the column
    unique_col = st.selectbox(
        "Select Column to View Unique Values",
        options=df.columns,
        key="dataset_overview_unique_col_select",
        help="Choose a column to display its unique values."
    )

    # Retrieve unique values
    unique_values = df[unique_col].unique()

    # Sort unique values for better readability with consistent key types
    try:
        unique_values_sorted = sorted(
            unique_values,
            key=lambda x: (pd.isnull(x), str(x).lower() if pd.notnull(x) else '')
        )
    except Exception as e:
        st.error(f"Error sorting unique values: {e}")
        unique_values_sorted = unique_values  # Fallback to unsorted if sorting fails

    # Handle columns with a large number of unique values
    max_display = 1000
    if len(unique_values_sorted) > max_display:
        st.warning(
            f"The column '{unique_col}' has {len(unique_values_sorted)} unique values. Displaying the first {max_display} unique entries."
        )
        unique_values_display = unique_values_sorted[:max_display]
    else:
        unique_values_display = unique_values_sorted

    # Convert unique values to a DataFrame for better display
    unique_values_df = pd.DataFrame(unique_values_display, columns=[f"Unique Values in '{unique_col}'"])

    st.dataframe(unique_values_df)

    st.markdown("---")

    # **New Section: Encoding Mappings**
    st.write("### 🔠 Encoding Mappings")

    encoding_mappings = st.session_state.get('encoding_mappings', {})

    if encoding_mappings:
        for col, mapping in encoding_mappings.items():
            st.write(f"**Column:** `{col}`")
            if all(isinstance(k, int) for k in mapping.keys()):
                # Label Encoding
                mapping_df = pd.DataFrame(list(mapping.items()), columns=[f"Encoded Value in '{col}'", f"Original Category"])
            else:
                # One-Hot Encoding
                mapping_df = pd.DataFrame(list(mapping.items()), columns=[f"Encoded Column in '{col}'", f"Original Category"])
            st.dataframe(mapping_df)
    else:
        st.write("No encoding mappings available. Ensure that categorical variables have been encoded in the 'Data Cleaning' section.")

    st.markdown("---")

    st.write("### 🔄 Dataset Updates")
    if st.session_state.get('last_update'):
        st.write(f"**Last Update:** {st.session_state['last_update']}")
    else:
        st.write("No updates yet.")

    st.markdown("---")


    st.write("### 📥 Export Current Data")
    download_dataframe(df, filename="eda_data.csv", file_format="CSV")
