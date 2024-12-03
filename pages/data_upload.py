# pages/data_upload.py

import streamlit as st
import pandas as pd
from utils import (
    get_data_preview, initialize_session_state, save_state
)

import base64
from datetime import datetime


def download_cleaned_data(df):
    """Provide a download link for the cleaned DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
    href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download Cleaned Data</a>'
    st.markdown(href, unsafe_allow_html=True)


def app():
    st.title("📥 Upload Data")

    initialize_session_state()

    uploaded_file = st.file_uploader("Choose a file to upload", type=["csv", "xlsx", "json", "parquet"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # Initialize variables for sheet selection and header row
        sheet_selected = None
        header_row = 0  # Default header row

        if file_extension == 'xlsx':
            # Read the Excel file to get sheet names
            try:
                excel = pd.ExcelFile(uploaded_file)
                sheets = excel.sheet_names
                st.session_state['sheets'] = sheets  # Store sheet names in session_state

                # Allow user to select a sheet
                sheet_selected = st.selectbox("Select a sheet to upload", sheets, key="data_upload_sheet_selection")
            except Exception as e:
                st.error(f"Error reading Excel file: {e}")
                sheet_selected = None

        # Header row selection (applicable for CSV and Excel)
        if file_extension in ['csv', 'xlsx']:
            # Determine the total number of rows to set as options
            if file_extension == 'csv':
                # Read a small chunk to get the total rows
                try:
                    preview = pd.read_csv(uploaded_file, nrows=1000)
                    total_rows = len(preview) + 1  # Approximate total rows
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
                    total_rows = 10
            elif file_extension == 'xlsx' and sheet_selected:
                try:
                    # Reset the pointer to read the selected sheet
                    uploaded_file.seek(0)
                    df_preview = pd.read_excel(uploaded_file, sheet_name=sheet_selected, nrows=1000)
                    total_rows = len(df_preview) + 1  # Approximate total rows
                except Exception as e:
                    st.error(f"Error reading selected sheet: {e}")
                    total_rows = 10
            else:
                total_rows = 10

            header_row = st.number_input(
                "Select the row number to use as header (0-based index)",
                min_value=0,
                max_value=total_rows,
                value=0,
                step=1,
                key="data_upload_header_row"
            )

        # Reset the file pointer to read the file again with the selected parameters
        uploaded_file.seek(0)

        # Read the file with the selected sheet and header row
        try:
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file, header=header_row, na_values=['', 'NA', 'NaN'])
            elif file_extension == 'xlsx':
                if sheet_selected:
                    df = pd.read_excel(uploaded_file, sheet_name=sheet_selected, header=header_row, na_values=['', 'NA', 'NaN'])
                else:
                    st.error("No sheet selected.")
                    return
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            elif file_extension == 'parquet':
                df = pd.read_parquet(uploaded_file)
            else:
                st.error("Unsupported file format.")
                return

            # Handle cases where the header_row might skip actual data
            if file_extension in ['csv', 'xlsx'] and header_row > 0:
                df.reset_index(drop=True, inplace=True)

            # Store the DataFrame and related info in session_state
            st.session_state['df'] = df
            st.session_state['file_path'] = uploaded_file.name
            st.session_state['encoding_mappings'] = {}  # Reset encoding mappings on new upload
            st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save the initial state to history with summary
            summary = f"Uploaded data from '{uploaded_file.name}'."
            if file_extension in ['csv', 'xlsx']:
                summary += f" Sheet: '{sheet_selected}'." if file_extension == 'xlsx' else ""
                summary += f" Header row: {header_row}."
            save_state(summary)

            st.success("Data uploaded successfully!")
            st.write("### Data Preview")
            st.dataframe(get_data_preview(df, n=10))

            # Provide download link for the uploaded data (optional)
            download_cleaned_data(df)

        except Exception as e:
            st.error(f"Error uploading file: {e}")

