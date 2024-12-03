# pages/data_cleaning.py

import streamlit as st
import pandas as pd
from utils import (
    get_data_preview, initialize_session_state,
    identify_outliers_cached, remove_outliers, cap_outliers,
    standardize_columns, normalize_columns, encode_categorical,
    get_numerical_columns, save_state, download_dataframe
)

from visualizations import (
    plot_missing_values,
    plot_outliers_scatter,
    plot_interactive_box,
)

import base64
from datetime import datetime
import re


def download_cleaned_data(df):
    """Provide a download link for the cleaned DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
    href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download Cleaned Data</a>'
    st.markdown(href, unsafe_allow_html=True)


def drop_columns():
    """Drop selected columns from the DataFrame."""
    selected_columns = st.session_state.get('data_cleaning_columns_to_drop', [])
    if selected_columns:
        df_before_drop = st.session_state['df']
        try:
            df_cleaned = df_before_drop.drop(columns=selected_columns)
            st.session_state['df'] = df_cleaned
            st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            summary = f"Dropped columns: {', '.join(selected_columns)}."
            save_state(summary)  # Save state after dropping columns
            st.success(f"Successfully dropped columns: **{', '.join(selected_columns)}**")

            # Reset the multiselect selection by using a separate flag
            st.session_state['drop_columns_trigger'] = True

            # Provide a preview of the updated dataset
            st.write("### Data Preview After Dropping Columns")
            st.dataframe(get_data_preview(df_cleaned, n=10))

            # Provide download link
            download_cleaned_data(df_cleaned)

        except Exception as e:
            st.error(f"Error while dropping columns: {e}")
    else:
        st.error("No columns selected to drop.")


def detect_date_related_columns(df, keywords=None, exclude_suffixes=None, exclude_types=None):
    """
    Detect columns in the DataFrame that are related to dates based on keywords.
    Exclude columns that have certain suffixes or data types to avoid re-processing transformed columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame to scan.
    - keywords (list): List of keywords to look for in column names.
    - exclude_suffixes (list): List of suffixes to exclude.
    - exclude_types (list): List of data types to exclude.

    Returns:
    - list: Columns that match the keywords.
    """
    if keywords is None:
        keywords = ['year', 'date', 'commissioned', 'installed']
    if exclude_suffixes is None:
        exclude_suffixes = ['_year', '_datetime']
    if exclude_types is None:
        exclude_types = ['int64', 'float64', 'datetime64[ns]', 'datetime64[ns, UTC]']

    date_columns = []
    for col in df.columns:
        col_clean = col.strip()
        # Exclude columns with certain suffixes (case-insensitive)
        if any(col_clean.lower().endswith(suffix.lower()) for suffix in exclude_suffixes):
            continue
        # Exclude columns with certain data types
        if str(df[col].dtype) in exclude_types:
            continue
        for keyword in keywords:
            if keyword.lower() in col_clean.lower():
                date_columns.append(col_clean)
                break  # Avoid duplicate entries if multiple keywords match
    return date_columns


def extract_years_from_column(series):
    """
    Extract years from a pandas Series containing strings with embedded years.

    Parameters:
    - series (pd.Series): The Series to process.

    Returns:
    - pd.Series: Series containing the extracted years as integers. NaN where extraction fails.
    """
    # Define a regex pattern to match four-digit years (e.g., 2015)
    pattern = re.compile(r'(19|20)\d{2}')

    def extract_year(value):
        if isinstance(value, str):
            match = pattern.search(value)
            if match:
                return int(match.group())
        return pd.NA

    return series.apply(extract_year)


def recommend_and_apply_year_extraction(df, date_columns):
    """
    Recommend and apply year extraction transformations for detected date-related columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process.
    - date_columns (list): List of date-related columns detected.

    Returns:
    - pd.DataFrame: The transformed DataFrame.
    """
    for col in date_columns:
        st.write(f"#### 📅 Column: `{col}`")

        # Attempt to extract years
        extracted_years = extract_years_from_column(df[col])
        num_extracted = extracted_years.notna().sum()
        total = len(df[col])

        if num_extracted / total > 0.5:  # Arbitrary threshold; adjust as needed
            st.write(f"✅ **Detected embedded years in `{col}`.**")
            st.write(f"Extracted {num_extracted} out of {total} entries.")

            # Provide options for transformation
            transformation_option = st.selectbox(
                f"Choose a transformation for `{col}`:",
                ["None", "Extract Year into New Column", "Convert to Datetime (Overwrite Existing Column)"],
                key=f"transformation_option_{col}"
            )

            if transformation_option == "Extract Year into New Column":
                new_col_name = f"{col}_Year"
                if new_col_name not in df.columns:
                    if st.button(f"Confirm: Extract Year and create `{new_col_name}`", key=f"confirm_extract_year_{col}"):
                        # Save current state to timeline
                        save_state(f"Extracted Year from `{col}` into `{new_col_name}`.")

                        # Apply the transformation
                        df[new_col_name] = extracted_years

                        st.write(f"✅ Extracted year and created new column `{new_col_name}`.")
                else:
                    st.warning(f"Column `{new_col_name}` already exists. Skipping extraction to avoid duplication.")

            elif transformation_option == "Convert to Datetime (Overwrite Existing Column)":
                if df[col].dtype == 'object' or df[col].dtype.name.startswith('str'):
                    if st.button(f"Confirm: Convert `{col}` to Datetime", key=f"confirm_convert_datetime_{col}"):
                        # Save current state to timeline
                        save_state(f"Converted `{col}` to Datetime.")

                        # Apply the transformation
                        df[col] = pd.to_datetime(extracted_years, format='%Y', errors='coerce')

                        st.write(f"✅ Converted `{col}` to datetime format.")
                else:
                    st.warning(f"Column `{col}` is not of type object/string. Skipping conversion.")
        else:
            st.write(f"⚠️ **Insufficient data to extract years from `{col}`.**")
            st.write(f"Extracted {num_extracted} out of {total} entries.")


def app():
    st.title("🧹 Data Cleaning")

    # Initialize session state variables
    initialize_session_state()

    if st.session_state['df'] is None:
        st.warning("Please upload data first in the 'Upload Data' section.")
        return

    df = st.session_state['df']
    st.write("### Original Data Preview")
    st.dataframe(get_data_preview(df, n=10))

    st.write("### Missing Values Visualization")
    fig = plot_missing_values(df)
    st.plotly_chart(fig)

    st.write("### Handle Missing Values")
    # Use unique keys for widgets to preserve their state
    missing_option = st.selectbox(
        "Choose how to handle missing values",
        ["None", "Drop Rows with Any Missing Values",
         "Drop Rows Based on Specific Columns",
         "Fill Missing Values"],
        key="data_cleaning_missing_option"
    )

    if missing_option == "Drop Rows with Any Missing Values":
        if st.button("Drop All Rows with Missing Values", key="data_cleaning_drop_all_missing"):
            df_cleaned = df.dropna()
            st.session_state['df'] = df_cleaned
            st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            summary = "Dropped all rows with any missing values."
            save_state(summary)  # Save state after dropping rows
            st.success("Dropped all rows with any missing values.")
            st.write("### Data Preview After Dropping Missing Values")
            st.dataframe(get_data_preview(df_cleaned, n=10))
            download_cleaned_data(df_cleaned)

    elif missing_option == "Drop Rows Based on Specific Columns":
        columns_with_missing = df.columns[df.isnull().any()].tolist()
        selected_columns = st.multiselect(
            "Select columns to check for missing values",
            columns_with_missing,
            key="data_cleaning_selected_columns_drop_missing"
        )
        if selected_columns:
            if st.button("Drop Rows with Missing Values in Selected Columns",
                         key="data_cleaning_drop_selected_missing"):
                df_cleaned = df.dropna(subset=selected_columns)
                st.session_state['df'] = df_cleaned
                st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                summary = f"Dropped rows with missing values in columns: {', '.join(selected_columns)}."
                save_state(summary)  # Save state after dropping rows
                st.success(f"Dropped rows with missing values in columns: {', '.join(selected_columns)}.")
                st.write("### Data Preview After Dropping Missing Values")
                st.dataframe(get_data_preview(df_cleaned, n=10))
                download_cleaned_data(df_cleaned)
        else:
            st.info("Please select at least one column.")

    elif missing_option == "Fill Missing Values":
        fill_methods = ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill", "Custom Value"]
        fill_method = st.selectbox("Select Fill Method", fill_methods, key="data_cleaning_fill_method")
        columns_with_missing = df.columns[df.isnull().any()].tolist()
        selected_columns = st.multiselect(
            "Select columns to fill missing values",
            columns_with_missing,
            key="data_cleaning_selected_columns_fill"
        )
        custom_value = None
        if fill_method == "Custom Value":
            custom_value = st.text_input("Enter custom value to fill missing values",
                                         key="data_cleaning_custom_fill_value")

        if st.button("Fill Missing Values", key="data_cleaning_fill_missing"):
            if not selected_columns:
                st.error("Please select at least one column to fill missing values.")
            else:
                df_filled = df.copy()
                try:
                    if fill_method == "Mean":
                        df_filled[selected_columns] = df_filled[selected_columns].fillna(
                            df_filled[selected_columns].mean())
                        summary = f"Filled missing values in columns {', '.join(selected_columns)} using Mean."
                    elif fill_method == "Median":
                        df_filled[selected_columns] = df_filled[selected_columns].fillna(
                            df_filled[selected_columns].median())
                        summary = f"Filled missing values in columns {', '.join(selected_columns)} using Median."
                    elif fill_method == "Mode":
                        for col in selected_columns:
                            mode_val = df_filled[col].mode()
                            if not mode_val.empty:
                                df_filled[col].fillna(mode_val[0], inplace=True)
                        summary = f"Filled missing values in columns {', '.join(selected_columns)} using Mode."
                    elif fill_method == "Forward Fill":
                        df_filled[selected_columns] = df_filled[selected_columns].fillna(method='ffill')
                        summary = f"Filled missing values in columns {', '.join(selected_columns)} using Forward Fill."
                    elif fill_method == "Backward Fill":
                        df_filled[selected_columns] = df_filled[selected_columns].fillna(method='bfill')
                        summary = f"Filled missing values in columns {', '.join(selected_columns)} using Backward Fill."
                    elif fill_method == "Custom Value":
                        if custom_value == "":
                            st.error("Please enter a custom value.")
                            return
                        df_filled[selected_columns] = df_filled[selected_columns].fillna(custom_value)
                        summary = f"Filled missing values in columns {', '.join(selected_columns)} using Custom Value '{custom_value}'."
                except Exception as e:
                    st.error(f"Error while filling missing values: {e}")
                    return

                st.session_state['df'] = df_filled
                st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_state(summary)  # Save state after filling missing values
                st.success(f"Filled missing values in columns: {', '.join(selected_columns)} using {fill_method}.")
                st.write("### Data Preview After Filling Missing Values")
                st.dataframe(get_data_preview(df_filled, n=10))
                download_cleaned_data(df_filled)

    st.markdown("---")

    # **New Section: Remove Duplicates**
    st.write("### Remove Duplicate Rows")
    if st.checkbox("Show Duplicate Rows", key="data_cleaning_show_duplicates"):
        duplicate_rows = df[df.duplicated()]
        st.write(f"**Total Duplicate Rows:** {duplicate_rows.shape[0]}")
        st.dataframe(duplicate_rows)

    if st.button("Remove Duplicate Rows", key="data_cleaning_remove_duplicates"):
        df_cleaned = df.drop_duplicates()
        st.session_state['df'] = df_cleaned
        st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = "Removed duplicate rows."
        save_state(summary)  # Save state after removing duplicates
        st.success("Duplicate rows have been removed.")
        st.write("### Data Preview After Removing Duplicates")
        st.dataframe(get_data_preview(df_cleaned, n=10))
        download_cleaned_data(df_cleaned)

    st.markdown("---")

    st.write("### Outlier Detection and Handling")
    outlier_methods = ["None", "Z-Score", "IQR"]
    selected_method = st.selectbox("Select Outlier Detection Method", outlier_methods,
                                   key="data_cleaning_outlier_method")

    if selected_method in ["Z-Score", "IQR"]:
        numerical_columns = get_numerical_columns(df)
        selected_columns = st.multiselect("Select columns for outlier detection", numerical_columns,
                                          key="data_cleaning_selected_columns_outliers")
        if selected_columns:
            threshold = st.slider(
                "Select threshold for outlier detection",
                min_value=1.0, max_value=5.0, value=1.5, step=0.1,
                key="data_cleaning_outlier_threshold"
            )
            if st.button("Detect Outliers", key="data_cleaning_detect_outliers"):
                outliers = identify_outliers_cached(df, method=selected_method, column=None, threshold=threshold)
                total_outliers = sum([len(indices) for indices in outliers.values()])
                st.write(f"**Total Outliers Detected:** {total_outliers}")
                for col, indices in outliers.items():
                    if col in selected_columns and indices:
                        st.write(f"**{col}**: {len(indices)} outliers")
                        # Interactive Scatter Plot
                        fig_scatter = plot_outliers_scatter(df, col, indices)
                        st.pyplot(fig_scatter)
                        # Interactive Box Plot
                        fig_box = plot_interactive_box(df, col)
                        st.plotly_chart(fig_box)
        else:
            st.info("Please select at least one numerical column.")

        st.write("#### Handle Outliers")
        handle_options = ["None", "Remove Outliers", "Cap Outliers"]
        handle_choice = st.selectbox("Choose how to handle outliers", handle_options,
                                     key="data_cleaning_handle_outliers")

        if handle_choice == "Remove Outliers":
            if st.button("Remove Detected Outliers", key="data_cleaning_remove_outliers"):
                outliers = identify_outliers_cached(df, method=selected_method, column=None, threshold=threshold)
                df_cleaned = remove_outliers(df, outliers)
                st.session_state['df'] = df_cleaned
                st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                summary = f"Removed outliers from columns {', '.join(selected_columns)} using {selected_method} method."
                save_state(summary)  # Save state after removing outliers
                st.success("Outliers removed from the dataset.")
                st.write("### Data Preview After Removing Outliers")
                st.dataframe(get_data_preview(df_cleaned, n=10))
                download_cleaned_data(df_cleaned)

        elif handle_choice == "Cap Outliers":
            if st.button("Cap Detected Outliers", key="data_cleaning_cap_outliers"):
                outliers = identify_outliers_cached(df, method=selected_method, column=None, threshold=threshold)
                df_capped = cap_outliers(df, outliers, method=selected_method)
                st.session_state['df'] = df_capped
                st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                summary = f"Capped outliers in columns {', '.join(selected_columns)} using {selected_method} method."
                save_state(summary)  # Save state after capping outliers
                st.success("Outliers have been capped.")
                st.write("### Data Preview After Capping Outliers")
                st.dataframe(get_data_preview(df_capped, n=10))
                download_cleaned_data(df_capped)

    st.markdown("---")

    st.write("### Data Transformation")
    transformation_options = ["None", "Standardization", "Normalization", "Encode Categorical Variables"]
    selected_transformation = st.selectbox("Select Transformation", transformation_options,
                                           key="data_cleaning_transformation")

    if selected_transformation == "Standardization":
        numeric_columns = get_numerical_columns(df)
        selected_columns = st.multiselect("Select numerical columns to standardize", numeric_columns,
                                          key="data_cleaning_selected_columns_standardize")
        if selected_columns:
            if st.button("Standardize Selected Columns", key="data_cleaning_standardize_columns"):
                df_transformed = standardize_columns(df, selected_columns)
                st.session_state['df'] = df_transformed
                st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                summary = f"Standardized columns: {', '.join(selected_columns)}."
                save_state(summary)  # Save state after standardization
                st.success(f"Standardized columns: {', '.join(selected_columns)}.")
                st.write("### Data Preview After Standardization")
                st.dataframe(get_data_preview(df_transformed, n=10))
                download_cleaned_data(df_transformed)
        else:
            st.info("Please select at least one numerical column.")

    elif selected_transformation == "Normalization":
        numeric_columns = get_numerical_columns(df)
        selected_columns = st.multiselect("Select numerical columns to normalize", numeric_columns,
                                          key="data_cleaning_selected_columns_normalize")
        if selected_columns:
            if st.button("Normalize Selected Columns", key="data_cleaning_normalize_columns"):
                df_transformed = normalize_columns(df, selected_columns)
                st.session_state['df'] = df_transformed
                st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                summary = f"Normalized columns: {', '.join(selected_columns)}."
                save_state(summary)  # Save state after normalization
                st.success(f"Normalized columns: {', '.join(selected_columns)}.")
                st.write("### Data Preview After Normalization")
                st.dataframe(get_data_preview(df_transformed, n=10))
                download_cleaned_data(df_transformed)
        else:
            st.info("Please select at least one numerical column.")

    elif selected_transformation == "Encode Categorical Variables":
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        selected_columns = st.multiselect("Select categorical columns to encode", categorical_columns,
                                          key="data_cleaning_selected_columns_encode")
        encoding_methods = ["One-Hot", "Label"]
        if selected_columns:
            # Check for already encoded columns
            already_encoded = [col for col in selected_columns if col in st.session_state.get('encoding_mappings', {})]
            if already_encoded:
                st.warning(
                    f"The following columns have already been encoded and will be re-encoded: {', '.join(already_encoded)}. Previous mappings will be overwritten.")
            encoding_method = st.selectbox("Select Encoding Method", encoding_methods,
                                           key="data_cleaning_encoding_method")
            if st.button("Encode Selected Columns", key="data_cleaning_encode_columns"):
                df_encoded = encode_categorical(df, selected_columns, encoding=encoding_method)
                st.session_state['df'] = df_encoded
                st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                summary = f"Encoded categorical columns {', '.join(selected_columns)} using {encoding_method} Encoding."
                save_state(summary)  # Save state after encoding
                st.success(f"Encoded columns: {', '.join(selected_columns)} using {encoding_method} Encoding.")
                st.write("### Data Preview After Encoding")
                st.dataframe(get_data_preview(df_encoded, n=10))
                download_cleaned_data(df_encoded)
        else:
            st.info("Please select at least one categorical column.")

    st.markdown("---")

    st.write("### Custom Transformation")
    custom_transformation = st.text_area(
        "Enter a custom transformation function",
        help="Example: df['new_col'] = df['col1'] + df['col2']",
        key="data_cleaning_custom_transformation"
    )
    if st.button("Apply Custom Transformation", key="data_cleaning_apply_custom_transformation"):
        if custom_transformation.strip() == "":
            st.error("Please enter a valid transformation function.")
        else:
            try:
                # Define the allowed built-ins and variables for exec
                exec_env = {'df': df, 'pd': pd, 'st': st}
                exec(custom_transformation, exec_env)
                # Retrieve the modified DataFrame
                df_modified = exec_env.get('df')
                if isinstance(df_modified, pd.DataFrame):
                    # Save current state to timeline
                    save_state("Applied custom transformation.")

                    st.session_state['df'] = df_modified
                    st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    summary = "Applied custom transformation."
                    save_state(summary)  # Save state after custom transformation
                    st.success("Custom transformation applied successfully.")
                    st.write("### Data Preview After Custom Transformation")
                    st.dataframe(get_data_preview(df_modified, n=10))
                    download_cleaned_data(df_modified)
                else:
                    st.error("The custom transformation did not return a DataFrame.")
            except Exception as e:
                st.error(f"Error in custom transformation: {e}")

    st.markdown("---")

    # **New Section: Change Column Data Types**
    st.write("### Change Column Data Types")
    st.write("Select columns and specify the target data type.")

    # Identify columns and their current data types
    col_types = df.dtypes
    options = df.columns.tolist()

    selected_columns = st.multiselect(
        "Select Columns to Change Data Types",
        options=options,
        help="Select one or more columns whose data types you want to change.",
        key="data_cleaning_selected_columns_type_change"
    )

    if selected_columns:
        # Define available target data types based on current types
        type_options = {}
        for col in selected_columns:
            current_type = str(col_types[col])
            if 'int' in current_type:
                type_options[col] = ['int64', 'float64']
            elif 'float' in current_type:
                type_options[col] = ['float64', 'int64']
            elif 'object' in current_type:
                type_options[col] = ['string', 'category', 'int64']
            else:
                type_options[col] = [current_type]  # No change available

        # Display selectboxes for each selected column
        new_types = {}
        for col in selected_columns:
            available_types = type_options[col]
            if len(available_types) > 1:
                new_type = st.selectbox(
                    f"Select new data type for `{col}` (Current: {col_types[col]})",
                    options=available_types,
                    key=f"data_cleaning_new_type_{col}"
                )
                new_types[col] = new_type
            else:
                st.info(f"No alternative data types available for `{col}`.")
                new_types[col] = col_types[col]

        # **Additional Section: Convert Numeric Strings to Integers**
        # Detect columns with object type containing only numeric strings
        numeric_string_columns = []
        for col in selected_columns:
            if str(col_types[col]) == 'object':
                # Check if all non-null entries are numeric strings
                is_numeric = df[col].dropna().apply(lambda x: str(x).isdigit()).all()
                if is_numeric:
                    numeric_string_columns.append(col)

        if numeric_string_columns:
            st.write("### 🔢 Convert Numeric String Columns to Integers")
            convert_columns = st.multiselect(
                "Select columns to convert from string to integer:",
                options=numeric_string_columns,
                key="data_cleaning_numeric_string_columns"
            )

            if convert_columns:
                if st.button("Convert Selected Columns to Integer", key="data_cleaning_convert_numeric_strings"):
                    try:
                        for col in convert_columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                            st.write(f"**{col}**: Converted from string to integer.")
                            summary = f"Converted column `{col}` from string to integer."
                            save_state(summary)  # Save state after conversion

                        st.session_state['df'] = df
                        st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.success("Selected string columns have been converted to integers.")
                    except Exception as e:
                        st.error(f"Error while converting columns: {e}")
            else:
                st.info("Select at least one column to convert from string to integer.")
        else:
            st.info("No string columns with purely numeric values detected among the selected columns.")

        if st.button("Change Data Types", key="data_cleaning_change_data_types"):
            try:
                df_transformed = df.copy()
                for col, new_type in new_types.items():
                    if new_type != str(col_types[col]):
                        df_transformed[col] = df_transformed[col].astype(new_type)
                        st.write(f"**{col}**: Changed from `{col_types[col]}` to `{new_type}`.")
                st.session_state['df'] = df_transformed
                st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                summary = f"Changed data types for columns: {', '.join(new_types.keys())}."
                save_state(summary)  # Save state after changing data types
                st.success("Data types updated successfully.")
                st.write("### Data Preview After Changing Data Types")
                st.dataframe(get_data_preview(df_transformed, n=10))
                download_cleaned_data(df_transformed)
            except Exception as e:
                st.error(f"Error while changing data types: {e}")
    else:
        st.info("Select one or more columns to change their data types.")

    st.markdown("---")

    # **New Section: Drop Columns**
    st.write("### Drop Columns")
    st.write("Select the columns you want to drop from the dataset.")

    columns_to_drop = st.multiselect(
        "Select Columns to Drop",
        options=df.columns,
        help="Select one or more columns to remove from the dataset.",
        key="data_cleaning_columns_to_drop"
    )
    # Removed the manual assignment to st.session_state to prevent the error

    if columns_to_drop:
        st.write(f"You have selected to drop the following columns: **{', '.join(columns_to_drop)}**")
        if st.button("Confirm Drop Columns", key="data_cleaning_confirm_drop_columns"):
            drop_columns()
    else:
        st.info("Select one or more columns to drop from the dataset.")

    st.markdown("---")

    # **New Section: Detect and Transform Date-related Columns**
    st.write("### 🕵️‍♂️ Detect and Transform Date-related Columns")
    st.write(
        "The app has detected the following columns that may contain embedded years or dates based on their names.")

    # Detect date-related columns based on keywords, excluding already transformed columns
    date_keywords = ['year', 'date', 'commissioned', 'installed']
    exclude_suffixes = ['_year', '_datetime']
    exclude_types = ['int64', 'float64', 'datetime64[ns]', 'datetime64[ns, UTC]']
    date_columns = detect_date_related_columns(df, keywords=date_keywords, exclude_suffixes=exclude_suffixes, exclude_types=exclude_types)

    if date_columns:
        st.write("### 📅 Detected Date-related Columns:")
        for col in date_columns:
            st.write(f"- `{col}`")

        # Recommend transformations
        st.write("### 🔍 Recommendations:")
        recommend_and_apply_year_extraction(df, date_columns)

        # Update the DataFrame in session state after transformations
        st.session_state['df'] = df
    else:
        st.write("No date-related columns detected based on the predefined keywords.")

    st.markdown("---")

    # Add Download Button
    st.write("### 📥 Export Current Data")
    download_dataframe(df, filename="eda_data.csv", file_format="CSV")
