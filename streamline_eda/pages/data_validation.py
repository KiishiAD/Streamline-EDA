import streamlit as st
import pandas as pd
import re
from utils import (
    initialize_session_state,
    get_data_preview,
    get_numerical_columns,
    save_state, download_dataframe
)
from datetime import datetime

def add_rule():
    """Add a validation rule based on user input."""
    # Ensure 'df' exists in session state
    if st.session_state.get('df') is None:
        st.error("No dataset found. Please upload data first.")
        return

    validation_type = st.session_state.get('validation_type_select', "Allowed Values")
    column = st.session_state.get('validation_column_select', st.session_state['df'].columns[0])

    rule_detail = None

    if validation_type == "Range":
        if pd.api.types.is_numeric_dtype(st.session_state['df'][column]):
            min_val = st.session_state.get('validation_min_val', st.session_state['df'][column].min())
            max_val = st.session_state.get('validation_max_val', st.session_state['df'][column].max())
            if min_val > max_val:
                st.error("Minimum value cannot be greater than maximum value.")
                return
            rule_detail = {"type": "range", "column": column, "min": min_val, "max": max_val}
        else:
            st.error("Range validation can only be applied to numeric columns.")
            return
    elif validation_type == "Allowed Values":
        allowed = st.session_state.get('validation_allowed_values', '')
        allowed_list = [x.strip() for x in allowed.split(",") if x.strip()]
        if not allowed_list:
            st.error("Please provide at least one allowed value.")
            return
        allowed_list = [str(x) for x in allowed_list]
        rule_detail = {"type": "allowed_values", "column": column, "allowed": allowed_list}
    elif validation_type == "Regex Pattern":
        pattern = st.session_state.get('validation_regex_pattern', '')
        if not pattern:
            st.error("Please provide a regex pattern.")
            return
        try:
            re.compile(pattern)
        except re.error:
            st.error("Invalid regex pattern.")
            return
        rule_detail = {"type": "regex", "column": column, "pattern": pattern}

    if rule_detail:
        st.session_state['validation_rules'].append(rule_detail)
        st.success(f"Added {validation_type} rule for column '{column}'.")
        summary = f"Added {validation_type} validation on column '{column}'."
        save_state(summary)  # Save state with summary

def validate_data():
    """Validate the data based on defined rules and modify the main DataFrame."""
    if not st.session_state['validation_rules']:
        st.error("No validation rules defined.")
        return

    if st.session_state.get('df') is None:
        st.error("No dataset found. Please upload data first.")
        return

    df = st.session_state['df']
    invalid_mask = pd.Series([False] * len(df), index=df.index)
    validation_results = []
    total_invalid = 0

    # Get preserve_missing value
    preserve_missing = st.session_state.get('preserve_missing', False)

    for rule in st.session_state['validation_rules']:
        col = rule['column']
        if rule['type'] == "range":
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Invalid rows: outside range
                current_invalid = (df[col] < rule['min']) | (df[col] > rule['max'])
                if preserve_missing:
                    # Exclude 'NaN' from invalid rows
                    current_invalid = current_invalid & (~df[col].isna())
            except Exception as e:
                st.error(f"Error processing Range validation on column '{col}': {e}")
                current_invalid = pd.Series([False] * len(df), index=df.index)
            invalid_mask |= current_invalid

            count = current_invalid.sum()
            validation_results.append({
                "rule": f"Range {rule['min']} - {rule['max']} on '{col}'",
                "invalid_count": int(count),
                "details": df.loc[current_invalid, [col]]
            })
            total_invalid += count
        elif rule['type'] == "allowed_values":
            series = df[col]
            if preserve_missing:
                current_invalid = ~series.isin(rule['allowed']) & ~series.isna()
            else:
                current_invalid = ~series.isin(rule['allowed'])
            invalid_mask |= current_invalid

            count = current_invalid.sum()
            validation_results.append({
                "rule": f"Allowed Values {', '.join(map(str, rule['allowed']))} on '{col}'",
                "invalid_count": int(count),
                "details": df.loc[current_invalid, [col]]
            })
            total_invalid += count
        elif rule['type'] == "regex":
            pattern = rule['pattern']
            series = df[col]
            try:
                matches = series.str.match(pattern, na=False)
                if preserve_missing:
                    # Exclude 'NaN' from invalid rows
                    current_invalid = ~matches & ~series.isna()
                else:
                    current_invalid = ~matches
            except Exception as e:
                st.error(f"Error processing Regex validation on column '{col}': {e}")
                current_invalid = pd.Series([False] * len(df), index=df.index)
            invalid_mask |= current_invalid

            count = current_invalid.sum()
            validation_results.append({
                "rule": f"Regex Pattern `{pattern}` on '{col}'",
                "invalid_count": int(count),
                "details": df.loc[current_invalid, [col]]
            })
            total_invalid += count

    st.session_state['validation_results'] = validation_results
    st.session_state['total_invalid'] = total_invalid

def remove_invalid_entries():
    """Remove invalid entries from the dataset based on validation results."""
    if not st.session_state.get('validation_results') or st.session_state['total_invalid'] == 0:
        st.info("No invalid entries to remove.")
        return

    if st.session_state.get('df') is None:
        st.error("No dataset found. Please upload data first.")
        return

    df = st.session_state['df']
    invalid_mask = pd.Series([False] * len(df), index=df.index)

    # Get preserve_missing value
    preserve_missing = st.session_state.get('preserve_missing', False)

    for rule in st.session_state['validation_rules']:
        col = rule['column']
        if rule['type'] == "range":
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                current_invalid = (df[col] < rule['min']) | (df[col] > rule['max'])
                if preserve_missing:
                    # Exclude 'NaN' from invalid rows
                    current_invalid = current_invalid & (~df[col].isna())
            except Exception as e:
                st.error(f"Error removing Range invalid entries on column '{col}': {e}")
                current_invalid = pd.Series([False] * len(df), index=df.index)
            invalid_mask |= current_invalid
        elif rule['type'] == "allowed_values":
            series = df[col]
            if preserve_missing:
                current_invalid = ~series.isin(rule['allowed']) & ~series.isna()
            else:
                current_invalid = ~series.isin(rule['allowed'])
            invalid_mask |= current_invalid
        elif rule['type'] == "regex":
            pattern = rule['pattern']
            series = df[col]
            try:
                matches = series.str.match(pattern, na=False)
                if preserve_missing:
                    # Exclude 'NaN' from invalid rows
                    current_invalid = ~matches & ~series.isna()
                else:
                    current_invalid = ~matches
            except Exception as e:
                st.error(f"Error removing Regex invalid entries on column '{col}': {e}")
                current_invalid = pd.Series([False] * len(df), index=df.index)
            invalid_mask |= current_invalid

    # Remove invalid entries
    df_cleaned = df[~invalid_mask].copy()

    # Update session state
    st.session_state['df'] = df_cleaned
    st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create a summary of the action
    summary = "Removed invalid entries based on validation rules."
    save_state(summary)  # Save state after removing invalid entries

    # Clear validation results
    st.session_state['validation_results'] = []
    st.session_state['total_invalid'] = 0

    st.success("Invalid entries have been removed from the dataset.")
    st.write("### Data Preview After Validation")
    st.session_state['df'] = df_cleaned  # Ensure 'df' is updated in session state
    st.dataframe(get_data_preview(df_cleaned, n=10))

def app():
    st.title("✅ Data Validation")

    # Initialize session state variables
    initialize_session_state()

    if st.session_state['df'] is None:
        st.warning("Please upload data first in the 'Upload Data' section.")
        return

    df = st.session_state['df']

    st.write("### Data Validation Rules")

    # Initialize validation rules in session state
    if 'validation_rules' not in st.session_state:
        st.session_state['validation_rules'] = []

    st.header("📋 Define Validation Rules")

    rule_container = st.container()

    with rule_container:
        col1, col2 = st.columns([2, 1])
        with col1:
            column = st.selectbox(
                "Select Column",
                options=df.columns,
                key="validation_column_select"
            )
        with col2:
            validation_type = st.selectbox(
                "Validation Type",
                options=["Allowed Values", "Range", "Regex Pattern"],
                key="validation_type_select"
            )

        # Define inputs based on validation type
        if validation_type == "Range":
            # Attempt to infer if the column is numeric
            if pd.api.types.is_numeric_dtype(df[column]):
                min_val = st.number_input(
                    "Minimum Value",
                    value=float(df[column].min()),
                    key="validation_min_val"
                )
                max_val = st.number_input(
                    "Maximum Value",
                    value=float(df[column].max()),
                    key="validation_max_val"
                )
            else:
                st.error("Range validation can only be applied to numeric columns.")
        elif validation_type == "Allowed Values":
            allowed = st.text_area(
                "Allowed Values (comma-separated)",
                key="validation_allowed_values",
                help='Specify allowed values separated by commas.'
            )
        elif validation_type == "Regex Pattern":
            pattern = st.text_input(
                "Regex Pattern",
                value="",
                key="validation_regex_pattern"
            )

        # Button to add the rule
        add_rule_button = st.button("Add Rule", on_click=add_rule, key="validation_add_rule")

    # Display current rules
    st.header("📑 Current Validation Rules")
    if st.session_state['validation_rules']:
        for idx, rule in enumerate(st.session_state['validation_rules'], 1):
            st.write(f"**Rule {idx}:** {rule['type'].capitalize()} on column '{rule['column']}'")
            if rule['type'] == "range":
                st.write(f" - Range: {rule['min']} to {rule['max']}")
            elif rule['type'] == "allowed_values":
                st.write(f" - Allowed Values: {', '.join(map(str, rule['allowed']))}")
            elif rule['type'] == "regex":
                st.write(f" - Regex Pattern: `{rule['pattern']}`")
    else:
        st.write("No validation rules defined yet.")

    if st.button("Clear All Rules", key="validation_clear_all_rules"):
        st.session_state['validation_rules'] = []
        st.success("All validation rules have been cleared.")
        summary = "Cleared all validation rules."
        save_state(summary)  # Save state after clearing rules

    st.markdown("---")

    st.header("🔍 Run Validation")

    # Section: Handle Invalid Entries
    st.subheader("📝 Handle Invalid Entries")

    # Checkbox to decide whether to preserve rows with missing values
    preserve_missing = st.checkbox(
        "Preserve rows with missing values (NaN)",
        value=True,
        help="When checked, rows with missing values will not be removed, even if they violate validation rules.",
        key="validation_preserve_missing"
    )
    st.session_state['preserve_missing'] = preserve_missing

    # Button to run validation
    validate_button = st.button("Validate Data", on_click=validate_data, key="validation_run_validation")

    # Display validation results if they exist
    if 'validation_results' in st.session_state and 'total_invalid' in st.session_state:
        total_invalid = st.session_state['total_invalid']
        validation_results = st.session_state['validation_results']

        st.write(f"**Total Invalid Entries:** {total_invalid}")

        for res in validation_results:
            st.write(f"**Rule:** {res['rule']}")
            st.write(f"**Invalid Count:** {res['invalid_count']}")
            if res['invalid_count'] > 0:
                with st.expander("View Invalid Entries"):
                    st.dataframe(res['details'])

        # Option to remove invalid entries
        if total_invalid > 0:
            remove_button = st.button("Remove Invalid Entries", on_click=remove_invalid_entries, key="validation_remove_invalid")

    st.markdown("---")


    st.write("### 📥 Export Current Data")
    download_dataframe(df, filename="eda_data.csv", file_format="CSV")
