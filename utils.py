import base64
import os
import pandas as pd
from io import BytesIO
from config import DATA_DIR, ALLOWED_EXTENSIONS
import logging
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from datetime import datetime
from scipy import stats  # Ensure scipy is imported for outlier detection
import uuid  # Import uuid for unique session IDs



def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to the DATA_DIR."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logging.info(f"Saved file: {file_path}")
        st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return file_path
    except Exception as e:
        logging.error(f"Error saving file: {e}")
        st.error(f"Error saving file: {e}")
        return None

def load_data(file_path):
    """Load data from the given file path based on its extension."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            logging.error("Unsupported file format")
            st.error("Unsupported file format")
            return None
        st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return None

def get_data_preview(df, n=10):
    """Return the first n rows of the DataFrame."""
    return df.head(n)

def initialize_session_state():
    """Initialize necessary session state variables without overwriting existing ones."""
    if 'df' not in st.session_state:
        st.session_state['df'] = None  # The current DataFrame
    if 'history' not in st.session_state:
        st.session_state['history'] = []  # List to store DataFrame states with summaries
    if 'encoding_mappings' not in st.session_state:
        st.session_state['encoding_mappings'] = {}  # For encoding info
    if 'last_update' not in st.session_state:
        st.session_state['last_update'] = None  # Timestamp of last update
    # Initialize per-page states if needed
    # Example for Data Cleaning page
    if 'data_cleaning_missing_option' not in st.session_state:
        st.session_state['data_cleaning_missing_option'] = "None"
    # Initialize EDA-specific states
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())  # Unique session ID
def save_state(summary):
    """
    Save the current DataFrame state to history with a summary.

    Parameters:
    - summary (str): A brief description of the action performed.
    """
    if st.session_state['df'] is not None:
        # Make a deep copy to ensure independence from future changes
        state = {
            'df': st.session_state['df'].copy(),
            'summary': summary,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state['history'].append(state)
        logging.info(f"State saved: {summary}")

def revert_to_state(index):
    """
    Revert the DataFrame to a specific state in history.

    Parameters:
    - index (int): The index of the state in the history list.
    """
    if 0 <= index < len(st.session_state['history']):
        st.session_state['df'] = st.session_state['history'][index]['df'].copy()
        st.session_state['last_update'] = f"Reverted to: {st.session_state['history'][index]['summary']}"
        st.success(f"Reverted to state: {st.session_state['history'][index]['summary']}")
        logging.info(f"Reverted to state index {index}: {st.session_state['history'][index]['summary']}")
    else:
        st.error("Invalid history index.")

def clear_history():
    """Clear all saved states in history."""
    st.session_state['history'] = []
    st.success("History has been cleared.")
    logging.info("History cleared.")

def get_numerical_columns(df):
    """Return a list of numerical columns in the DataFrame."""
    return df.select_dtypes(include=['number']).columns.tolist()

def identify_outliers(df, method='IQR', column=None, threshold=1.5):
    """
    Identify outliers in a DataFrame using Z-Score or IQR method.

    Parameters:
    - df (pd.DataFrame): The DataFrame to analyze.
    - method (str): The method to use ('Z-Score' or 'IQR').
    - column (str): The column to analyze. If None, analyze all numerical columns.
    - threshold (float): The threshold to identify outliers.

    Returns:
    - outliers (dict): Dictionary with column names as keys and outlier indices as values.
    """
    outliers = {}
    numeric_cols = df.select_dtypes(include=['number']).columns if column is None else [column]

    for col in numeric_cols:
        if method == 'Z-Score':
            z_scores = stats.zscore(df[col].dropna())
            abs_z_scores = abs(z_scores)
            outlier_indices = df[col].dropna().index[abs_z_scores > threshold].tolist()
        elif method == 'IQR':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        else:
            outlier_indices = []

        outliers[col] = outlier_indices

    return outliers

def remove_outliers(df, outliers):
    """
    Remove outliers from the DataFrame based on provided indices.

    Parameters:
    - df (pd.DataFrame): The original DataFrame.
    - outliers (dict): Dictionary with column names as keys and outlier indices as values.

    Returns:
    - df_cleaned (pd.DataFrame): DataFrame with outliers removed.
    """
    all_outliers = set()
    for indices in outliers.values():
        all_outliers.update(indices)
    df_cleaned = df.drop(index=all_outliers)
    st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Removed outliers: {len(all_outliers)} rows removed.")
    return df_cleaned

def cap_outliers(df, outliers, method='IQR'):
    """
    Cap outliers to the lower and upper bounds.

    Parameters:
    - df (pd.DataFrame): The original DataFrame.
    - outliers (dict): Dictionary with column names as keys and outlier indices as values.
    - method (str): The method used to identify outliers ('IQR' or 'Z-Score').

    Returns:
    - df_capped (pd.DataFrame): DataFrame with outliers capped.
    """
    df_capped = df.copy()
    for col, indices in outliers.items():
        if method == 'Z-Score':
            z_scores = stats.zscore(df[col].dropna())
            abs_z_scores = abs(z_scores)
            upper_limit = df[col].mean() + 3 * df[col].std()
            lower_limit = df[col].mean() - 3 * df[col].std()
        elif method == 'IQR':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            upper_limit = Q3 + 1.5 * IQR
            lower_limit = Q1 - 1.5 * IQR
        else:
            continue

        df_capped.loc[df_capped[col] > upper_limit, col] = upper_limit
        df_capped.loc[df_capped[col] < lower_limit, col] = lower_limit

    st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Capped outliers using {method} method.")
    return df_capped

def standardize_columns(df, columns):
    """Standardize specified columns using StandardScaler."""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Standardized columns: {', '.join(columns)}.")
    return df_scaled

def normalize_columns(df, columns):
    """Normalize specified columns using MinMaxScaler."""
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Normalized columns: {', '.join(columns)}.")
    return df_scaled

def encode_categorical(df, columns, encoding='One-Hot'):
    """
    Encode categorical columns.

    Parameters:
    - df (pd.DataFrame): The original DataFrame.
    - columns (list): List of columns to encode.
    - encoding (str): Type of encoding ('One-Hot' or 'Label').

    Returns:
    - df_encoded (pd.DataFrame): DataFrame with encoded columns.
    """
    df_encoded = df.copy()
    if encoding == 'One-Hot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns)
        for col in columns:
            if col not in st.session_state['encoding_mappings']:
                st.session_state['encoding_mappings'][col] = {}
            for cat in df[col].unique():
                encoded_col = f"{col}_{cat}"
                st.session_state['encoding_mappings'][col][encoded_col] = cat
        summary = f"Encoded categorical columns {', '.join(columns)} using One-Hot Encoding."
    elif encoding == 'Label':
        label_encoders = {}
        for col in columns:
            if col not in st.session_state['encoding_mappings']:
                st.session_state['encoding_mappings'][col] = {}
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            # Invert the mapping to get number to category
            inverted_mapping = {v: k for k, v in mapping.items()}
            st.session_state['encoding_mappings'][col].update(inverted_mapping)
        summary = f"Encoded categorical columns {', '.join(columns)} using Label Encoding."
    st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_state(summary)  # Save state with summary
    logging.info(summary)
    return df_encoded


def get_download_link(df, filename="data.csv", file_format="CSV"):
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded
    in a specific format.

    Parameters:
    - df (pd.DataFrame): The DataFrame to download.
    - filename (str): The default filename for the download.
    - file_format (str): The format of the file ('CSV', 'Excel').

    Returns:
    - str: An HTML anchor tag to download the data.
    """
    if file_format.upper() == "CSV":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    elif file_format.upper() == "EXCEL":
        towrite = BytesIO()
        df.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel</a>'
    else:
        href = ""
    return href


def download_dataframe(df, filename="data.csv", file_format="CSV"):
    """
    Displays a download button for the DataFrame in Streamlit.

    Parameters:
    - df (pd.DataFrame): The DataFrame to download.
    - filename (str): The default filename for the download.
    - file_format (str): The format of the file ('CSV', 'Excel').
    """
    if df is not None and not df.empty:
        st.markdown(get_download_link(df, filename, file_format), unsafe_allow_html=True)
    else:
        st.warning("No data available to download.")

@st.cache_data
def identify_outliers_cached(df, method='IQR', column=None, threshold=1.5):
    return identify_outliers(df, method, column, threshold)
