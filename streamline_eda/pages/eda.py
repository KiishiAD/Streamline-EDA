import streamlit as st
import pandas as pd
from utils import (
    get_data_preview,
    initialize_session_state,
    get_numerical_columns,
    download_dataframe
)
from pygwalker.api.streamlit import StreamlitRenderer  # Correct Import

import matplotlib.pyplot as plt
import seaborn as sns

def app():
    st.title("📊 Exploratory Data Analysis (EDA)")

    # Initialize session state variables
    initialize_session_state()

    if st.session_state['df'] is None:
        st.warning("Please upload or clean data first in the 'Upload Data' and 'Data Cleaning' sections.")
        return

    df = st.session_state['df']

    st.write("### 📈 First Few Rows")
    st.dataframe(df.head(20))

    st.write("### 🔍 Current Columns:")
    st.write(df.columns.tolist())

    st.write("### 📝 Data Types")
    data_types = pd.DataFrame(df.dtypes, columns=["Data Type"])
    data_types.index.name = "Column"
    st.dataframe(data_types)

    st.write("### 📋 Missing Values")
    # Existing Missing Values Plot (Using Plotly)
    try:
        import plotly.express as px
        missing = df.isnull().sum().reset_index()
        missing.columns = ['Column', 'Missing']
        fig = px.bar(missing, x='Column', y='Missing', title='Missing Values per Column')
        st.plotly_chart(fig)
    except ImportError:
        st.error("Plotly is not installed. Please install it using `pip install plotly`.")

    st.write("### 📊 Descriptive Statistics")
    st.dataframe(df.describe(include='all').transpose())

    st.markdown("---")

    # 🔗 Correlation Matrix Heatmap
    st.write("### 🔗 Correlation Matrix Heatmap")
    try:
        numerical_cols = get_numerical_columns(df)
        if len(numerical_cols) < 2:
            st.warning("Not enough numerical columns to compute correlation.")
        else:
            corr_matrix = df[numerical_cols].corr()

            # Set up the matplotlib figure
            plt.figure(figsize=(10, 8))

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            # Draw the heatmap with the mask and correct aspect ratio
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, square=True,
                        linewidths=.5, cbar_kws={"shrink": .5})
            plt.title("Correlation Matrix Heatmap", fontsize=16)

            # Display the heatmap in Streamlit
            st.pyplot(plt)
    except ImportError:
        st.error(
            "Seaborn and Matplotlib are required for the Correlation Heatmap. Please install them using `pip install seaborn matplotlib`.")
    except Exception as e:
        st.error(f"An error occurred while generating the Correlation Heatmap: {e}")

    st.markdown("---")

    # Integrate PyGWalker for Interactive EDA
    st.write("### 🔍 Interactive Data Exploration")
    try:
        # Initialize StreamlitRenderer without caching
        renderer = StreamlitRenderer(df)  # Removed spec parameter

        st.write("You can explore your data interactively below:")
        renderer.explorer()
    except Exception as e:
        st.error(f"An error occurred while initializing PyGWalker: {e}")

    st.markdown("---")

    st.write("### 📥 Export Current Data")
    download_dataframe(df, filename="eda_data.csv", file_format="CSV")
