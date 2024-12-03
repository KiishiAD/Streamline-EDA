import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def plot_missing_values(df):
    """Plot missing values using Plotly."""
    missing = df.isnull().sum().reset_index()
    missing.columns = ['Column', 'Missing']
    fig = px.bar(missing, x='Column', y='Missing', title='Missing Values per Column')
    return fig

def plot_distribution(df, column):
    """Plot the distribution of a single column using Seaborn."""
    fig, ax = plt.subplots()
    sns.histplot(df[column].dropna(), color='green', label='Distribution', kde=True, ax=ax, stat="density", bins=30)
    ax.set_title(f"Distribution of {column}")
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.close(fig)
    return fig

def plot_outliers_scatter(df, column, outlier_indices):
    """
    Plot a scatter plot highlighting outliers in a specific column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column (str): The column to plot.
    - outlier_indices (list): Indices of the outlier data points.

    Returns:
    - fig: The generated plot figure.
    """
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=df.index, y=column, ax=ax, label='Data Points')
    if outlier_indices:
        sns.scatterplot(data=df.loc[outlier_indices], x=outlier_indices, y=column, color='red', label='Outliers', ax=ax)
    ax.set_title(f"Scatter Plot of {column} with Outliers Highlighted")
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.close(fig)
    return fig

def plot_interactive_box(df, column):
    """
    Plot an interactive box plot for a specific column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column (str): The column to plot.

    Returns:
    - fig: The generated plot figure.
    """
    fig = px.box(df, y=column, title=f"Box Plot of {column}")
    return fig



