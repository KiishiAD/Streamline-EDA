# Streamline-EDA


## Table of Contents

- [Data Cleaning & Exploratory Data Analysis (EDA) App](#-data-cleaning--exploratory-data-analysis-eda-app)
  - [Features](#-features)
    - [ Upload Data](#-upload-data)
    - [Data Cleaning](#-data-cleaning)
    - [Data Validation](#-data-validation)
    - [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
    - [Export Data](#-export-data)
    - [Timeline History](#-timeline-history)
  - [Installation](#-installation)

## Features

Streamline-EDA is designed to streamline the data analysis workflow. Below are the key features:

### Upload Data and View your Data

Easily upload your datasets in various formats including CSV, Excel, JSON, and Parquet and output summary stats.

![Upload Data](images/data_preview.png)
![Upload Data](images/dataset_overview.png)

### Data Cleaning

Perform essential data cleaning operations such as handling missing values, removing duplicates, detecting and managing outliers, and encoding categorical variables.

- **Handle Missing Values:** Drop or fill missing data using different strategies.
- **Remove Duplicates:** Identify and eliminate duplicate records.
- **Outlier Detection & Handling:** Use Z-Score or IQR methods to detect and remove or cap outliers.
- **Data Transformation:** Standardize, normalize, and encode categorical variables.

![Data Cleaning](images/data_cleaning_1.png)
![Data Cleaning](images/data_cleaning_2.png)

### Data Validation
- **Set Rules:** Set strict rules records should follow helping to fix ensure reliable consistency
  
  ![Data Validation](images/data_validation_1.png)
  ![Data Validation](images/data_validation_2.png)

### Exploratory Data Analysis (EDA)

Gain insights into your data with comprehensive EDA tools, including data previews, descriptive statistics, correlation heatmaps, and interactive visualizations using PyGWalker.

- **Data Preview:** View the first few rows and understand data types.
- **Descriptive Statistics:** Get summary statistics for numerical and categorical columns.
- **Correlation Matrix Heatmap:** Visualize correlations between numerical variables.
- **Interactive Data Exploration:** Use PyGWalker for dynamic and interactive data exploration.

![EDA](images/eda.png)

### Export Data

Export the current state of your DataFrame at any point in the analysis. Choose between CSV and Excel formats for your downloads.


### Timeline History

Track all the transformations and changes made to your dataset with a comprehensive timeline. Revert to any previous state effortlessly to ensure data integrity.

![Timeline History](images/timeline.png)

---

## Installation

Follow these steps to set up the application locally:

1. **Clone the Repository**

- Git clone the project repo: https://github.com/KiishiAD/Streamline-EDA.git

2. **Create a virtual env and install the libraries in the requirements.txt file**
- python -m venv venv
- pip install -r requirements.txt

3. **Run the app in the project directory**
- Streamlit run main.py
