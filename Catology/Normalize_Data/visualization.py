import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def display_value_distributions(df):
    # List of columns to exclude
    excluded_columns = ['id', 'Horodateur', 'Plus']

    for column in df.columns:
        # Skip excluded columns
        if column in excluded_columns:
            continue

        plt.figure(figsize=(10, 5))

        if pd.api.types.is_numeric_dtype(df[column]):
            # For numeric columns, show a histogram
            sns.histplot(df[column].dropna(), kde=True)
            plt.title(f'Distribution of {column} (Numeric)')
        else:
            # For categorical columns, show a count plot
            sns.countplot(y=df[column], order=df[column].value_counts().index)
            plt.title(f'Distribution of {column} (Categorical)')

        plt.show()


# Function to compute and show heatmap between 'Race' and other attributes
def plot_heatmap_race_correlation(df):
    # Copy the DataFrame
    df_encoded = df.copy()

    # Columns to exclude
    excluded_columns = ['id']

    # Mixed-type columns that need to be encoded
    mixed_columns = ['Age', 'Nombre', 'Logement', 'Zone', 'Abondance']

    # One-hot encode the 'Race' column (each race becomes a new column)
    df_race_encoded = pd.get_dummies(df_encoded['Race'], prefix='Race')

    # Encode the mixed-type columns (convert categorical to numeric)
    for col in mixed_columns:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype('category').cat.codes

    # Select only numeric columns (excluding 'id')
    numeric_columns = df_encoded.select_dtypes(include=[np.number]).drop(columns=excluded_columns, errors='ignore')

    # Concatenate the one-hot encoded 'Race' with the other numeric columns
    df_combined = pd.concat([df_race_encoded, numeric_columns], axis=1)

    # Compute correlation matrix between the one-hot encoded 'Race' and other numeric attributes
    correlation_matrix = df_combined.corr().loc[df_race_encoded.columns, numeric_columns.columns]

    # Plot the heatmap with larger cells and smaller font size for annotations
    plt.figure(figsize=(14, 10))  # Increase the figure size to make cells larger
    sns.heatmap(correlation_matrix, annot=True, fmt=".3f", cmap='coolwarm', cbar=True,
                annot_kws={"size": 8})  # Limit to 3 decimals and reduce font size of numbers

    plt.title('Correlation between Race categories and Numeric Attributes')
    plt.xlabel('Attributes')
    plt.ylabel('Race Categories')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.show()


# Function to plot distribution of each attribute both as chart and pie chart
def plot_distribution_with_pie(df):
    excluded_columns = ['id', 'Horodateur', 'Plus']  # Columns to exclude

    for column in df.columns:
        # Skip excluded columns
        if column in excluded_columns:
            continue

        plt.figure(figsize=(14, 7))

        # Plot a bar chart or histogram depending on the type
        plt.subplot(1, 2, 1)
        if pd.api.types.is_numeric_dtype(df[column]):
            sns.histplot(df[column].dropna(), kde=False)
            plt.title(f'{column} Distribution (Bar)')
        else:
            sns.countplot(y=df[column], order=df[column].value_counts().index)
            plt.title(f'{column} Distribution (Bar)')

        # Plot a pie chart
        plt.subplot(1, 2, 2)
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column].dropna().value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('husl'))
        else:
            df[column].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('husl'))
        plt.title(f'{column} Distribution (Pie)')
        plt.ylabel('')  # Hide y-label for pie chart

        plt.show()