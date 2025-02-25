from math import log2
import pandas as pd
from Catology.Normalize_Data.constants import predefined_values, expected_columns

def check_new_columns_and_values(data, new_attributes_file_path):
    with open(new_attributes_file_path, 'w') as f:
        f.write("CHECK FOR NEW ATTRIBUTES OR VALUES:\n")

        # Verificam pentru coloane noi
        columns_in_data = set(data.columns)
        new_columns = columns_in_data - expected_columns
        if new_columns:
            f.write(f"New columns detected: {new_columns}\n")
        else:
            f.write("No new columns detected.\n")

        for col, valid_values in predefined_values.items():
            if col in data.columns:
                invalid_values = data[~data[col].isin(valid_values) & data[col].notna()]
                if not invalid_values.empty:
                    f.write(f"New values detected in column '{col}': {invalid_values[col].unique()}\n")
        f.write("Check completed.\n")

def check_missing_values(data, missing_file_path):
    with open(missing_file_path, 'w') as f:
        f.write("THE MISSING VALUES ARE:\n")
        for _, row in data.iterrows():
            for col in data.columns:
                if col != 'Plus':
                    id_value = row['id']
                    if pd.isna(row[col]):
                        f.write(f"At id {id_value} we have a missing value for the attribute '{col}'\n")
                    if row[col] == 'NSP':
                        f.write(f"At id {id_value} we have 'NSP' for the attribute '{col}'\n")

def check_outliers_values(data, predefined_values, outliers_file_path):
    with open(outliers_file_path, 'w') as f:
        f.write("THE OUTLIERS VALUES ARE:\n")
        for _, row in data.iterrows():
            for col, valid_values in predefined_values.items():
                if col in data.columns:
                    if not pd.isna(row[col]) and row[col] != 'NSP':
                        if row[col] not in valid_values:
                            id_value = row['id']
                            f.write(f"At id {id_value} the attribute '{col}' has an invalid value: '{row[col]}'\n")


def check_identical_rows(data, identical_rows_file_path):
    with open(identical_rows_file_path, 'w') as f:
        f.write("IDENTICAL LINES FOUND:\n")

        data_without_specs = data.drop(columns=['id', 'Horodateur', 'Plus'])

        for i in range(len(data_without_specs)):
            for j in range(i + 1, len(data_without_specs)):
                if data_without_specs.iloc[i].equals(data_without_specs.iloc[j]):
                    id_i = data.loc[i, 'id']
                    id_j = data.loc[j, 'id']
                    f.write(f"Line with id {id_i} and Line with id {id_j} are identical: {data_without_specs.iloc[i].to_dict()}\n")



def display_instances_per_class(data):
    class_counts = data['Race'].value_counts(dropna=True)
    return class_counts

def display_distinct_values(data):
    columns_to_exclude = ['id', 'Horodateur', 'Plus']
    distinct_values_info = {}
    for column in data.columns:
        if column not in columns_to_exclude:
            clean_data = data[column].dropna().replace('NSP', pd.NA)
            # clean_data = data[column].dropna()
            value_counts = clean_data.value_counts(dropna=True)
            distinct_values_info[column] = value_counts

    return distinct_values_info


def calculate_entropy(value_counts):
    total = value_counts.sum()
    entropy = 0
    for count in value_counts:
        p_x = count / total
        entropy -= p_x * log2(p_x)
    return entropy


# Modified function to display distinct values and calculate entropy for each attribute per class
def display_distinct_values_per_class(data):
    class_distinct_values = {}
    min_entropies_per_class = {}
    breeds = data['Race'].dropna().unique()

    for breed in breeds:
        # Filter data for the current breed
        breed_data = data[data['Race'] == breed]

        # Reuse 'display_distinct_values' function for each breed subset
        breed_info = display_distinct_values(breed_data)
        class_distinct_values[breed] = breed_info

        # Calculate entropy for each attribute within this breed and find the minimum entropy, excluding 'Race'
        min_entropy = float('inf')  # Initialize with a high value
        min_entropy_column = None

        for column, value_counts in breed_info.items():
            if column in ['Race', 'Sexe']:  # Exclude 'Race' and 'Sexe' columns
                continue

            entropy = calculate_entropy(value_counts)
            if entropy < min_entropy:
                min_entropy = entropy
                min_entropy_column = column

        # Store the minimum entropy and corresponding attribute, ensuring it's not from 'Race'
        if min_entropy_column is not None:
            min_entropies_per_class[breed] = {'attribute': min_entropy_column, 'entropy': min_entropy}

    return class_distinct_values, min_entropies_per_class
