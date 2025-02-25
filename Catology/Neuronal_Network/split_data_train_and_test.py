from Catology.data_import import import_data_from_excel
from Catology.Normalize_Data.constants import output_file_path

def split_data_train_and_test():

    output_data = import_data_from_excel(output_file_path)
    shuffled_data = output_data.sample(frac=1, random_state=23).reset_index(drop=True)

    # Calculam dimensiunea setului de antrenare
    train_ratio = 0.7
    num_rows = len(shuffled_data)
    split_index = int(num_rows * train_ratio)

    # Impartim datele
    train_data = shuffled_data.iloc[:split_index]  # Primele 70% din date
    test_data = shuffled_data.iloc[split_index:]  # Restul de 30%

    return train_data, test_data