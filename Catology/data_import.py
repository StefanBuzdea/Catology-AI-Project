import pandas as pd


# import data from excel
def import_data_from_excel(file_path):
    df = pd.read_excel(file_path)
    return df