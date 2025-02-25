import numpy as np
from deep_translator import GoogleTranslator
import pandas as pd

from Catology.Normalize_Data.constants import predefined_values, conversion_map
from Catology.Normalize_Data.constants import missing_file_path, outliers_file_path, identical_rows_file_path, \
    new_attributes_file_path
from Catology.Normalize_Data.data_analysis import check_missing_values, check_outliers_values, check_identical_rows
from Catology.Normalize_Data.data_analysis import check_new_columns_and_values

pd.set_option('future.no_silent_downcasting', True) # for fillna warning ignoring


# translate the 'Plus' column from French to English
def translate_plus_column(data):
    translator = GoogleTranslator(source='fr', target='en')

    # Apply the translation to the 'Plus' column
    data['Plus'] = data['Plus'].apply(lambda x: translator.translate(x) if isinstance(x, str) else x)
    return data


def replace_nsp_with_distribution(data, conversion_map):
    """
    Inlocuieste valorile `NSP` din coloanele definite in `conversion_map` utilizand distributia atributelor pe clase
    """
    for column, mapping in conversion_map.items():
        if column in data.columns:
            for race in data['Race'].unique():
                # Filtram datele pentru clasa curenta
                class_data = data[(data['Race'] == race) & (data[column] != 'NSP')]

                # Calculeaza distributia valorilor non-NSP pentru atribut
                value_counts = class_data[column].value_counts(normalize=True)
                value_probs = value_counts.to_dict()

                # indeparteaza valoarea `NSP` din distributie daca exista
                value_probs.pop('NSP', None)

                # Daca nu exista alte valori valide, sa continuam
                if not value_probs:
                    continue

                # Converteste valorile din distributie in numere
                numeric_probs = {mapping.get(k, k): v for k, v in value_probs.items()}

                # Normalizeaza probabilitatile pentru siguranta
                total_prob = sum(numeric_probs.values())
                if total_prob == 0:  # Daca probabilitatile sunt invalide, continuam
                    continue
                normalized_probs = {k: v / total_prob for k, v in numeric_probs.items()}

                # inlocuieste `NSP` cu o valoare aleatoare bazata pe distributie
                nsp_indices = data[(data['Race'] == race) & (data[column] == 'NSP')].index
                if len(nsp_indices) > 0:
                    random_values = np.random.choice(
                        list(normalized_probs.keys()),
                        size=len(nsp_indices),
                        p=list(normalized_probs.values())
                    )
                    data.loc[nsp_indices, column] = random_values

            # Aplica conversia finala (daca exista inca valori in text)
            data[column] = data[column].map(mapping).fillna(data[column])

    return data


def converts_and_validations(data):

    # data = data.fillna({
    #     'id': -1,
    #     'TimeDate': '2024-01-01 12:00:00',
    #     'Sex': 'M',  # sau 'F', în funcție de logică
    #     'Age': 'Moinsde1',  # valoare implicită pentru Age
    #     'Breed': 'NSP',  # valoare implicită pentru Breed
    #     'Number': '1',  # valoare implicită pentru Number
    #     'Housing': 'ASB',  # valoare implicită pentru Housing
    #     'Zone': 'U',  # valoare implicită pentru Zone
    #     'OutdoorTime': '1',  # valoare implicită pentru OutdoorTime
    #     'DailyTime': '1',  # valoare implicită pentru DailyTime
    #     'Abundance': '1',  # valoare implicită pentru Abundance
    #     'BirdCaptureFrequency': '1',  # valoare implicită pentru BirdCaptureFrequency
    #     'MammalCaptureFrequency': '1',  # valoare implicită pentru MammalCaptureFrequency
    #     'Shy': 1,  # valoare implicită pentru Shy
    #     'Calm': 1,  # valoare implicită pentru Calm
    #     'Scared': 1,  # valoare implicită pentru Scared
    #     'Intelligent': 1,  # valoare implicită pentru Intelligent
    #     'Vigilant': 1,  # valoare implicită pentru Vigilant
    #     'Perseverant': 1,  # valoare implicită pentru Perseverant
    #     'Affectionate': 1,  # valoare implicită pentru Affectionate
    #     'Friendly': 1,  # valoare implicită pentru Friendly
    #     'Solitary': 1,  # valoare implicită pentru Solitary
    #     'Brutal': 1,  # valoare implicită pentru Brutal
    #     'Dominant': 1,  # valoare implicită pentru Dominant
    #     'Aggressive': 1,  # valoare implicită pentru Aggressive
    #     'Impulsive': 1,  # valoare implicită pentru Impulsive
    #     'Predictable': 1,  # valoare implicită pentru Predictable
    #     'Distracted': 1,  # valoare implicită pentru Distracted
    #     'Plus': ''  # valoare implicită pentru Plus
    # })

    # dropping Row.names column and adding a good id
    if 'Row.names' in data.columns:
        data = data.drop(columns=['Row.names'])

    data['id'] = range(1, len(data) + 1)
    # putting id in the first position
    data = data[['id'] + [col for col in data.columns if col != 'id']]
    # outliers
    # check_outliers_values(data, predefined_values, outliers_file_path)
    # missing values
    # check_missing_values(data, missing_file_path)
    # check identical rows
    # check_identical_rows(data, identical_rows_file_path) #UNCOMMENT FOR CHECKING IDENTICAL
    # check new columns and values
    # check_new_columns_and_values(data, new_attributes_file_path)
    # translating Plus column from french to english
    data = translate_plus_column(data) # UNCOMMENT FOR TRANSLATING PLUS
    return data

#
# def convert_to_numeric(data):
#     # inlocuieste valorile `NSP` folosind distributiile calculate
#     data = replace_nsp_with_distribution(data, conversion_map)
#
#     # Aplica restul conversiilor numerice
#     for col, mapping in conversion_map.items():
#         if col in data.columns:
#             data[col] = data[col].map(mapping).fillna(data[col])
#
#     # Normalizeaza coloanele numerice
#     numeric_columns = data.drop(columns=["id"], errors="ignore").select_dtypes(include=[np.number]).columns
#     for col in numeric_columns:
#         col_min = data[col].min()
#         col_max = data[col].max()
#         if col_max != col_min:  # Evita divizarea la 0
#             data[col] = (data[col] - col_min) / (col_max - col_min)
#         else:
#             data[col] = 0  # Daca toate valorile sunt identice, le seteaza pe 0
#
#     return data