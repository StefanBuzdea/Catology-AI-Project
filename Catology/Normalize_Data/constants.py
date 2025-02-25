from pathlib import Path

#expected columns
expected_columns = {
    "id", "Horodateur", "Sexe", "Age", "Race", "Nombre", "Logement", "Zone", "Ext", "Obs",
    "Timide", "Calme", "Effrayé", "Intelligent", "Vigilant", "Perséverant", "Affectueux",
    "Amical", "Solitaire", "Brutal", "Dominant", "Agressif", "Impulsif", "Prévisible",
    "Distrait", "Abondance", "PredOiseau", "PredMamm", "Plus"
}

expected_columns_eng = {
    "id": "id",
    "Horodateur": "Timestamp",
    "Sexe": "Gender",
    "Age": "Age",
    "Race": "Breed",
    "Nombre": "Number",
    "Logement": "Housing",
    "Zone": "Area",
    "Ext": "OutdoorTime",
    "Obs": "ObservationTime",
    "Timide": "Shy",
    "Calme": "Calm",
    "Effrayé": "Scared",
    "Intelligent": "Intelligent",
    "Vigilant": "Vigilant",
    "Perséverant": "Perseverant",
    "Affectueux": "Affectionate",
    "Amical": "Friendly",
    "Solitaire": "Solitary",
    "Brutal": "Brutal",
    "Dominant": "Dominant",
    "Agressif": "Aggressive",
    "Impulsif": "Impulsive",
    "Prévisible": "Predictable",
    "Distrait": "Distracted",
    "Abondance": "Abundance",
    "PredOiseau": "BirdPredationFrequency",
    "PredMamm": "MammalPredationFrequency",
    "Plus": "Additional"
}


#domain values
predefined_values = {
    'Sexe': ['M', 'F', 'NSP'],
    'Age': ['Moinsde1', '1a2', '2a10', 'Plusde10'],
    'Race': ['BEN', 'SBI', 'BRI', 'CHA', 'EUR', 'MCO', 'PER', 'RAG','SAV', 'SPH', 'ORI', 'TUV', 'NR', 'Autre', 'NSP'],
    'Nombre': ['1', '2', '3', '4', '5', 'Plusde5'],
    'Logement': ['ASB', 'AAB', 'ML', 'MI'],
    'Zone': ['U', 'PU', 'R'],
    'Ext': [0, 1, 2, 3, 4],
    'Obs': [0, 1, 2, 3],
    'Abondance': ['1', '2', '3', 'NSP'],
    'PredOiseau': [0, 1, 2, 3, 4],
    'PredMamm': [0, 1, 2, 3, 4],
    'Timide': [1, 2, 3, 4, 5],
    'Calme': [1, 2, 3, 4, 5],
    'Effrayé': [1, 2, 3, 4, 5],
    'Intelligent': [1, 2, 3, 4, 5],
    'Vigilant': [1, 2, 3, 4, 5],
    'Perséverant': [1, 2, 3, 4, 5],
    'Affectueux': [1, 2, 3, 4, 5],
    'Amical': [1, 2, 3, 4, 5],
    'Solitaire': [1, 2, 3, 4, 5],
    'Brutal': [1, 2, 3, 4, 5],
    'Dominant': [1, 2, 3, 4, 5],
    'Agressif': [1, 2, 3, 4, 5],
    'Impulsif': [1, 2, 3, 4, 5],
    'Prévisible': [1, 2, 3, 4, 5],
    'Distrait': [1, 2, 3, 4, 5],
}

#conversion map

conversion_map = {
    'Sexe': {'M': 1, 'F': 2, 'NSP': -1},
    'Age': {'Moinsde1': 1, '1a2': 2, '2a10': 3, 'Plusde10': 4},
    'Race': {'BEN': 1, 'SBI': 2, 'BRI': 3, 'CHA': 4, 'EUR': 5, 'MCO': 6, 'PER': 7, 'RAG': 8, 'SAV': 9, 'SPH': 10,
             'ORI': 11, 'TUV': 12, 'NR': 13, 'Autre': 14, 'NSP': -1},
    'Nombre': {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'Plusde5': 6},
    'Logement': {'ASB': 1, 'AAB': 2, 'ML': 3, 'MI': 4},
    'Zone': {'U': 1, 'PU': 2, 'R': 3},
    'Abondance': {'1': 1, '2': 2, '3': 3, 'NSP': -1}
}


translated_attributes_fr = {
    'Sexe': {'M': 1, 'F': 2, 'NSP': -1},
    'Age': {'MoreThan1': 1, '1to2': 2, '2to10': 3, 'MoreThan10': 4},
    'Nombre': {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'MoreThan5': 6}, #no ofcats in household
    'Logement': {'Apartment without balcony': 1, 'Apartment with balcony or terrace': 2,
                 'House in a subdivision': 3, 'Individual house': 4},
    'Zone': {'Urban': 1, 'Suburban': 2, 'Rural': 3},
    'Ext': [0, 1, 2, 3, 4], # how much time your cat spend each day outdoors
    'Obs': [0, 1, 2, 3], # How much time do you spend each day with you cat
    'Timide': [1, 2, 3, 4, 5],
    'Calme': [1, 2, 3, 4, 5],
    'Effrayé': [1, 2, 3, 4, 5],
    'Intelligent': [1, 2, 3, 4, 5],
    'Vigilant': [1, 2, 3, 4, 5],
    'Perséverant': [1, 2, 3, 4, 5],
    'Affectueux': [1, 2, 3, 4, 5],
    'Amical': [1, 2, 3, 4, 5],
    'Solitaire': [1, 2, 3, 4, 5],
    'Brutal': [1, 2, 3, 4, 5],
    'Dominant': [1, 2, 3, 4, 5],
    'Agressif': [1, 2, 3, 4, 5],
    'Impulsif': [1, 2, 3, 4, 5],
    'Prévisible': [1, 2, 3, 4, 5],
    'Distrait': [1, 2, 3, 4, 5],
    'Abondance': {'1': 1, '2': 2, '3': 3, 'NSP': -1}, # would you say that abundance of natural areas is Low around you
    'PredOiseau': [0, 1, 2, 3, 4], # frequency your cat capture birds
    'PredMamm': [0, 1, 2, 3, 4], # frequency your cat capture mammals
}



# Current Directory
CURRENT_DIR = Path(__file__).resolve().parent

# output files paths
missing_file_path = CURRENT_DIR / 'missing_values.txt'
outliers_file_path = CURRENT_DIR / 'outliers_values.txt'
identical_rows_file_path = CURRENT_DIR / 'identical_rows.txt'
data_analysis_file = CURRENT_DIR / 'cat_data_analysis.txt'
new_attributes_file_path = CURRENT_DIR / 'new_attributes.txt'

# Parent Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# excel paths
file_path = BASE_DIR / 'OutputData2.xlsx'
output_file_path = BASE_DIR / 'dataset.xlsx'
path_for_rn = BASE_DIR / 'OutputData.xlsx'
# excel paths for NN
training_data_file_path = BASE_DIR / 'TrainData.xlsx'
test_data_file_path = BASE_DIR / 'TestData.xlsx'
