import pandas as pd
import openai

# Configurare OpenAI API
from Catology.Normalize_Data.constants import output_file_path, conversion_map, expected_columns_eng

openai.api_key = "secret-key"

def extract_attributes(description):
    """
    Extrage atribute relevante dintr-o descriere în limbaj natural folosind OpenAI API.
    """
    prompt = (
        "You are a helpful assistant. For the following description in English, "
        "extract relevant attributes and map them to their appropriate domain. "
        "Use the following attribute domains to guide your mapping and return only the attribute names followed by their values."
        "Provide the output as a dictionary format (only numbers as values, not strings) ready for use in Python:"
        """Attributes domains:
        'Sexe': {1, 2, -1}, # 'M', 'F', 'NSP'
        'Age': {1, 2, 3, 4}, # 'MoreThan1', '1to2', '2to10', 'MoreThan10'
        'Nombre': {1, 2, 3, 4, 5, 6}, #no ofcats in household: 1, 2, 3, 4, 5, more than 5
        'Logement': {1, 2, 3, 4}, # 'Apartment without balcony', 'Apartment with balcony or terrace',  'House in a subdivision',  'Individual house'
        'Zone': {1, 2, 3}, # Urban, Suburban, Rural
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
        'Abondance': {1, 2, 3, -1}, # would you say that abundance of natural areas is Low around you: 1, 2, 3, NSP
        'PredOiseau': [0, 1, 2, 3, 4], # frequency your cat capture birds
        'PredMamm': [0, 1, 2, 3, 4], # frequency your cat capture mammals
        """
        "\n\nDescription: " + description
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )

    # Salvăm rezultatul într-un dicționar și îl afișăm
    response_content = response['choices'][0]['message']['content'].strip()
    try:
        incomplete_instance = eval(response_content)

        translated_instance = {expected_columns_eng.get(k, k): v for k, v in incomplete_instance.items()}

        print("Incomplete Instance:", translated_instance)
        return incomplete_instance
    except Exception as e:
        print("Error processing response:", e)
        return {}

def complete_instance(dataset, incomplete_instance):
    """
    Completează o instanță incompletă cu cele mai comune valori din dataset,
    păstrând ordinea atributelor din dataset.

    :param dataset: DataFrame-ul care conține datele originale.
    :param incomplete_instance: Instanța incompletă sub formă de dicționar (JSON-like).
    :return: Instanța completată sub formă de dicționar.
    """
    # Determinăm cea mai comună valoare pentru fiecare atribut
    most_common_values = dataset.mode().iloc[0].to_dict()

    # Începem cu instanța incompletă
    complete_instance = incomplete_instance.copy()

    # Creăm instanța finală în ordinea atributelor datasetului
    ordered_instance = {}
    for attribute in dataset.columns:
        # Verificăm dacă atributul este deja în instanța incompletă
        if attribute in complete_instance:
            # Aplicăm mapping-ul pentru valorile din incomplete_instance
            if attribute in conversion_map:
                value = complete_instance[attribute]
                complete_instance[attribute] = conversion_map[attribute].get(value, value)

            ordered_instance[attribute] = complete_instance[attribute]
        else:
            # Dacă lipsește și există valoarea 'NSP' în dataset, folosim -1
            if -1 in dataset[attribute].values:
                ordered_instance[attribute] = -1
            else:
                # Dacă lipsește, folosim valoarea cea mai comună
                ordered_instance[attribute] = most_common_values.get(attribute, None)

            # Aplicăm mapping-ul și pentru valori implicite
            if attribute in conversion_map:
                value = ordered_instance[attribute]
                ordered_instance[attribute] = conversion_map[attribute].get(value, value)

    return ordered_instance

#
# # Exemplu de test
# if __name__ == "__main__":
#
#     try:
#         dataset = pd.read_excel(output_file_path)  # Înlocuiește cu pd.read_csv dacă e un fișier CSV
#     except Exception as e:
#         print(f"Error reading dataset from {output_file_path}: {e}")
#         exit(1)
#
#     # Descriere text pentru extragerea atributelor
#     description = "My cat is very timid, calm, and affectionate. She is intelligent, vigilant, and spends most of the day outdoors in a rural area, but she only catches birds occasionally and is not aggressive."
#
#     # Extragem atributele
#     incomplete_instance = extract_attributes(description)
#
#     # Apelăm funcția pentru a completa instanța
#     if incomplete_instance:
#         result = complete_instance(dataset, incomplete_instance)
#         # Afișăm rezultatul
#         print("Instanța completată:", result)  # Rezultatul completat
