import numpy as np
import pandas as pd
import openai
from Catology.Neuronal_Network.neuronal_network_main import neuronal_network
from Catology.Normalize_Data.constants import output_file_path

# Configurație pentru OpenAI API
openai.apikey = "secret-key"

ATTRIBUTE_DOMAINS = """
'Sexe': ['M', 'F', 'NSP'], # 'M', 'F', 'NSP'
'Age': ['Moinsde1', '1a2', '2a10', 'Plusde10'], # 'MoreThan1', '1to2', '2to10', 'MoreThan10'
'Nombre': {1, 2, 3, 4, 5, 6}, #no of cats in household: 1, 2, 3, 4, 5, more than 5
'Logement': ['ASB', 'AAB', 'ML', 'MI'], # 'Apartment without balcony', 'Apartment with balcony or terrace',  'House in a subdivision',  'Individual house'
'Zone': ['U', 'PU', 'R'], # Urban, Suburban, Rural
'Ext': [0, 1, 2, 3, 4], # how much time your cat spends each day outdoors
'Obs': [0, 1, 2, 3], # How much time do you spend each day with your cat
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
'Abondance': {1, 2, 3, -1}, # abundance of natural areas: Low (1), Medium (2), High (3), NSP
'PredOiseau': [0, 1, 2, 3, 4], # frequency your cat captures birds
'PredMamm': [0, 1, 2, 3, 4], # frequency your cat captures mammals
"""

def extract_attribute_weights(weights_and_bias):
    """
    Extrage ponderile medii pentru fiecare caracteristică pentru fiecare clasă.
    :param weights_and_bias: Dicționar cu ponderi și biase ale rețelei neuronale.
    :return: Dicționar cu ponderi normalizate pentru fiecare clasă și caracteristică.
    """
    L = len(weights_and_bias) // 2  # Numărul de straturi
    output_weights = weights_and_bias[f"W{L}"]  # Strat final de ieșire

    # Calculăm ponderea medie a fiecărei caracteristici pentru fiecare clasă
    attribute_weights = {}
    for class_idx in range(output_weights.shape[0]):
        class_weights = output_weights[class_idx, :]
        normalized_weights = class_weights / np.sum(np.abs(class_weights))
        attribute_weights[class_idx] = normalized_weights

    return attribute_weights

def process_weights_for_relevance(attribute_weights, attributes, epsilon=0.05):
    """
    Prelucrează ponderile atributelor pentru a le face interpretabile și selectează atributele relevante.
    :param attribute_weights: Ponderile normalizate pentru fiecare clasă.
    :param attributes: Lista atributelor disponibile.
    :param epsilon: Pragul peste care un atribut este considerat semnificativ (în procente).
    :return: Dicționar cu atributele relevante pentru fiecare clasă.
    """
    processed_weights = {}
    for class_idx, weights in attribute_weights.items():
        processed_weights[class_idx] = []
        for attr, weight in zip(attributes, weights):
            abs_weight = abs(weight) * 100  # Convertim la procent pozitiv
            if abs_weight > epsilon:  # Considerăm doar atributele semnificative
                if weight < 0:
                    processed_weights[class_idx].append((f"Not {attr}", abs_weight))
                else:
                    processed_weights[class_idx].append((attr, abs_weight))

    return processed_weights

def normalize_relevant_attributes(relevant_attributes):
    """
    Normalizează ponderile atributele relevante pentru fiecare clasă astfel încât suma să fie 100%.
    :param relevant_attributes: Dicționar cu atributele relevante pentru fiecare clasă.
    :return: Dicționar cu atributele normalizate pentru fiecare clasă.
    """
    normalized_attributes = {}
    for class_idx, attrs in relevant_attributes.items():
        total_weight = sum(weight for _, weight in attrs)
        if total_weight > 0:
            normalized_attributes[class_idx] = [(attr, (weight / total_weight) * 100) for attr, weight in attrs]
        else:
            normalized_attributes[class_idx] = attrs  # Dacă suma e 0, nu normalizăm

    return normalized_attributes

def get_most_frequent_value_for_attribute(data, target_class, target_attribute):
    """
    Returnează valoarea cu cea mai mare frecvență pentru un atribut specific dintr-o clasă specificată.
    :param data: DataFrame-ul cu datele complete.
    :param target_class: Numele clasei (rasa) pentru care se calculează valoarea.
    :param target_attribute: Atributul pentru care se caută valoarea cea mai frecventă.
    :return: Valoarea cea mai frecventă a atributului în clasa specificată.
    """
    # Filtrăm datele pentru clasa specificată
    class_data = data[data['Race'] == target_class]

    # Verificăm dacă atributul există în date
    if target_attribute not in data.columns:
        raise ValueError(f"Atributul '{target_attribute}' nu există în date.")

    # Verificăm dacă există valori pentru atributul specificat
    if class_data[target_attribute].dropna().empty:
        raise ValueError(f"Nu există valori pentru atributul '{target_attribute}' în clasa '{target_class}'.")

    # Calculăm valoarea cea mai frecventă
    most_frequent_value = class_data[target_attribute].mode().iloc[0]

    return most_frequent_value

def extract_proportions_with_frequent_values():
    # Extragem ponderile atributelor
    weights_and_bias = np.load("weights_and_bias.npy", allow_pickle=True).item()  # Încarcă fișierul

    attribute_weights = extract_attribute_weights(weights_and_bias)

    # Atributele disponibile (se extrag din coloanele datelor)
    data = pd.read_excel(output_file_path)

    ignore_columns = {'Row.names', 'Horodateur', 'Plus', 'Race', 'id'}
    attributes = [col for col in data.columns if col not in ignore_columns]

    # Prelucrăm ponderile pentru relevanță
    epsilon = 5  # Prag pentru atributele semnificative (în procente)
    relevant_attributes = process_weights_for_relevance(attribute_weights, attributes, epsilon=epsilon)

    # Normalizează ponderile astfel încât suma să fie 100%
    normalized_attributes = normalize_relevant_attributes(relevant_attributes)

    # Obține numele claselor din date
    # Valorile unice din coloana 'Race', fără NSP
    class_names = [name for name in data['Race'].unique() if name != 'NSP']

    # Combina ponderile cu valorile cele mai frecvente
    combined_results = {}
    for class_idx, attrs in normalized_attributes.items():
        # Verificăm dacă indexul există în `class_names`
        if class_idx >= len(class_names):
            continue  # Sărim peste clasele care nu au corespondență în `class_names`

        target_class = class_names[class_idx]  # Asociază indexul cu numele clasei
        combined_results[target_class] = []
        for attr, weight in attrs:
            attribute_name = attr.split()[-1] if "Not" in attr else attr  # Extragem numele atributului
            most_frequent_value = get_most_frequent_value_for_attribute(data, target_class, attribute_name)
            combined_results[target_class].append((attr, weight, most_frequent_value))

    return combined_results

def generate_description_for_class(target_class):
    """
    Generează o descriere folosind OpenAI API pentru o clasă specificată.
    :param target_class: Numele clasei pentru care se generează descrierea.
    """
    combined_results = extract_proportions_with_frequent_values()

    if target_class not in combined_results:
        raise ValueError(f"Clasa '{target_class}' nu există în rezultatele combinate.")

    attributes_info = combined_results[target_class]
    prompt = "Bazându-te strict pe următoarele informații, generează o propoziție despre caracteristicile principale ale rasei. Nu te lega de aspecte fizice, precum blana sau ochi. Ai grija la negatii(negatia unei valori mici inseamna o valoare mare). Nu-mi spune valorile atributelor ca numere, interpreteaza-le in limbaj natural:\n"
    for attr, _, value in attributes_info:
        prompt += f"- {attr}: valoare frecventă '{value}'\n"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Ești un expert în descrierea raselor de pisici."},
                  {"role": "user", "content": prompt}]
    )

    return response['choices'][0]['message']['content']


def generate_comparison_between_classes(class1, class2):
    """
    Generează o comparație folosind OpenAI API între două clase specificate.
    :param class1: Numele primei clase pentru comparație.
    :param class2: Numele celei de-a doua clase pentru comparație.
    """
    combined_results = extract_proportions_with_frequent_values()

    if class1 not in combined_results or class2 not in combined_results:
        raise ValueError(f"Una sau ambele clase specificate ('{class1}', '{class2}') nu există în rezultatele combinate.")

    attributes_info1 = combined_results[class1]
    attributes_info2 = combined_results[class2]

    prompt = "Bazându-te strict pe următoarele informații, realizează o comparație între cele două rase, evidențiind diferențele și asemănările lor. Ai grijă să interpretezi valorile în limbaj natural și să folosești negări unde este cazul. Nu-mi spune valorile atributelor ca numere, ci descrie-le:"
    prompt += f"Pentru rasa {class1}:"
    for attr, _, value in attributes_info1:
        prompt += f"- {attr}: valoare frecventă '{value}'"

    prompt += f"\nPentru rasa {class2}:"
    for attr, _, value in attributes_info2:
        prompt += f"- {attr}: valoare frecventă '{value}'"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Ești un expert în descrierea raselor de pisici."},
                  {"role": "user", "content": prompt}]
    )

    return response['choices'][0]['message']['content']
