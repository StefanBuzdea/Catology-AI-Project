from Catology.Language_Processing.extract_atribute_from_description import extract_attributes, complete_instance
from Catology.Neuronal_Network.backward_propagation import backward_propagation
from Catology.Neuronal_Network.forward_propagation import forward_propagation
from Catology.Neuronal_Network.neuronal_network_main import load_and_prepare_data_cats, initialize_weights_and_bias
from Catology.Neuronal_Network.activation_and_loss_functions import cross_entropy_loss
import pandas as pd
import numpy as np
from Catology.Normalize_Data.constants import output_file_path, conversion_map, expected_columns_eng


def classify_instance(instance, weights_and_bias):
    """
    Clasifică o instanță individuală folosind rețeaua neuronală.
    :param instance: Dicționar cu atributele instanței completate (procesată).
    :param weights_and_bias: Ponderile și biasele rețelei antrenate.
    :return: Clasa prezisă pentru instanță.
    """

    columns_to_remove = {'Row.names', 'Horodateur', 'Race', 'Plus'}
    instance = {k: v for k, v in instance.items() if k not in columns_to_remove}

    translated_instance = {expected_columns_eng.get(k, k): v for k, v in instance.items()}
    print("Complete instance: ", translated_instance)

    # Conversia instanței într-un array NumPy pentru propagare
    instance_array = np.array(list(instance.values()), dtype=float).reshape(-1, 1)

    # Propagare înainte
    AL, _ = forward_propagation(instance_array, weights_and_bias)

    # Predicție
    predicted_class = np.argmax(AL, axis=0)[0]
    return predicted_class


def apply_nn_to_instance():
    weights_and_bias = np.load("weights_and_bias.npy", allow_pickle=True).item()

    # Solicitați utilizatorului să introducă o descriere a pisicii
    print("Describe the cat to be classified:")
    description = input("Description: ")

    # Extragem și completăm instanța
    dataset = pd.read_excel(output_file_path).drop(columns=["Row.names", "Horodateur", "Race", "Plus"], errors="ignore")
    incomplete_instance = extract_attributes(description)
    if incomplete_instance:
        complete_instance_data = complete_instance(dataset, incomplete_instance)

        # Clasificăm instanța completă
        predicted_class = classify_instance(complete_instance_data, weights_and_bias)

        # Convertim clasa numerică în valoarea string folosind mapping-ul invers
        reverse_race_map = {v: k for k, v in conversion_map['Race'].items()}
        predicted_class_str = reverse_race_map.get(predicted_class, "Unknown")

        print(f"Predicted class (numeric): {predicted_class}")
        print(f"Predicted class (string): {predicted_class_str}")
    else:
        print("Could not generate a new complete instance for classification")
