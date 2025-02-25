from Catology.Neuronal_Network.backward_propagation import backward_propagation
from Catology.Neuronal_Network.forward_propagation import forward_propagation
from Catology.Neuronal_Network.neuronal_network_main import load_and_prepare_data_cats, initialize_weights_and_bias
from Catology.Neuronal_Network.activation_and_loss_functions import cross_entropy_loss
import pandas as pd
import numpy as np
from Catology.Normalize_Data.constants import output_file_path


def classify_instance(instance, weights_and_bias, feature_columns, min_values, max_values):
    """
    Clasifică o instanță individuală folosind rețeaua neuronală.
    :param instance: Dicționar cu atributele instanței.
    :param weights_and_bias: Ponderile și biasele rețelei antrenate.
    :param feature_columns: Lista coloanelor caracteristicilor utilizate în antrenare.
    :param min_values: Valorile minime utilizate pentru normalizare.
    :param max_values: Valorile maxime utilizate pentru normalizare.
    :return: Clasa prezisă pentru instanță.
    """
    # Conversia instanței într-un DataFrame pentru procesare
    instance_df = pd.DataFrame([instance])

    # Selectăm doar coloanele relevante conform `feature_columns`
    instance_df = instance_df.loc[:, feature_columns.intersection(instance_df.columns)]

    # Reordonăm coloanele conform setului de date de antrenare
    missing_columns = set(feature_columns) - set(instance_df.columns)
    for col in missing_columns:
        instance_df[col] = 0  # Adăugăm coloanele lipsă cu valoare implicită 0

    # Reordonăm coloanele conform ordinii în `feature_columns`
    instance_df = instance_df[feature_columns]

    # Normalizare
    instance_normalized = (instance_df - min_values) / (max_values - min_values)
    instance_normalized = instance_normalized.fillna(0).to_numpy().T  # Înlocuim valorile NaN cu 0

    # Propagare înainte
    AL, _ = forward_propagation(instance_normalized, weights_and_bias)

    # Predicție
    predicted_class = np.argmax(AL, axis=0)[0]
    return predicted_class


def neuronal_network_instance_classify_test():
    # Încarcă și pregătește datele
    train_X, test_X, train_Y, test_Y = load_and_prepare_data_cats()

    # Parametrii rețelei
    input_size = train_X.shape[1]
    hidden_layers = [64, 32]
    output_size = train_Y.shape[1]
    learning_rate = 0.1
    epochs = 1000
    batch_size = 32

    # Normalizează datele
    min_values = train_X.min(axis=0)
    max_values = train_X.max(axis=0)

    # Initializează rețeaua
    weights_and_bias = initialize_weights_and_bias(input_size, hidden_layers, output_size)

    # Antrenare
    losses = []
    for epoch in range(epochs):
        permutation = np.random.permutation(train_X.shape[0])
        train_X = train_X[permutation]
        train_Y = train_Y[permutation]

        for i in range(0, train_X.shape[0], batch_size):
            X_batch = train_X[i:i + batch_size].T
            Y_batch = train_Y[i:i + batch_size].T

            # Propagare înainte
            AL, cache = forward_propagation(X_batch, weights_and_bias)

            # Calcul pierdere
            loss = cross_entropy_loss(AL, Y_batch)

            # Propagare înapoi
            weights_and_bias = backward_propagation(AL, Y_batch, cache, weights_and_bias, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoca {epoch}, Pierdere: {loss}")
            losses.append(loss)

    # Clasifică o instanță individuală
    feature_columns = pd.read_excel(output_file_path).drop(columns=["Race", "Row.names", "Horodateur", "Plus"], errors="ignore").columns
    new_instance = {
        'Sexe': 2, 'Age': 1, 'Nombre': 1, 'Logement': 2, 'Zone': 1, 'Ext': 0, 'Obs': 2,
        'Timide': 1, 'Calme': 1, 'Effrayé': 3, 'Intelligent': 4, 'Vigilant': 4, 'Perséverant': 5,
        'Affectueux': 5, 'Amical': 5, 'Solitaire': 1, 'Brutal': 2, 'Dominant': 2, 'Agressif': 3,
        'Impulsif': 4, 'Prévisible': 4, 'Distrait': 3, 'Abondance': -1, 'PredOiseau': 0, 'PredMamm': 0
    }
    predicted_class = classify_instance(new_instance, weights_and_bias, feature_columns, min_values, max_values)
    print(f"Clasa prezisă pentru instanța dată: {predicted_class}")
