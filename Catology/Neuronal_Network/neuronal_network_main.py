import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from Catology.Neuronal_Network.activation_and_loss_functions import cross_entropy_loss
from Catology.Neuronal_Network.backward_propagation import backward_propagation
from Catology.Neuronal_Network.evaluate_performance import print_train_data_accuracy, print_test_data_loss, \
    print_test_data_accuracy, plot_loss_curve, plot_confusion_matrix, print_classification_report, write_to_file
from Catology.Normalize_Data.constants import training_data_file_path, test_data_file_path, conversion_map, \
    output_file_path, path_for_rn
from Catology.Neuronal_Network.split_data_train_and_test import split_data_train_and_test
from Catology.data_export import save_data_to_excel
from Catology.Neuronal_Network.forward_propagation import forward_propagation
import pandas as pd

# Functia pentru incarcarea si pregatirea datelor din XLSX
def load_and_prepare_data_cats():
    # Importam datele
    data = pd.read_excel(output_file_path)

    ignore_columns = {'Row.names', 'id', 'Horodateur', 'Plus'}
    data = data.drop(columns=list(ignore_columns), errors='ignore')

    # Mapam valorile conform conversion_map
    for column, mapping in conversion_map.items():
        if column in data.columns:
            data[column] = data[column].map(mapping)

    # Eliminam coloanele care nu sunt numerice dupa mapare
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna(axis=1)  # Eliminam coloanele cu valori NaN

    # Separam caracteristicile (X) și etichetele (Y)
    X = data.drop(columns=["Race"], errors='ignore')
    Y = data["Race"]

    # Normalizare
    X = (X - X.min()) / (X.max() - X.min())
    X = X.to_numpy()

    # One-hot encoding pentru etichete
    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(Y.values.reshape(-1, 1))

    # impartire in seturi de antrenament si test
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

    return train_X, test_X, train_Y, test_Y


# Functia pentru initializarea ponderilor si bias-urilor
def initialize_weights_and_bias(input_size, hidden_layers, output_size):
    np.random.seed(42)
    parameters = {}
    layer_sizes = [input_size] + hidden_layers + [output_size]

    for i in range(1, len(layer_sizes)):
        parameters[f"W{i}"] = np.random.randn(layer_sizes[i], layer_sizes[i - 1]) * np.sqrt(2 / layer_sizes[i - 1])
        parameters[f"b{i}"] = np.zeros((layer_sizes[i], 1))

    return parameters


# Functia principala pentru rularea retelei pe setul de date
def neuronal_network():
    # incarcare si pregatire date
    train_X, test_X, train_Y, test_Y = load_and_prepare_data_cats()

    # Parametri de retea
    input_size = train_X.shape[1]
    hidden_layers = [64, 32]  # Straturi ascunse
    output_size = train_Y.shape[1]
    learning_rate = 0.1
    epochs = 1000
    batch_size = 32

    # Initializare
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

            # Propagare inainte
            AL, cache = forward_propagation(X_batch, weights_and_bias)

            # Calcul pierdere
            loss = cross_entropy_loss(AL, Y_batch)

            # Propagare inapoi
            weights_and_bias = backward_propagation(AL, Y_batch, cache, weights_and_bias, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoca {epoch}, Pierdere: {loss}")
            losses.append(loss)
    # Afisam graficul cu evolutia pierderii
    plot_loss_curve(losses)

    # Testare
    X_test = test_X.T
    Y_test = test_Y.T
    AL_test, _ = forward_propagation(X_test, weights_and_bias)

    # Calcul pierdere pe setul de test
    print_test_data_loss(AL_test, Y_test)

    # Calcul acuratețe pe setul de test
    print_test_data_accuracy(AL_test, Y_test)

    # Evaluare pe setul de antrenament
    AL_train, _ = forward_propagation(train_X.T, weights_and_bias)
    print_train_data_accuracy(AL_train, train_Y.T)

    # Afisare matrice de confuzie pentru testare
    classes = [f"Class {i}" for i in range(output_size)]
    plot_confusion_matrix(AL_test, Y_test, classes)

    # Raport de clasificare pentru setul de test
    print_classification_report(AL_test, Y_test, classes)

    np.save("weights_and_bias.npy", weights_and_bias)  # Salvare în fișier