from Catology.Neuronal_Network.neuronal_network_main import neuronal_network


# Funcție pentru optimizarea parametrilor
def optimize_hyperparameters():
    best_accuracy = 0
    best_params = None

    # Posibile valori pentru hiperparametri
    hidden_layers_options = [
        [64, 32],
        [128, 64],
        [64, 64],
        [128, 64, 32],
        [256, 128, 64],
        [128, 128],
        [64, 32, 16],
        [256, 128, 64, 32]
    ]

    learning_rate_options = [0.01, 0.001, 0.0001, 0.1, 0.00001]
    max_epochs_options = [500, 1000, 1500, 2000, 3000, 4000]

    # Iteram prin toate combinatiile posibile
    for hidden_layers in hidden_layers_options:
        for learning_rate in learning_rate_options:
            for max_epochs in max_epochs_options:
                print(
                    f"Antrenam cu: hidden_layers={hidden_layers}, learning_rate={learning_rate}, max_epochs={max_epochs}")

                # Apeleaza functia neuronal_network
                accuracy = neuronal_network(hidden_layers, learning_rate, max_epochs)

                print(f"Acuratete obtinuta: {accuracy:.2f}%")

                # Actualizam cea mai buna combinatie daca acuratetea este mai mare
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = (hidden_layers, learning_rate, max_epochs)

    # Afiseaza cei mai buni parametri si performanta obtinuta
    print("\nCei mai buni parametri gasiți:")
    print(f"hidden_layers: {best_params[0]}")
    print(f"learning_rate: {best_params[1]}")
    print(f"max_epochs: {best_params[2]}")
    print(f"Acuratetea maxima: {best_accuracy:.2f}%")

