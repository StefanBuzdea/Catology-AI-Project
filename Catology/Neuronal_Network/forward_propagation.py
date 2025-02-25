import numpy as np
from Catology.Neuronal_Network.activation_and_loss_functions import relu, softmax


# Calculeaza activarile neuronilor pentru toate straturile prin propagare inainte.
#     - X: intrarile (dimensiunea [num_caracteristici, num_exemple])
#     - weights_and_bias: dictionar cu ponderile si biasele fiecarui strat
def forward_propagation(X, weights_and_bias):
    cache = {}  # Pentru a stoca Z și A pentru fiecare strat (necesar pt backpropagation)
    cache["A0"] = X  # Salvam datele de intrare ca A0
    A = X  # Intrarea initiala este X
    L = len(weights_and_bias) // 2  # Numarul de straturi (W1, b1, ..., WL, bL)

    for l in range(1, L):  # Pentru straturile ascunse
        Z = np.dot(weights_and_bias[f"W{l}"], A) + weights_and_bias[f"b{l}"]  # Calcul Z = W * A + b
        A = relu(Z)  # Aplica ReLU
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A

    # Strat de iesire
    ZL = np.dot(weights_and_bias[f"W{L}"], A) + weights_and_bias[f"b{L}"]  # Calcul Z pentru stratul final
    AL = softmax(ZL)  # Aplica softmax
    cache[f"Z{L}"] = ZL
    cache[f"A{L}"] = AL

    return AL, cache  # Returneaza iesirea finala si cache-ul pentru backward propagation
