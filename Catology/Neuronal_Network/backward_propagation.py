import numpy as np
from Catology.Neuronal_Network.activation_and_loss_functions import cross_entropy_loss_derivative, relu_derivative


# Propagare inapoi pentru a calcula gradientele si a actualiza ponderile.
#     - AL: iesirea finala (probabilitatile softmax)
#     - Y: etichetele reale (one-hot encoded)
#     - cache: valori intermediare (Z si A pentru fiecare strat)
#     - weights_and_bias: ponderile si biasele retelei
#     - learning_rate: rata de invatare pentru actualizarea ponderilor
def backward_propagation(AL, Y, cache, weights_and_bias, learning_rate):
    gradients = {}
    L = len(weights_and_bias) // 2  # Numarul de straturi
    m = Y.shape[1]  # Numarul de exemple

    # Gradient pentru stratul de iesire
    dZL = cross_entropy_loss_derivative(AL, Y)  # Derivata pierderii (softmax + cross-entropy)
    gradients[f"dW{L}"] = np.dot(dZL, cache[f"A{L-1}"].T) / m
    gradients[f"db{L}"] = np.sum(dZL, axis=1, keepdims=True) / m

    # Propagare inapoi pentru straturile ascunse
    for l in range(L - 1, 0, -1):
        dZ = np.dot(weights_and_bias[f"W{l+1}"].T, dZL) * relu_derivative(cache[f"Z{l}"])
        gradients[f"dW{l}"] = np.dot(dZ, cache[f"A{l-1}"].T) / m
        gradients[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m
        dZL = dZ  # Pregatire pentru stratul anterior

    # optimizare - regularizare L2
    # lambda_reg = 0.01  # Coeficientul de regularizare
    # for l in range(1, L + 1):
    #     gradients[f"dW{l}"] += lambda_reg * weights_and_bias[f"W{l}"] / m

    # Cliparea gradientilor - optimizare
    for key in gradients:
        np.clip(gradients[key], -1, 1, out=gradients[key])

    # Actualizare ponderi si biase
    for l in range(1, L + 1):
        weights_and_bias[f"W{l}"] -= learning_rate * gradients[f"dW{l}"]
        weights_and_bias[f"b{l}"] -= learning_rate * gradients[f"db{l}"]

    return weights_and_bias