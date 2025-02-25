import numpy as np


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return Z > 0


# Functia softmax pentru stratul de iesire
def softmax(Z):
    exp_values = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Stabilizare numerica
    probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
    return probabilities


# Funcția de eroare: entropia incrucisata
#     - Y_hat: predictiile modelului (dimensiunea: [num_clase, num_exemple])
#     - Y: etichetele reale (dimensiunea: [num_clase, num_exemple], one-hot encoded)
def cross_entropy_loss(Y_hat, Y, weights_and_bias = None):
    m = Y.shape[1]  # Numarul de exemple
    loss = -np.sum(Y * np.log(Y_hat + 1e-9)) / m  # adaugam 1e-9 pentru stabilitate numerica

    # lambda_reg = 0.01
    # if weights_and_bias is not None:  # Adaugam regularizare L2 daca e cazul
    #     l2_regularization = (lambda_reg / (2 * m)) * sum(
    #         [np.sum(np.square(weights_and_bias[f"W{l}"])) for l in range(1, len(weights_and_bias) // 2 + 1)]
    #     )
    #     loss += l2_regularization

    return loss


# Derivata functiei de eroare fata de predictii (cross-entropy + softmax derivative)
def cross_entropy_loss_derivative(Y_hat, Y):
    return Y_hat - Y