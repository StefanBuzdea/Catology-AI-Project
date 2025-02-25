import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from Catology.Neuronal_Network.activation_and_loss_functions import cross_entropy_loss
import os


def write_to_file(content, file_path=None, mode="a"):
    """
    Scrie conținutul în fișier.
    :param content: Textul care trebuie scris.
    :param file_path: Calea fișierului în care se scrie (implicit None).
    :param mode: Modul de scriere ('w' pentru înlocuire, 'a' pentru adăugare).
    """
    # Path complet către fișierul din directorul `Neuronal_Network`
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), "nn_statistics.txt")

    with open(file_path, mode) as file:
        file.write(content + "\n")


def print_train_data_accuracy(AL, Y):
    predictions = np.argmax(AL, axis=0)
    true_labels = np.argmax(Y, axis=0)  # Dacă Y este one-hot encoded
    accuracy = np.mean(predictions == true_labels)

    content = (
        "\n\n------------------------------------------------------------\n"
        "Evaluare acuratete pe setul de antrenament:\n"
        f"Acuratetea pe setul de antrenament: {accuracy * 100:.2f}%\n"
        f"Iesirile stratului final (probabilitati):\n{AL}\n"
        "------------------------------------------------------------\n\n"
    )
    write_to_file(content)


def print_test_data_loss(AL_test, Y_test):
    loss_test = cross_entropy_loss(AL_test, Y_test)

    content = (
        "\n\n------------------------------------------------------------\n"
        "Calcul pierdere pe setul de test:\n"
        f"Pierdere pe setul de test: {loss_test}\n"
        "------------------------------------------------------------\n\n"
    )
    write_to_file(content)


def print_test_data_accuracy(AL_test, Y_test):
    predictions_test = np.argmax(AL_test, axis=0)
    true_labels_test = np.argmax(Y_test, axis=0)
    accuracy_test = np.mean(predictions_test == true_labels_test)

    content = (
        "\n\n------------------------------------------------------------\n"
        "Calcul acuratete pe setul de test:\n"
        f"Acuratetea pe setul de test: {accuracy_test * 100:.2f}%\n"
        "------------------------------------------------------------\n\n"
    )
    write_to_file(content)


def plot_loss_curve(losses, title="Evolutia pierderii"):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Pierdere")
    plt.title(title)
    plt.xlabel("Epoca")
    plt.ylabel("Pierdere")
    plt.legend()
    plt.grid()
    plt.show()


def plot_confusion_matrix(AL, Y, classes):
    true_labels = np.argmax(Y, axis=0)
    predictions = np.argmax(AL, axis=0)
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='viridis', xticks_rotation=45)
    plt.title("Matricea de confuzie")
    plt.show()


def print_classification_report(AL, Y, classes):
    true_labels = np.argmax(Y, axis=0)
    predictions = np.argmax(AL, axis=0)
    classes = [str(cls) for cls in classes]
    report = classification_report(true_labels, predictions, target_names=classes, zero_division=0)

    content = (
        "\nRaport de clasificare:\n"
        f"{report}\n"
    )
    write_to_file(content)

