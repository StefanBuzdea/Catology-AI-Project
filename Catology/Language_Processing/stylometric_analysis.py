import nltk
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from Catology.Normalize_Data.constants import output_file_path

def stylometric_analysis(text):
    """
    Afiseaza informatii stilometrice pentru textul dat.
    """
    try:
        ########################################
        # codul asta trebuie rulat o singura data
        # Descarcare automata a resursei 'punkt'
        # nltk.download('punkt')
        # nltk.download('punkt_tab')
        # nltk.download('stopwords')
        ########################################
        # Tokenizare si procesare
        words = word_tokenize(text)
        words = [word.lower() for word in words if word.isalnum()]  # Pastreaza doar cuvintele alfanumerice

        # Elimina cuvintele de tip stopwords
        stop_words = set(stopwords.words('english'))  # Foloseste limba engleza
        filtered_words = [word for word in words if word not in stop_words]

        # Lungimea textului
        num_characters = len(text)
        num_words = len(words)
        num_filtered_words = len(filtered_words)

        # Frecventa cuvintelor
        word_frequencies = Counter(filtered_words)

        # Afișează informațiile stilometrice
        print("\nInformatii stilometrice pentru textul selectat:")
        print(f"Text original: {text}")
        print(f"Numar de caractere: {num_characters}")
        print(f"Numar de cuvinte: {num_words}")
        print(f"Numar de cuvinte (fara stopwords): {num_filtered_words}")
        print("Frecventa cuvintelor:")
        for word, freq in word_frequencies.most_common():
            print(f"  {word}: {freq}")

    except Exception as e:
        print(f"A aparut o eroare la analiza stilometrica: {e}")