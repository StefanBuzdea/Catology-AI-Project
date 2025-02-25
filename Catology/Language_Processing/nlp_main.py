import pandas as pd
from deep_translator import GoogleTranslator
from langdetect import detect

from Catology.Language_Processing.keywords_and_generated_sentences import process_keywords_and_generate_sentences
from Catology.Language_Processing.stylometric_analysis import stylometric_analysis
from Catology.Language_Processing.text_paraphrasing import generate_alternative_versions, download_nltk_resources
from Catology.Normalize_Data.constants import output_file_path
from Catology.Normalize_Data.data_conversions import translate_plus_column

def nlp_main():
    """
    Citeste primul text din coloana 'Plus' din fisierul Excel si identifica limba in care este scris textul.
    """
    try:
        # Citeste datele din fisierul Excel
        data = pd.read_excel(output_file_path)

        # Verifica daca coloana 'Plus' exista
        if 'Plus' not in data.columns:
            print("Coloana 'Plus' nu exista în fisierul Excel.")
            return

        # Obtine primul text din coloana 'Plus'
        first_text = data['Plus'].dropna().iloc[0]  # Ignora valorile lipsa (NaN)
        #traducem textul in engleza
        translator = GoogleTranslator(source='auto', target='en')
        translated_text = translator.translate(first_text) if isinstance(first_text, str) else first_text
        # first_text = translated_text

        # Detecteaza limba textului
        detected_language = detect(first_text)

        print(f"Primul text din coloana 'Plus': {first_text}")
        print(f"Text din coloana 'Plus' tradus in engleza: {translated_text}")
        print(f"Limba detectata: {detected_language}")
        print("-------------------------------------------------------------------------------")

        # Apeleaza analiza stilometrica pe textul in engleza
        stylometric_analysis(translated_text)
        print("-------------------------------------------------------------------------------")

        # Generam variante diferite de propozitii similare
        # download_nltk_resources()
        alternative_texts = generate_alternative_versions(translated_text)
        print("\nVersiuni alternative:")
        for i, alt_text in enumerate(alternative_texts, 1):
            print(f"Versiunea {i}: {alt_text}")

        print("-------------------------------------------------------------------------------")

        # Extragere cuvinte cheie si generare propozitii relevante
        process_keywords_and_generate_sentences(translated_text)
        print("-------------------------------------------------------------------------------")

    except Exception as e:
        print(f"A aparut o eroare: {e}")


