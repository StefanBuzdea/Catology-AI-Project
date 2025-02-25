import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
from collections import Counter

def download_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def replace_with_synonyms_or_hypernyms(word):
    synonyms = []
    hypernyms = []

    # Obtine sinonime
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                synonyms.append(lemma.name().replace('_', ' '))

    # Obtine hipernime
    for syn in wordnet.synsets(word):
        for hypernym in syn.hypernyms():
            for lemma in hypernym.lemmas():
                hypernyms.append(lemma.name().replace('_', ' '))

    # Intoarce aleatoriu un sinonim, hipernim sau pastreaza cuvantul
    alternatives = synonyms + hypernyms
    if alternatives:
        return random.choice(alternatives)
    return word

def replace_with_negated_antonym(word):
    antonyms = []

    # Obtine antonime
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            for antonym in lemma.antonyms():
                antonyms.append("not " + antonym.name().replace('_', ' '))

    # Intoarce aleatoriu un antonim negat sau pastreaza cuvantul
    if antonyms:
        return random.choice(antonyms)
    return word

def generate_alternative_versions(text, replacement_rate=0.6):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))

    new_versions = []

    for _ in range(5):  # Genereaza 5 versiuni alternative
        new_text = []
        for word in words:
            if word.lower() in stop_words or not word.isalpha():
                new_text.append(word)
            else:
                if random.random() < replacement_rate:
                    if random.choice([True, False]):
                        new_text.append(replace_with_synonyms_or_hypernyms(word))
                    else:
                        new_text.append(replace_with_negated_antonym(word))
                else:
                    new_text.append(word)
        new_versions.append(' '.join(new_text))

    return new_versions
