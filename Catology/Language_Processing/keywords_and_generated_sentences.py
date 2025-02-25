import openai
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Dacă nu sunt descărcate resursele NLTK, decomentează liniile de mai jos
# nltk.download('punkt')
# nltk.download('stopwords')

import openai
openai.api_key = "secret-key"

def extract_keywords(text, top_n=5):
    """
    Extrage cele mai relevante cuvinte cheie dintr-un text folosind TF-IDF.
    """
    stop_words = list(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([text])

    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Returnează cuvintele în forma originală din text, fără duplicare
    keywords = []
    for word, _ in sorted_scores[:top_n]:
        for original_word in word_tokenize(text):
            if original_word.lower() == word and original_word not in keywords:
                keywords.append(original_word)
                break

    return keywords

def call_openai_to_generate_sentence(keyword, context):
    """
    Apelează API-ul OpenAI pentru a genera o propoziție nouă utilizând cuvântul cheie și contextul său.
    """
    prompt = (
        f"Generate a completely new sentence using the word '{keyword}' in the context of: \"{context}\". "
        f"Make sure the sentence preserves the original meaning of the keyword in this context."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for generating sentences."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error generating sentence for '{keyword}': {e}"

def generate_sentences_from_keywords(text, keywords):
    """
    Generează propoziții noi pentru fiecare cuvânt cheie identificat, utilizând propoziția în care apare cuvântul cheie drept context.
    """
    sentences = sent_tokenize(text)
    keyword_sentences = {}

    for keyword in keywords:
        for sentence in sentences:
            if keyword in sentence:
                context = sentence
                generated_sentence = call_openai_to_generate_sentence(keyword, context)
                keyword_sentences[keyword] = generated_sentence
                break

    return keyword_sentences

def process_keywords_and_generate_sentences(text, top_n=5):
    """
    Extrage cuvinte cheie din text și generează propoziții relevante pentru fiecare.
    """
    keywords = extract_keywords(text, top_n=top_n)
    print("Cuvinte cheie identificate:")
    print(keywords)

    print("\nPropoziții generate pentru cuvintele cheie:")
    keyword_sentences = generate_sentences_from_keywords(text, keywords)
    for keyword, sentence in keyword_sentences.items():
        print(f"{keyword}: {sentence}")

    return keyword_sentences
