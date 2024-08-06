from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import json
# Define the chatbot's responses

# Initialize the TfidF Vectorizer
vectorizer = TfidfVectorizer()
with open("corpus.json", "r") as f:
    data = json.load(f)
# Fit and transform the responses
response_matrix = vectorizer.fit_transform(data['responses'])

def process_input(user_input):
    # Transform the user input
    user_vector = vectorizer.transform([user_input])
    
    # Compute cosine similarities
    similarities = cosine_similarity(user_vector, response_matrix)
    
    # Get the index of the most similar response
    best_match_index = np.argmax(similarities)
    
    # Return the most similar response
    return data['responses'][best_match_index]

def chatbot():
    print('Tongasoa eto amin\'i Rabearivelo!')
    while True:
        user_input = input('Ianao: ')
        response = process_input(user_input)
        print('Rabearivelo:', response)

# Run the chatbot
chatbot()
