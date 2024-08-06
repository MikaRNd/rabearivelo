from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define the chatbot's responses
responses = [
    'Hi! How can I help you today?',
    'Hello! What brings you here?',
    'I\'m doing great, thanks! How about you?',
    'My name is Chatty, nice to meet you!',
    'See you later! Have a great day!'
]

# Define the user inputs that correspond to each response
responses_keys = [
    'hello',
    'hi',
    'how are you',
    'what is your name',
    'goodbye'
]

# Initialize the TfidF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the responses
response_matrix = vectorizer.fit_transform(responses)

def process_input(user_input):
    # Transform the user input
    user_vector = vectorizer.transform([user_input])
    
    # Compute cosine similarities
    similarities = cosine_similarity(user_vector, response_matrix)
    
    # Get the index of the most similar response
    best_match_index = np.argmax(similarities)
    
    # Return the most similar response
    return responses[best_match_index]

def chatbot():
    print('Welcome to Chatty!')
    while True:
        user_input = input('You: ')
        response = process_input(user_input)
        print('Chatty:', response)

# Run the chatbot
chatbot()
