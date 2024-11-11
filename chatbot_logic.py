import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Download NLTK resources if needed
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and load intents file
lemmatizer = WordNetLemmatizer()
try:
    with open(r'C:\chatbot\intents.json') as file:
        intents = json.load(file)
except FileNotFoundError:
    print("Error: The 'intents.json' file was not found. Please check the file path.")

# Load words, classes, and trained model
try:
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.h5')
except (FileNotFoundError, IOError) as e:
    print(f"Error loading model or data files: {e}")

# Function to clean up and tokenize the input sentence
def clean_up_sentence(sentence):
    sentence = sentence.lower()  # Convert to lowercase for consistency
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Function to create a bag of words from the input sentence
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

# Function to predict the intent class from the input sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.1  # Lowered threshold to allow more predictions
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# Function to get response based on predicted intent
def get_response(message):
    intents_list = predict_class(message)  # Get the predicted class
    if len(intents_list) == 0:
        return "Sorry, I didn't understand that."  # Fallback for no prediction
    tag = intents_list[0]['intent']
    list_of_intents = intents['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Optional: Start chatbot in command line mode
if __name__ == "__main__":
    print("GO! Bot is running!")
    while True:
        try:
            message = input("You: ")
            res = get_response(message)
            print(f"Bot: {res}")
        except Exception as e:
            print(f"Error: {str(e)}")
