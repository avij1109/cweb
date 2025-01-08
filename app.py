from flask import Flask, render_template, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.urandom(24)
db = SQLAlchemy(app)

# Download NLTK resources if needed
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and load intents file
lemmatizer = WordNetLemmatizer()
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

# Load words, classes, and trained model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# Function to clean up and tokenize the input sentence
def clean_up_sentence(sentence):
    sentence = sentence.lower()
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
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# Function to get response based on predicted intent
def get_response(message):
    intents_list = predict_class(message)
    if len(intents_list) == 0:
        return "Sorry, I didn't understand that."
    tag = intents_list[0]['intent']
    list_of_intents = intents['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        user = User.query.filter_by(username=data['username']).first()
        if user and check_password_hash(user.password, data['password']):
            session['user_id'] = user.id
            return jsonify({'success': True, 'message': 'Login successful'})
        return jsonify({'success': False, 'message': 'Invalid username or password'})
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.json
        existing_user = User.query.filter((User.username == data['username']) | (User.email == data['email'])).first()
        if existing_user:
            return jsonify({'success': False, 'message': 'Username or email already exists'})
        
        hashed_password = generate_password_hash(data['password'])
        new_user = User(username=data['username'], email=data['email'], password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Signup successful'})
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True, 'message': 'Logout successful'})

@app.route('/check_login')
def check_login():
    if 'user_id' in session:
        return jsonify({'logged_in': True})
    return jsonify({'logged_in': False})

@app.route('/get_response', methods=['POST'])
def respond():
    user_message = request.json['message']
    bot_response = get_response(user_message)
    return jsonify({'response': bot_response})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)