from flask import Flask, request, jsonify, render_template
import json
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
from langdetect import detect
import random
import torch
import torch.nn as nn
import torch.optim as optim

app = Flask(__name__)

# Initialize stemmer
stemmer = PorterStemmer()

# Load intents file (update your intents.json with multilingual support)
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Lists to hold words, classes, and documents
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Process each pattern
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [stemmer.stem(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Neural network model
class ChatBotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatBotModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x

input_size = len(train_x[0])
hidden_size = 8
output_size = len(classes)
model = ChatBotModel(input_size, hidden_size, output_size)

# Bag of words function
def bag_of_words(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

def detect_language(sentence):
    return detect(sentence)

def predict_intent(sentence):
    language = detect_language(sentence)
    bow = bag_of_words(sentence, words)
    input_data = torch.Tensor(bow).unsqueeze(0)
    output = model(input_data)
    _, predicted = torch.max(output, dim=1)
    tag = classes[predicted.item()]

    # Generate report or return response based on detected language
    for intent in intents['intents']:
        if tag == intent['tag']:
            if language == 'ta':  # Tamil
                return random.choice(intent.get('responses', [''])) + " (Tamil Response)"
            elif language == 'hi':  # Hindi
                return random.choice(intent.get('responses', [''])) + " (Hindi Response)"
            else:
                return random.choice(intent['responses'])  # Default to English

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json.get('message')
    response = predict_intent(message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
