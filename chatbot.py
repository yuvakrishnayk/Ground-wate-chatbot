import json
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
from langdetect import detect
import random
import torch
import torch.nn as nn
import torch.optim as optim

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
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents
        documents.append((w, intent['tag']))
        # Add to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Stemming and lowercasing
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    # Initialize bag of words
    bag = []
    pattern_words = [stemmer.stem(word.lower()) for word in doc[0]]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # Output is a 1 for the current intent, otherwise 0
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# Shuffle and convert into numpy arrays
random.shuffle(training)
training = np.array(training, dtype=object)

# Split the features and labels
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Define neural network model
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

# Hyperparameters
input_size = len(train_x[0])
hidden_size = 8
output_size = len(classes)
learning_rate = 0.001
num_epochs = 1000

# Model, loss function, optimizer
model = ChatBotModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Convert training data to PyTorch tensors
train_x_tensor = torch.Tensor(train_x)
train_y_tensor = torch.Tensor(train_y)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(train_x_tensor)
    loss = criterion(output, train_y_tensor.argmax(dim=1))
    loss.backward()
    optimizer.step()

# Bag of words function
def bag_of_words(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

# Function to generate a report (customized response)
def generate_report():
    report = """
    **Ground Water Resource Assessment:**
    - Detailed assessment of groundwater resources in the area.

    **Categorization of the Area:**
    - Classification based on groundwater availability and usage.

    **GW Management Practices to be Adopted:**
    - Recommended practices for sustainable groundwater management.

    **Conditions for Obtaining NOC for Groundwater Extraction:**
    - Guidelines and conditions for obtaining No Objection Certificate (NOC).

    **Guidance on How to Obtain NOC:**
    - Step-by-step process for applying and obtaining NOC.

    **Definition of Groundwater Terms:**
    - Explanation of common terms related to groundwater.

    **Training Opportunities Related to Groundwater:**
    - Information on available training programs and workshops.
    """
    return report

# Function to detect the language
def detect_language(sentence):
    return detect(sentence)  # 'en', 'ta', 'hi', etc.

# Function to predict the intent
def predict_intent(sentence):
    language = detect_language(sentence)
    bow = bag_of_words(sentence, words)
    input_data = torch.Tensor(bow).unsqueeze(0)
    output = model(input_data)
    _, predicted = torch.max(output, dim=1)
    tag = classes[predicted.item()]
    
    # Check if the intent is to generate a report
    if tag == "generate_report":
        return generate_report()
    
    # Return a random response based on detected language
    for intent in intents['intents']:
        if tag == intent['tag']:
            if language == 'ta':  # Tamil
                return random.choice(intent.get('responses', [''])) + " (Tamil Response)"
            elif language == 'hi':  # Hindi
                return random.choice(intent.get('responses', [''])) + " (Hindi Response)"
            else:
                return random.choice(intent['responses'])  # Default to English

# Start the chatbot
print("Chatbot is ready to talk! (type 'quit' to stop)")
while True:
    query = input("You: ")
    if query.lower() == "quit":
        break
    response = predict_intent(query)
    print(f"Bot: {response}")
