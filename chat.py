import random
import json
import torch
import spacy
import mysql.connector  # Import MySQL connector

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents JSON file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load trained model data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Load model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Initialize spaCy's small English model for entity recognition
nlp = spacy.load("en_core_web_sm")

bot_name = "Sam"

# MySQL connection setup
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",  # Replace with your MySQL username
        password="",  # Replace with your MySQL password
        database="blog_database"  # Your database name
    )

# Function to fetch restaurant names based on location
def fetch_restaurants_by_location(location):
    connection = connect_db()
    cursor = connection.cursor()

    query = "SELECT name FROM restaurants WHERE location = %s"
    cursor.execute(query, (location,))
    results = cursor.fetchall()
    
    cursor.close()
    connection.close()

    # If there are results, return restaurant names
    if results:
        restaurant_names = [row[0] for row in results]  # Extract restaurant names
        return restaurant_names
    else:
        return []

# Function to extract entities using spaCy
def get_entities(sentence):
    doc = nlp(sentence)
    entities = [(ent.text, ent.label_) for ent in doc.ents]  # Extract entities and their labels
    return entities

# Function to get the response from the bot
def get_response(msg):
    # Tokenize and create bag of words
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Get output from model
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    # Get softmax probability
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # If confidence is high enough
    if prob.item() > 0.60:  # Reduced threshold to 0.60 for flexibility
        # Find matching intent
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # Extract entities using spaCy
                entities = get_entities(msg)
                location = None
                
                # Detect location in entities
                for ent in entities:
                    if ent[1] == "GPE":  # GPE is Geopolitical Entity in spaCy (for places)
                        location = ent[0]

                # If the location is found, query the database
                if location:
                    restaurants = fetch_restaurants_by_location(location)
                    if restaurants:
                        restaurant_list = ', '.join(restaurants)
                        return f"You can visit {restaurant_list} in {location}."
                    else:
                        return f"Sorry, I couldn't find any restaurants in {location}."

                # Return a random response from the matched intent
                return random.choice(intent['responses'])

    return "I do not understand..."

# Main loop for the chatbot in terminal
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        # Get bot response and print it
        resp = get_response(sentence)
        print(f"{bot_name}: {resp}")
