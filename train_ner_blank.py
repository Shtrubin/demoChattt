import spacy
from spacy.training import Example
import random

# Define your extended training data
TRAIN_DATA = [
    # Examples with LOCATION
    ("Where can I eat in Bhaktapur?", {"entities": [(18, 28, "LOCATION")]}),
    ("I want to visit Kathmandu next summer.", {"entities": [(20, 30, "LOCATION")]}),
    ("Pokhara has beautiful lakes and views.", {"entities": [(0, 7, "LOCATION")]}),
    ("The best restaurants are in Kathmandu.", {"entities": [(25, 35, "LOCATION")]}),
    ("I went to Pokhara for a vacation.", {"entities": [(9, 16, "LOCATION")]}),
    ("I am planning a trip to New York.", {"entities": [(22, 30, "LOCATION")]}),
    ("The Eiffel Tower is in Paris.", {"entities": [(18, 23, "LOCATION")]}),
    ("Tokyo is a bustling city.", {"entities": [(0, 5, "LOCATION")]}),
    ("I visited San Francisco last year.", {"entities": [(10, 23, "LOCATION")]}),
    ("London is known for its landmarks.", {"entities": [(0, 6, "LOCATION")]}),
    
    # Examples with FOOD
    ("I love eating momo and pizza.", {"entities": [(12, 16, "FOOD"), (21, 26, "FOOD")]}),
    ("The restaurant serves pasta and sushi.", {"entities": [(19, 24, "FOOD"), (29, 34, "FOOD")]}),
    ("My favorite foods are burgers and fries.", {"entities": [(20, 26, "FOOD"), (31, 35, "FOOD")]}),
    ("Do you like chocolate and ice cream?", {"entities": [(12, 20, "FOOD"), (25, 35, "FOOD")]}),
    ("Pizza is my go-to meal.", {"entities": [(0, 5, "FOOD")]}),
    ("I had a delicious meal of spaghetti and meatballs.", {"entities": [(20, 29, "FOOD"), (34, 46, "FOOD")]}),
    ("Sushi is a popular dish in Japan.", {"entities": [(0, 4, "FOOD")]}),
    ("Indian curry is rich and flavorful.", {"entities": [(0, 11, "FOOD")]}),
    ("I enjoy a good steak with vegetables.", {"entities": [(10, 15, "FOOD")]}),
    ("Tacos and burritos are great for lunch.", {"entities": [(0, 5, "FOOD"), (10, 18, "FOOD")]}),
    
    # Mixed Examples
    ("The food in Bangkok is amazing.", {"entities": [(12, 19, "LOCATION")]}),
    ("I had sushi and ramen in Tokyo.", {"entities": [(7, 12, "FOOD"), (17, 22, "FOOD"), (27, 32, "LOCATION")]}),
    ("I love visiting new places and trying different foods.", {"entities": []}),
    ("San Francisco has some great Chinese restaurants.", {"entities": [(0, 13, "LOCATION"), (28, 35, "FOOD")]}),
    ("The best tacos are found in Mexico City.", {"entities": [(14, 19, "FOOD"), (31, 43, "LOCATION")]}),
    ("I visited Rome and had some great gelato.", {"entities": [(9, 13, "LOCATION"), (24, 30, "FOOD")]}),
    ("Enjoy your meal in Los Angeles.", {"entities": [(16, 29, "LOCATION")]}),
    ("Eating ramen in Seoul was fantastic.", {"entities": [(7, 12, "FOOD"), (17, 22, "LOCATION")]}),
    ("The local cuisine in Barcelona is diverse.", {"entities": [(20, 30, "LOCATION")]}),
    ("Have you tried the local cheese in Amsterdam?", {"entities": [(36, 45, "LOCATION")]}),
]
# Create a blank English model
nlp = spacy.blank("en")

# Create the NER component and add it to the pipeline
ner = nlp.add_pipe("ner")

# Add new entity labels to the NER component
ner.add_label("FOOD")
ner.add_label("LOCATION")

# Prepare the training data
train_examples = []
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    train_examples.append(example)

# Create the optimizer
optimizer = nlp.begin_training()

# Training the model
for epoch in range(20):  # Increased number of epochs
    random.shuffle(train_examples)
    losses = {}
    for example in train_examples:
        nlp.update([example], drop=0.5, losses=losses)
    print(f"Epoch {epoch} Losses: {losses}")

# Save the trained model
nlp.to_disk("custom_ner_model")

print("Model training completed and saved.")