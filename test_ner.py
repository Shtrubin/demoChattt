import spacy

# Load the trained model
nlp = spacy.load("custom_ner_model")

# Test the model with new text
texts = [
    "I love eating momo and visiting Bhaktapur.",
    "Where is the best place to eat pizza in Kathmandu?",
    "Pokhara is beautiful during the summer.",
    "I had a great time in Bhaktapur."
]

for text in texts:
    print(f"Text: {text}")
    doc = nlp(text)
    if doc.ents:
        for ent in doc.ents:
            print(f"Entity: {ent.text}, Label: {ent.label_}")
    else:
        print("No entities found.")
    print()