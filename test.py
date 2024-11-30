import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Test sentence
sentence = "Where to eat in Bhaktapur?"

# Process the sentence
doc = nlp(sentence)

# Extract entities
entities = [(ent.text, ent.label_) for ent in doc.ents]
print(f"Entities detected: {entities}")