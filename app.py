from flask import Flask, request, jsonify
import pyarabic.araby
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os

app = Flask(__name__)

# Load the pre-trained model and tokenizer once at startup
model_path = "Dhalati707/FlaskModel"
tokenizer_path = "./tokenizer"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Define a list of Arabic prepositions to be ignored
prepositions = ["+ال", "و", "في", "من", "إلى", "على", "ب", "ك", "ل", "مع", "حتى", "هو", "هي", "هم", "هن", "أن", "آن", "كي", "+", "ب", "ت+"]

def remove_prefixes_suffixes(text):
    normalized_text = pyarabic.araby.strip_tashkeel(text)
    words = []
    for word in normalized_text.split():
        if word in prepositions:
            continue
        elif word.endswith(tuple(prepositions)):
            word = word[:-len(prepositions[0])]
        words.append(word)
    return " ".join(words)

def analyze_sentence(sentence):
    base_sentence = remove_prefixes_suffixes(sentence)
    sentiments = nlp(base_sentence)  # Process entire sentence at once
    word_sentiments = [(word, sentiment) for word, sentiment in zip(base_sentence.split(), sentiments) if word not in prepositions]
    return word_sentiments

@app.route('/', methods=['GET'])
def sentiment_analysis():
    query_params = request.args
    print("argument",query_params)
    request_text = query_params.get("text")
    print("stop 1")
    if not request_text:
        return jsonify({'error': 'No text provided'}), 400
    word_sentiments = analyze_sentence(request_text)
    print("stop 2, after word_sentiments")

    labels_line = ' '.join(sentiment['label'] for _, sentiment in word_sentiments)
    print("stop 3")
    return jsonify({'labels': labels_line}), 200

if __name__ == '__main__':
    app.run(debug=False)
