from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import pyarabic  # For Arabic text normalization
from pathlib import Path
import os

# Load model directly

# import jieba  # Optional for word segmentation

app = Flask(__name__)

# Mount Google Drive (authentication required)
output_model_dir = '/content/gk/kk/ColabNotebooks/'
BASE_DIR = Path(__file__).resolve().parent.parent
# Load the pre-trained model and tokenizer
model_path = "C:/Users/asala/OneDrive/Desktop/Model/model"
tokenizer_path =  "./tokenizer"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, force_download=True)
model = AutoModelForSequenceClassification.from_pretrained("Dhalati707/FlaskModel", force_download=True)



nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Define a list of Arabic prepositions to be ignored
prepositions = ["+ال", "و", "في", "من", "إلى", "على", "ب", "ك", "ل", "مع", "حتى", "هو", "هي", "هم", "هن", "أن", "آن", "كي", "+", "ب", "ت+"]


def remove_prefixes_suffixes(text):
  """
  Removes common Arabic prefixes and suffixes from a sentence.

  Args:
      text (str): The Arabic sentence.

  Returns:
      str: The sentence with prefixes and suffixes removed.
  """

  # Normalize Arabic text (remove diacritics, fix kasra ta marbuta, etc.)
  normalized_text = pyarabic.normalize(text)

  # Segment text into words (optional, comment out if not needed)
  # segmented_text = jieba.cut(normalized_text)

  # Remove prefixes and suffixes from each word
  words = []
  for word in normalized_text.split():
    if word in prepositions:
      continue  # Skip prepositions
    elif word.endswith(tuple(prepositions)):
      word = word[:-len(prepositions[0])]  # Remove prepositions as suffixes (optional)
    words.append(word)

  # Join the stemmed words back into a sentence
  stemmed_text = " ".join(words)

  return stemmed_text


def analyze_sentence(sentence):
  """
  Performs sentiment analysis on a sentence after converting Arabic words to base form.

  Args:
      sentence (str): The Arabic sentence.

  Returns:
      list: A list of tuples containing (word, sentiment) for relevant words.
  """

  # Convert sentence to base form
  base_sentence = remove_prefixes_suffixes(sentence)

  # Tokenize the base sentence into words
  words = base_sentence.split()

  # Initialize a list to store the sentiment of each relevant word
  word_sentiments = []

  # Process each word separately
  for word in words:
    # Ignore prepositions
    if word in prepositions:
      continue

    # Perform sentiment analysis on the word
    sentiment = nlp(word)[0]  # Extract the first (and only) element of the list

    # Append the sentiment result to the list
    word_sentiments.append((word, sentiment))

  return word_sentiments


@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
  # Check if file is present in the request
  if 'file' not in request.files:
    return jsonify({'error': 'No file provided'}), 400

  # Read text from file
  file = request.files['file']
  text = file.read().decode("utf-8")

  # Perform sentiment analysis on the text after converting words to base form
  word_sentiments = analyze_sentence(text)

  # Concatenate the labels of relevant words into one line
  labels_line = ' '.join(sentiment['label'] for _, sentiment in word_sentiments)

  # Return the concatenated labels
  return jsonify({'labels': labels_line}), 200


if __name__ == '__main__':
  app.run(debug=False, port=5000)
