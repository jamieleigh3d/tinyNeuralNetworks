import nltk
from nltk.tokenize import word_tokenize
from string import punctuation

nltk.download('punkt')  # Download the necessary resource if not already downloaded

def tokenize_without_punctuation(text):
    tokens = word_tokenize(text.lower())
    tokens_without_punctuation = [token for token in tokens if token not in punctuation]
    return tokens_without_punctuation

text = "Hello, this is an example sentence! It includes some punctuation."
tokens = tokenize_without_punctuation(text)

print(tokens)