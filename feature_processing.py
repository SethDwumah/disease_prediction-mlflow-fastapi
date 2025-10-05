from helper_prabowo_ml import *
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(data, col):
    data[col] = data[col].apply(func=clean_html)
    data[col] = data[col].apply(func=email_address)
    data[col] = data[col].apply(func=remove_)
    data[col] = data[col].apply(func=remove_digits)
    data[col] = data[col].apply(func=remove_links)
    data[col] = data[col].apply(func=remove_special_characters)
    data[col] = data[col].apply(func=removeStopWords)
    data[col] = data[col].apply(func=non_ascii)
    data[col] = data[col].apply(func=punct)
    data[col] = data[col].apply(func=lower)
    return data

def clean_text_for_prediction(text):
    # Apply the cleaning functions step by step on a single text instance
    text = clean_html(text)                # Remove HTML tags
    text = remove_links(text)              # Remove URLs
    text = email_address(text)             # Remove email addresses
    text = remove_digits(text)             # Remove digits
    text = remove_special_characters(text) # Remove special characters
    text = removeStopWords(text)           # Remove stopwords
    text = punct(text)                     # Remove punctuation
    text = non_ascii(text)                 # Remove non-ASCII characters
    text = lower(text)                     # Convert text to lowercase
    
    return text