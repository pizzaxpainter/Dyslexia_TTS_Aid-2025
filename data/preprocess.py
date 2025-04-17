# data/preprocess.py

import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download NLTK data if not already available
nltk.download("punkt")
nltk.download("stopwords")

# Define a set of stopwords (optional, adjust as needed)
STOPWORDS = set(stopwords.words("english"))

def clean_text(text, remove_stopwords=False):
    """
    Clean the input text by:
    - Lowercasing
    - Removing extra whitespace
    - Removing non-alphanumeric characters (punctuation)
    - Optionally removing stopwords
    """
    # Lowercase
    text = text.lower()
    
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    if remove_stopwords:
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in STOPWORDS]
        text = " ".join(tokens)
    
    return text

def preprocess_dataframe(df, complex_col="Normal", simple_col="Simple", remove_stopwords=False):
    """
    Apply cleaning functions on the dataset columns.
    Expects columns for complex and simple text.
    """
    # Drop rows where either column is missing
    df = df.dropna(subset=[complex_col, simple_col])
    
    # Clean both columns
    df[complex_col] = df[complex_col].apply(lambda x: clean_text(x, remove_stopwords))
    df[simple_col] = df[simple_col].apply(lambda x: clean_text(x, remove_stopwords))
    
    # Remove duplicates
    df = df.drop_duplicates(subset=[complex_col, simple_col])
    
    # Remove entries that are too short (adjust threshold as needed)
    df = df[df[complex_col].str.len() > 10]
    
    return df

def split_sentences(text):
    """
    Tokenizes input text into sentences.
    """
    return nltk.sent_tokenize(text)

def process_csv(input_csv, output_csv, complex_col="Normal", simple_col="Simple", remove_stopwords=False):
    """
    Load a CSV, preprocess it, and then write the cleaned data to output CSV.
    """
    df = pd.read_csv(input_csv)
    df_clean = preprocess_dataframe(df, complex_col, simple_col, remove_stopwords)
    df_clean.to_csv(output_csv, index=False)
    print(f"Processed data saved to: {output_csv}")

if __name__ == "__main__":
    # Example paths (adjust as needed)
    raw_csv_path = os.path.join("data", "raw", "wikilarge_text_simplification.csv")
    output_csv_path = os.path.join("data", "processed", "cleaned_simplification_dataset.csv")
    
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw CSV not found at {raw_csv_path}")
    
    # Process and clean the dataset
    process_csv(raw_csv_path, output_csv_path, remove_stopwords=True)
