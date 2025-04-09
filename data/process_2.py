import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob

# Download necessary NLTK data
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
df = pd.read_csv(r"C:\Users\User\Desktop\rewritten_texts_csv.csv", encoding="utf-8", encoding_errors="replace")

# Handling missing data and whitespaces
df = df.fillna("")  # Replace NaN values with empty strings
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # Strip whitespace

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters (punctuation, symbols)
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text)  # Only allow alphanumeric characters and hyphen
    
    # Tokenize text
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Optionally handle compound words more carefully (split hyphenated words if needed)
    words = [subword for word in words for subword in word.split("-")]

    # Apply Lemmatization (to keep words in full form like 'rewrite' instead of 'rewrit')
    words_processed = [lemmatizer.lemmatize(word) for word in words]  # Using lemmatization here

    # Reconstruct the text
    return " ".join(words_processed)

# Apply text preprocessing to all text columns
for column in ['original_text', 'rewritten_text', 'prompt', 'final_prompt']:
    df[f"{column}_cleaned"] = df[column].apply(preprocess_text)

# Deduplication (optional, if you want to keep only unique rows based on cleaned text)
df = df.drop_duplicates()

# Save the cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("Text preprocessing completed. Cleaned dataset saved as 'cleaned_dataset.csv'.")