import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset and convert to DataFrame
dataset = load_dataset("bogdancazan/wikilarge-text-simplification")
df = pd.DataFrame(dataset["train"])  # Convert only the 'train' split

# Handling missing data and whitespace
df = df.fillna("").applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):  # Ensure text is a string
        return ""
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text)  # Remove special characters except hyphen
    words = word_tokenize(text)  # Tokenize
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [subword for word in words for subword in word.split("-")]  # Split hyphenated words
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    
    return " ".join(words)

# Check actual column names before applying preprocessing
print("Dataset columns:", df.columns)

# Apply preprocessing to the correct column (assuming "Normal" is the target column)
df["cleaned_text"] = df["Normal"].apply(preprocess_text)

# Save cleaned dataset
df.to_csv(r"C:\Users\sushm\Downloads\cleaned_dataset.csv", index=False)

print("✅ Text preprocessing completed. Cleaned dataset saved as 'cleaned_dataset.csv'.")

