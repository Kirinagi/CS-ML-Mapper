import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load slang dictionary
indo_slang_word = pd.read_csv("https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv")

# Convert slang dictionary into a dict: slang -> formal
slang_dict = dict(zip(indo_slang_word['slang'], indo_slang_word['formal']))

# --- Preprocessing Functions (excluding stemming) ---

def casefolding(text):
    """Converts text to lowercase and removes unnecessary characters."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def normalize_slang_words(text):
    """Replaces slang words with their formal equivalents using the slang dictionary."""
    words = text.split()
    normalized_words = [slang_dict[word] if word in slang_dict else word for word in words]
    return ' '.join(normalized_words)

# def remove_stop_words(text):
#     """Removes stop words from the text."""
#     stop_words = set(stopwords.words('indonesian'))
#     words = nltk.word_tokenize(text)
#     return ' '.join([word for word in words if word not in stop_words])

# This is the stemming function (requires Sastrawi, which is commented out)
# def stemming(text):
#     factory = StemmerFactory()
#     stemmer = factory.create_stemmer()
#     return stemmer.stem(text)

def text_preprocessing_process(text):
    """Main text preprocessing pipeline."""
    text = casefolding(text)
    text = normalize_slang_words(text)
    # text = remove_stop_words(text)
    # text = stemming(text)  # Optional if Sastrawi is available
    return text

# --- Load and Process Data ---

try:
    df = pd.read_excel("D:/Work Documents/denormalized_Data.XLSX", engine='openpyxl')
    df['normalized_description'] = df['Description'].apply(text_preprocessing_process)
    df.to_csv('normalized_feedback.csv', index=False)
    print("Successfully processed the file. The output is in 'normalized_feedback.csv'")
    print(df[['Description', 'normalized_description']].head())

except FileNotFoundError:
    print("Error: The file 'denormalized_Data.XLSX' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
