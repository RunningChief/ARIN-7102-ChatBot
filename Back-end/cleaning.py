import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load your datasets
healthline_df = pd.read_csv('Data/healthline_articles_text.csv')

# Get all CSV files from MedQA directory
import os
import glob

# Load all MedQA datasets
medqa_files = glob.glob('Data/MedQA/*.csv')
medqa_dataframes = {}

for file_path in medqa_files:
    file_name = os.path.basename(file_path).split('.')[0]
    medqa_dataframes[file_name] = pd.read_csv(file_path)

def clean_text(text):
    """Clean text by removing special characters, extra spaces, etc."""
    if isinstance(text, str):
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Apply cleaning to the healthline dataset
healthline_df['clean_content'] = healthline_df['content'].apply(clean_text)

# Apply cleaning to all MedQA datasets
for dataset_name, df in medqa_dataframes.items():
    # Check for common column names and clean them
    if 'Question' in df.columns:
        df['clean_question'] = df['Question'].apply(clean_text)
    if 'Answer' in df.columns:
        df['clean_answer'] = df['Answer'].apply(clean_text)
    # Save cleaned dataset
    df.to_csv(f'Data/MedQA/{dataset_name}_cleaned.csv', index=False)

# Save cleaned healthline dataset
healthline_df.to_csv('Data/healthline_articles_cleaned.csv', index=False)

