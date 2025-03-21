import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os

# Try to download ALL required NLTK resources with error handling
required_packages = ['punkt', 'stopwords', 'wordnet']
for package in required_packages:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        print(f"Package {package} is already downloaded.")
    except LookupError:
        print(f"Downloading NLTK {package}...")
        nltk.download(package)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Modified function to avoid punkt_tab dependency
def simple_tokenize(text):
    """A simpler tokenization function that doesn't rely on punkt_tab"""
    # First lowercase and strip
    text = text.lower().strip()
    # Split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

# Load your datasets
print("Loading datasets...")
data_dir = '../Data'
# Check if the path exists before trying to access it
if os.path.exists(os.path.join(data_dir, 'healthline_articles_text.csv')):
    healthline_df = pd.read_csv(os.path.join(data_dir, 'healthline_articles_text.csv'))
    print(f"Loaded healthline data with {len(healthline_df)} entries")
else:
    print(f"Warning: Could not find healthline_articles_text.csv in {data_dir}")
    healthline_df = pd.DataFrame()

# Load all MedQA datasets
medqa_dir = os.path.join(data_dir, 'MedQA')
if os.path.exists(medqa_dir):
    medqa_files = [f for f in os.listdir(medqa_dir) if f.endswith('.csv')]
    medqa_dataframes = {}

    for file_name in medqa_files:
        file_path = os.path.join(medqa_dir, file_name)
        dataset_name = file_name.split('.')[0]
        medqa_dataframes[dataset_name] = pd.read_csv(file_path)
        print(f"Loaded {dataset_name} with {len(medqa_dataframes[dataset_name])} entries")
else:
    print(f"Warning: Could not find MedQA directory at {medqa_dir}")
    medqa_dataframes = {}

def clean_text(text):
    """Clean text by removing special characters, extra spaces, etc."""
    if isinstance(text, str):
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        # Remove special characters but keep letters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text
    return ""

def extract_keywords(text, top_n=10):
    """Extract the most important keywords from text using TF-IDF."""
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Use our simple tokenizer instead of word_tokenize
    tokens = simple_tokenize(text.lower())
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    medical_stopwords = {'disease', 'patient', 'treatment', 'condition', 'symptom', 'doctor', 'health', 'may', 'also', 'one', 'use'}
    stop_words.update(medical_stopwords)
    
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words and len(token) > 2]
    
    # Process with TF-IDF if there are tokens
    if filtered_tokens:
        # Use TF-IDF to get important words
        vectorizer = TfidfVectorizer(max_features=top_n)
        try:
            tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_tokens)])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores for each word
            scores = zip(feature_names, tfidf_matrix.toarray()[0])
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
            # Return top keywords
            return [word for word, score in sorted_scores[:top_n] if score > 0]
        except Exception as e:
            print(f"TF-IDF extraction failed: {e}")
            # Fallback if TF-IDF fails (e.g., not enough tokens)
            return filtered_tokens[:top_n]
    return []

print("Cleaning and extracting keywords from datasets...")

# Dictionary to store all QA pairs with their keywords
qa_database = []
keyword_index = {}  # Inverted index: keyword → [qa_id1, qa_id2, ...]

# Process Healthline articles (convert to QA format)
if not healthline_df.empty:
    print("Processing Healthline data...")
    for idx, row in healthline_df.iterrows():
        title = row.get('title', '')
        content = row.get('content', '')
        
        if not isinstance(title, str) or not isinstance(content, str):
            continue
            
        clean_title = clean_text(title)
        clean_content = clean_text(content)
        
        # Consider the title as the question and the content as the answer
        qa_pair = {
            'id': f"healthline_{idx}",
            'source': 'healthline',
            'question': clean_title,
            'answer': clean_content,
            'keywords': extract_keywords(clean_title + " " + clean_content)
        }
        
        qa_database.append(qa_pair)
        
        # Update inverted index
        for keyword in qa_pair['keywords']:
            if keyword not in keyword_index:
                keyword_index[keyword] = []
            keyword_index[keyword].append(qa_pair['id'])

# Process all MedQA datasets
if medqa_dataframes:
    print("Processing MedQA datasets...")
    for dataset_name, df in medqa_dataframes.items():
        # Map column names to standardized format
        question_col = next((col for col in ['Question', 'question'] if col in df.columns), None)
        answer_col = next((col for col in ['Answer', 'answer'] if col in df.columns), None)
        
        if not question_col or not answer_col:
            print(f"Skipping {dataset_name} - missing question/answer columns")
            continue
        
        for idx, row in df.iterrows():
            question = row.get(question_col, '')
            answer = row.get(answer_col, '')
            
            if not isinstance(question, str) or not isinstance(answer, str):
                continue
                
            clean_question = clean_text(question)
            clean_answer = clean_text(answer)
            
            qa_pair = {
                'id': f"{dataset_name}_{idx}",
                'source': dataset_name,
                'question': clean_question,
                'answer': clean_answer,
                'keywords': extract_keywords(clean_question + " " + clean_answer)
            }
            
            qa_database.append(qa_pair)
            
            # Update inverted index
            for keyword in qa_pair['keywords']:
                if keyword not in keyword_index:
                    keyword_index[keyword] = []
                keyword_index[keyword].append(qa_pair['id'])

print(f"Total QA pairs processed: {len(qa_database)}")
print(f"Total keywords in index: {len(keyword_index)}")

# Save the cleaned QA database and keyword index
processed_dir = os.path.join(data_dir, 'Processed')
os.makedirs(processed_dir, exist_ok=True)

with open(os.path.join(processed_dir, 'qa_database.json'), 'w') as f:
    json.dump(qa_database, f, indent=2)

with open(os.path.join(processed_dir, 'keyword_index.json'), 'w') as f:
    json.dump(keyword_index, f, indent=2)

print("Processing complete. Files saved to Data/Processed/")

