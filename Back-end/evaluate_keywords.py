import json
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt

# Load the processed data
with open('../Data/Processed/qa_database.json', 'r') as f:
    qa_database = json.load(f)

with open('../Data/Processed/keyword_index.json', 'r') as f:
    keyword_index = json.load(f)

# Analysis 1: Keyword distribution
keyword_counts = Counter()
for qa in qa_database:
    keyword_counts.update(qa['keywords'])

print(f"Total unique keywords: {len(keyword_counts)}")
print("\nTop 20 most common keywords:")
for keyword, count in keyword_counts.most_common(20):
    print(f"{keyword}: {count}")

# Analysis 2: Keywords per QA pair
keyword_per_qa = [len(qa['keywords']) for qa in qa_database]
avg_keywords = sum(keyword_per_qa) / len(keyword_per_qa) if keyword_per_qa else 0
print(f"\nAverage keywords per QA pair: {avg_keywords:.2f}")

# Analysis 3: Show random examples
print("\nRandom QA pairs with their keywords:")
for _ in range(3):
    random_qa = random.choice(qa_database)
    print(f"\nSource: {random_qa['source']}")
    print(f"Question: {random_qa['question'][:100]}...")
    print(f"Keywords: {', '.join(random_qa['keywords'])}")

# Plot keyword distribution
plt.figure(figsize=(12, 6))
plt.hist(keyword_per_qa, bins=20, alpha=0.7)
plt.title('Distribution of Keywords per QA Pair')
plt.xlabel('Number of Keywords')
plt.ylabel('Frequency')
plt.savefig('Data/Processed/keyword_distribution.png')
print("\nKeyword distribution plot saved to Data/Processed/keyword_distribution.png")

# Sample queries to test the keyword search
test_queries = [
    "cancer symptoms",
    "heart disease treatment",
    "diabetes management"
]

print("\nTesting sample queries:")
for query in test_queries:
    print(f"\nQuery: {query}")
    query_words = query.lower().split()
    relevant_qa_ids = set()
    
    for word in query_words:
        if word in keyword_index:
            relevant_qa_ids.update(keyword_index[word])
    
    print(f"Found {len(relevant_qa_ids)} potentially relevant QA pairs")
    if relevant_qa_ids:
        sample_id = random.choice(list(relevant_qa_ids))
        sample_qa = next((qa for qa in qa_database if qa['id'] == sample_id), None)
        if sample_qa:
            print(f"Sample match - Question: {sample_qa['question'][:100]}...") 