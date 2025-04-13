## 1. Data Preprocessing: Convert Raw Medical Text Data to Standardized Q&A Pairs
```bash
python Back-end/data_processing/process_qa.py
```

### Data Preprocessing Instructions

#### Feature Description
- Clean and convert raw medical text data into standardized Question-Answer (QA) pairs format
- Support processing data from multiple sources (Healthline articles, MedQA dataset, etc.)
- Use TF-IDF algorithm for keyword extraction
- Generate inverted index for QA pairs to facilitate retrieval
- Automatically remove duplicate and invalid QA pairs

#### Data Requirements
The data directory structure should be as follows:
```
./Data/
├── Raw/
│   ├── Healthline/
│   │   └── healthline_articles_text.csv
│   └── MedQA/
│       └── *.csv
└── Processed/
    └── cleaned_qa/
```

#### Input File Format
1. Healthline Article Data (CSV format):
   - Required columns: 'title' (as question), 'content' (as answer)

2. MedQA Data (CSV format):
   - Required columns: 'Question'/'question', 'Answer'/'answer'

#### Output Files
Processed data will be saved in `./Data/Processed/cleaned_qa/` and `keywords/` directories:
1. `qa_database.json`: Contains all processed QA pairs
   ```json
   [
     {
       "id": "unique_id",
       "source": "data_source",
       "question": "cleaned_question",
       "answer": "cleaned_answer",
       "keywords": ["keyword1", "keyword2", ...]
     },
     ...
   ]
   ```
2. `keyword_index.json`: Keyword inverted index
   ```json
   {
     "keyword1": ["qa_id1", "qa_id2", ...],
     "keyword2": ["qa_id3", "qa_id4", ...],
     ...
   }
   ```

#### Data Processing Steps
1. Text Cleaning:
   - Remove HTML tags
   - Normalize punctuation
   - Remove excess whitespace
   - Convert to lowercase

2. Keyword Extraction:
   - Use NLTK for tokenization
   - Remove stopwords (including general and medical domain-specific stopwords)
   - Use TF-IDF algorithm to extract important keywords

3. Quality Control:
   - Filter invalid QA pairs
   - Remove duplicate content
   - Generate unique identifiers

#### Dependency Installation
```bash
pip install nltk scikit-learn pandas tqdm
```

## 2. Text Vectorization: Convert QA Pairs to High-dimensional Vectors and Build Vector Database
```bash
python Back-end/models/vectorize.py
```

### Vectorization Process Instructions

#### Feature Description
- Use medical domain pre-trained model BioBERT-MNLI to convert QA pairs into high-dimensional vectors
- Build efficient vector database supporting semantic similarity search
- Implement vector caching mechanism to avoid repeated computation
- Multi-threaded parallel processing to improve data processing efficiency
- Support incremental updates and data persistence

#### System Architecture
1. Model Configuration:
   - Pre-trained model: `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`
   - Vector dimension: 768
   - Device support: Automatic GPU/CPU detection
   - Memory optimization: Batch processing mechanism

2. Vectorization Process:
   - Data loading and preprocessing
   - Text vector generation (with caching support)
   - Vector normalization (L2 regularization)
   - Batch processing (BATCH_SIZE=1024)
   - Multi-threaded parallel insertion (MAX_WORKERS=8)

3. Database Optimization:
   - Use ChromaDB for vector storage
   - HNSW index configuration:
     - Space metric: cosine
     - Build parameters: ef=100
     - Search parameters: ef=128
     - Graph connectivity: M=32/64
   - Batch persistence (40000 entries/batch)

#### Data Directory Structure
```
./Data/
├── database/          # ChromaDB persistent storage
├── Embeddings/        # Vector cache directory
└── Processed/
    ├── keywords/      # Keyword index
    └── cleaned_qa/    # Preprocessed QA data
```

#### Vectorization Process
1. Data Preparation:
   - Load QA data and keyword index
   - Merge questions, answers, and keywords into unified text
   - Build metadata (source, ID, keywords, etc.)

2. Vector Generation:
   - Calculate data hash for cache validation
   - Check and load cached vectors
   - Generate new text vectors in batches
   - Automatically save vectors to cache directory

3. Database Construction:
   - Create temporary in-memory database
   - Multi-threaded parallel data insertion
   - Batch persistence to disk
   - Automatic progress tracking and memory monitoring

#### Performance Metrics
- Batch size: 1024 entries/batch
- Insertion batch: 1024 entries/batch
- Persistence batch: 40000 entries/batch
- Parallel threads: 8
- Vector dimension: 768
- Memory usage monitoring: Real-time tracking

#### Dependency Requirements
```bash
pip install torch sentence-transformers chromadb tqdm numpy psutil
```

#### Hardware Recommendations
- Recommended configuration: NVIDIA GPU (8GB+ VRAM)
- Minimum configuration: 8GB system memory (CPU mode)

#### Usage Instructions
1. Run vectorization processing:
```bash
python Back-end/models/vectorize.py
```

2. Test database:
```bash
python Back-end/models/test_db.py
```

The test program will:
- Verify database integrity
- Display random sample data
- Execute example queries
- Show similarity scores

Example query result:
```
Results for query term 'diabetes':

Result 1:
----------------------------------------
Similarity score: 0.6597

Document content:
Question: what are the treatments for diabetes
Answer: diabetes is a very serious disease over time diabetes that is not well managed causes serious damage to the eyes kidneys nerves and heart gums and teeth if you have diabetes you are more likely than someone who does not have diabetes to have heart disease or a stroke people with diabetes also tend to develop heart disease or stroke at an earlier age than others the best way to protect yourself from the serious complications of diabetes is to manage your blood glucose blood pressure and cholesterol and avoid smoking it is not always easy but people who make an ongoing effort to manage their diabetes can greatly improve their overall health
Keywords: diabetes, heart, serious, blood, manage, people, stroke, best, complication, damage

Metadata:
{
    'item_id': 'MedicalQuestionAnswering_5480',
    'keywords': 'diabetes, heart, serious, blood, manage, people, stroke, best, complication, damage',
    'source': 'diabetes',
    'type': 'qa'
}
```

This example demonstrates:
- Semantic similarity search effectiveness
- Results including complete QA pairs
- Related metadata information
- Keyword extraction results

## 3. Topic Clustering: Perform Topic Clustering on Vectorized QA Pairs
```bash
python Back-end/models/cluster_topic.py
```

### Topic Clustering Instructions

#### Feature Description
- Use UMAP for high-dimensional vector dimensionality reduction
- Apply HDBSCAN algorithm for density clustering
- Support GPU acceleration (if available)
- Automatically update clustering results to vector database
- Clustering results persistence and caching mechanism

#### System Architecture
1. Dimensionality Reduction Configuration:
   - Algorithm: UMAP (Uniform Manifold Approximation and Projection)
   - Output dimension: 50
   - Number of neighbors: 50
   - Minimum distance: 0.2
   - Distance metric: cosine
   - GPU support: automatic detection

2. Clustering Configuration:
   - Algorithm: HDBSCAN
   - Minimum cluster size: 100
   - Minimum samples: 10
   - Distance metric: euclidean
   - Cluster selection method: EOM (Excess of Mass)
   - Parallel processing: multi-core support

3. Data Flow:
   - Load vectors from ChromaDB
   - UMAP dimensionality reduction
   - HDBSCAN clustering
   - Write results back to database

#### Data Directory Structure
```
./Data/
├── database/          # ChromaDB storage
├── Embeddings/        # Dimensionality reduction result cache
└── Processed/
    └── clusters/      # Clustering results
```

#### Performance Optimization
- Automatic GPU acceleration support
- Dimensionality reduction result caching
- Parallel computation optimization
- Memory usage optimization

#### Dependency Requirements
```bash
pip install umap-learn hdbscan joblib
# GPU acceleration (optional)
pip install cupy cuml
```

#### Usage Instructions
1. Run clustering processing:
```bash
python Back-end/models/cluster_topic.py
```

2. Clustering Results:
- Each document will be assigned a cluster label
- Label format: `cluster_N` (N is cluster number)
- Noise points marked as `noise`
- Results stored in vector database metadata

Example query result:
```json
{
    "id": "doc_id",
    "metadata": {
        "cluster": "cluster_1",
        "source": "original_source",
        "keywords": ["keyword1", "keyword2", ...]
    }
}
```

## 4. Topic Classification: Train Topic Classifier and Make Predictions
```bash
python Back-end/models/topic_classification.py
```

### Topic Classification Instructions

#### Feature Description
- Train random forest classifier based on clustering results
- Use 5-fold cross-validation to evaluate model performance
- Support model persistence and version management
- Provide topic prediction functionality for new documents
- Automatic integration with vector database system

#### System Architecture
1. Classifier Configuration:
   - Algorithm: Random Forest
   - Number of trees: 100
   - Parallel processing: multi-core support
   - Evaluation metrics: accuracy, macro-F1, weighted-F1
   - Cross-validation: 5-fold stratified validation

2. Training Process:
   - Load vectors and labels from ChromaDB
   - Data preprocessing and cleaning
   - Model training and validation
   - Performance evaluation and reporting
   - Save optimal model

3. Prediction Process:
   - Load trained model
   - New document vectorization
   - Topic prediction
   - Update results to database

#### Data Directory Structure
```
./Data/
├── database/          # ChromaDB storage
└── models/            # Model storage directory
    └── topic_classifier_*.joblib  # Timestamped model files
```

#### Performance Metrics
- Cross-validation folds: 5
- Evaluation metrics:
  - Accuracy
  - Macro-average F1
  - Weighted-average F1
- Model version control: timestamp naming

#### Dependency Requirements
```bash
pip install scikit-learn joblib numpy
```

#### Usage Instructions
1. Train classifier:
```bash
python Back-end/models/topic_classification.py
```

2. Model Output:
- Detailed performance report for each fold during training
- Final model saved in `./models/` directory
- Filename format: `topic_classifier_YYYYMMDD_HHMMSS.joblib`

Example output:
```
Loading data...
Data loading complete, feature shape: (N, 768)
Number of classes: K

Starting 5-fold cross-validation...
Fold 1 validation:
Classification report:
              precision    recall  f1-score 
   cluster_0       0.85      0.83      0.84       
   cluster_1       0.82      0.80      0.81      
...

Overall performance:
Average accuracy: 0.8234 ± 0.0256
Average macro F1 score: 0.8156 ± 0.0278
Average weighted F1 score: 0.8245 ± 0.0267

Model saved to: ./models/topic_classifier_20240315_143022.joblib
```

