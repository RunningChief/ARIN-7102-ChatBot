import json
import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any
from tqdm import tqdm
import hashlib

# 下载必要的NLTK资源
required_packages = ['punkt', 'stopwords', 'wordnet']
for package in required_packages:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        print(f"Package {package} is already downloaded.")
    except LookupError:
        print(f"Downloading NLTK {package}...")
        nltk.download(package)

class QAProcessor:
    def __init__(self, data_dir: str, output_path: str):
        self.data_dir = data_dir
        self.output_path = output_path
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.medical_stopwords = {'disease', 'patient', 'treatment', 'condition', 'symptom', 'doctor', 'health', 'may', 'also', 'one', 'use'}
        self.stop_words.update(self.medical_stopwords)
        
    def clean_text(self, text: str) -> str:
        """清理文本"""
        if isinstance(text, str):
            # 移除HTML标签
            text = re.sub(r'<.*?>', ' ', text)
            # 移除特殊字符但保留字母和数字
            text = re.sub(r'[^\w\s]', ' ', text)
            # 移除多余的空格
            text = re.sub(r'\s+', ' ', text).strip().lower()
            return text
        return ""

    def simple_tokenize(self, text: str) -> List[str]:
        """简单的分词函数"""
        text = text.lower().strip()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """使用TF-IDF提取关键词"""
        if not isinstance(text, str) or not text.strip():
            return []
        
        tokens = self.simple_tokenize(text)
        filtered_tokens = [self.lemmatizer.lemmatize(token) 
                         for token in tokens 
                         if token.isalpha() and token not in self.stop_words and len(token) > 2]
        
        if filtered_tokens:
            vectorizer = TfidfVectorizer(max_features=top_n)
            try:
                tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_tokens)])
                feature_names = vectorizer.get_feature_names_out()
                
                scores = zip(feature_names, tfidf_matrix.toarray()[0])
                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
                
                return [word for word, score in sorted_scores[:top_n] if score > 0]
            except Exception as e:
                print(f"TF-IDF提取失败: {e}")
                return filtered_tokens[:top_n]
        return []

    def process_data(self) -> tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        """处理所有数据源"""
        qa_database = []
        keyword_index = {}  # 倒排索引：keyword → [qa_id1, qa_id2, ...]

        # 处理Healthline文章
        healthline_path = os.path.join(self.data_dir,'Healthline', 'healthline_articles_text.csv')
        if os.path.exists(healthline_path):
            print("处理Healthline数据...")
            healthline_df = pd.read_csv(healthline_path)
            for idx, row in tqdm(healthline_df.iterrows(), total=len(healthline_df)):
                title = row.get('title', '')
                content = row.get('content', '')
                
                if not isinstance(title, str) or not isinstance(content, str):
                    continue
                    
                clean_title = self.clean_text(title)
                clean_content = self.clean_text(content)
                
                qa_pair = {
                    'id': f"healthline_{idx}",
                    'source': 'healthline',
                    'question': clean_title,
                    'answer': clean_content,
                    'keywords': self.extract_keywords(clean_title + " " + clean_content)
                }
                
                qa_database.append(qa_pair)
                
                # 更新倒排索引
                for keyword in qa_pair['keywords']:
                    if keyword not in keyword_index:
                        keyword_index[keyword] = []
                    keyword_index[keyword].append(qa_pair['id'])

        # 处理MedQA数据
        medqa_dir = os.path.join(self.data_dir, 'MedQA')
        if os.path.exists(medqa_dir):
            print("处理MedQA数据...")
            for file_name in os.listdir(medqa_dir):
                if file_name.endswith('.csv'):
                    dataset_name = file_name.split('.')[0]
                    df = pd.read_csv(os.path.join(medqa_dir, file_name))
                    
                    question_col = next((col for col in ['Question', 'question'] if col in df.columns), None)
                    answer_col = next((col for col in ['Answer', 'answer'] if col in df.columns), None)
                    
                    if not question_col or not answer_col:
                        print(f"跳过 {dataset_name} - 缺少问题/答案列")
                        continue
                    
                    for idx, row in tqdm(df.iterrows(), total=len(df)):
                        question = row.get(question_col, '')
                        answer = row.get(answer_col, '')
                        
                        if not isinstance(question, str) or not isinstance(answer, str):
                            continue
                            
                        clean_question = self.clean_text(question)
                        clean_answer = self.clean_text(answer)
                        
                        qa_pair = {
                            'id': f"{dataset_name}_{idx}",
                            'source': dataset_name,
                            'question': clean_question,
                            'answer': clean_answer,
                            'keywords': self.extract_keywords(clean_question + " " + clean_answer)
                        }
                        
                        qa_database.append(qa_pair)
                        
                        for keyword in qa_pair['keywords']:
                            if keyword not in keyword_index:
                                keyword_index[keyword] = []
                            keyword_index[keyword].append(qa_pair['id'])

        return qa_database, keyword_index

    def save_results(self, qa_database: List[Dict[str, Any]], keyword_index: Dict[str, List[str]]):
        """保存处理后的数据"""
        
        qa_output = os.path.join(self.output_path, 'cleaned_qa','qa_database.json')
        keyword_output = os.path.join(self.output_path, 'keywords','keyword_index.json')
        os.makedirs(os.path.dirname(qa_output), exist_ok=True)
        os.makedirs(os.path.dirname(keyword_output), exist_ok=True)

        
        with open(qa_output, 'w', encoding='utf-8') as f:
            json.dump(qa_database, f, ensure_ascii=False, indent=2)
            
        with open(keyword_output, 'w', encoding='utf-8') as f:
            json.dump(keyword_index, f, ensure_ascii=False, indent=2)
            
        print(f"处理完成的QA对数量: {len(qa_database)}")
        print(f"关键词索引中的关键词数量: {len(keyword_index)}")
        print(f"数据已保存到: {self.output_path}")

if __name__ == "__main__":
    # 设置路径
    data_dir = './Data/raw'
    output_path = './Data/Processed/'
    
    # 创建处理器实例并处理数据
    processor = QAProcessor(data_dir, output_path)
    qa_database, keyword_index = processor.process_data()
    processor.save_results(qa_database, keyword_index) 