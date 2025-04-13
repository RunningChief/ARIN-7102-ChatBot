import json
import torch
from sentence_transformers import SentenceTransformer
from chromadb import Client, Settings, EmbeddingFunction
from tqdm import tqdm
import numpy as np
import os
import psutil
import time
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

# 优化参数
CHROMA_URI = "./Data/database"
EMBEDDING_MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
BATCH_SIZE = 1024
VECTOR_DIM = 768
INSERT_BATCH_SIZE = 1024
EMBEDDINGS_DIR = "./Data/Embeddings"  

class BioEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.model.to(self.device)
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            input,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

# 初始化Chroma客户端
client = Client(
    Settings(
        persist_directory=CHROMA_URI,
        anonymized_telemetry=False,
        is_persistent=True
    )
)

# 初始化模型
embedding_function = BioEmbeddingFunction()
model = embedding_function.model

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 转换为MB

def format_time(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def batch_embed(texts):
    """
    使用sentence-transformers进行批量文本嵌入
    """
    # 使用tqdm显示进度
    embeddings = []
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="生成文本向量"):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2标准化
        )
        embeddings.append(batch_embeddings)
        
    return np.concatenate(embeddings, axis=0)

def parallel_upsert(collection, start_idx: int, end_idx: int, 
                   documents: List[str], embeddings: np.ndarray, 
                   metadatas: List[Dict[str, Any]]) -> None:
    """
    使用add而不是upsert，因为我们使用的是临时内存模式
    """
    batch_ids = [str(j) for j in range(start_idx, end_idx)]
    batch_embeddings = embeddings[start_idx:end_idx].tolist()
    batch_metadatas = metadatas[start_idx:end_idx]
    batch_documents = documents[start_idx:end_idx]
    
    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas,
        documents=batch_documents
    )

def calculate_data_hash(documents: List[str]) -> str:
    """
    计算文档列表的哈希值，用于验证数据是否改变
    """
    combined_text = "".join(documents)
    return hashlib.md5(combined_text.encode()).hexdigest()

def save_embeddings(embeddings: np.ndarray, data_hash: str):
    """
    保存embeddings到文件
    """
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    embedding_path = os.path.join(EMBEDDINGS_DIR, f"embeddings_{data_hash}.npy")
    np.save(embedding_path, embeddings)
    print(f"Embeddings已保存到: {embedding_path}")

def load_embeddings(data_hash: str) -> np.ndarray:
    """
    从文件加载embeddings
    """
    embedding_path = os.path.join(EMBEDDINGS_DIR, f"embeddings_{data_hash}.npy")
    if os.path.exists(embedding_path):
        return np.load(embedding_path)
    return None

def vectorize_data(documents, embeddings, metadatas):
    collection = client.get_or_create_collection(
        name="healthcare_qa",
        embedding_function=embedding_function
    )
    PERSIST_BATCH_SIZE = 5000 
    total_records = len(documents)
    
    with tqdm(total=total_records, desc="持久化进度") as pbar:
        for i in range(0, total_records, PERSIST_BATCH_SIZE):
            end_idx = min(i + PERSIST_BATCH_SIZE, total_records)
            
            batch_ids = [str(j) for j in range(i, end_idx)]
            batch_embeddings = embeddings[i:end_idx]
            batch_documents = documents[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            
            pbar.update(end_idx - i)
    
    return collection

if __name__ == "__main__":
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始向量化处理...")
    print(f"使用设备: {model.device}")
    print(f"初始内存使用: {get_memory_usage():.2f} MB")
    
    # 创建输出目录
    os.makedirs(CHROMA_URI, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # 加载数据
    print("\n[1/5] 加载数据文件...")
    loading_start = time.time()
    with open("./Data/Processed/keywords/keyword_index.json") as f:
        keyword_index = json.load(f)
    with open("./Data/Processed/cleaned_qa/qa_database.json") as f:
        qa_database = json.load(f)
    print(f"数据加载完成，用时: {format_time(time.time() - loading_start)}")
    print(f"当前内存使用: {get_memory_usage():.2f} MB")

    # 构建文档集合
    print("\n[2/5] 处理文档数据...")
    documents = []
    metadatas = []

    # 建立QA索引映射
    print("建立QA索引映射...")
    qa_map = {qa["id"]: qa for qa in qa_database}
    
    # 使用tqdm显示文档处理进度
    total_items = sum(len(item_ids) for item_ids in keyword_index.values())
    with tqdm(total=total_items, desc="处理文档") as pbar:
        for source, item_ids in keyword_index.items():
            for item_id in item_ids:
                qa = qa_map.get(item_id)
                if not qa:
                    pbar.update(1)
                    continue

                combined_text = f"Question: {qa['question']}\nAnswer: {qa['answer']}\nKeywords: {', '.join(qa.get('keywords', []))}"

                metadata = {
                    "source": source,
                    "item_id": item_id,
                    "keywords": ", ".join(qa.get("keywords", [])),
                    "type": "qa"
                }

                documents.append(combined_text)
                metadatas.append(metadata)
                pbar.update(1)

    print(f"文档处理完成，共处理 {len(documents)} 条记录")
    print(f"当前内存使用: {get_memory_usage():.2f} MB")

    if 0:
        documents = documents[:1000]
        metadatas = metadatas[:1000]

    # 生成嵌入向量
    print("\n[3/5] 生成文本向量...")
    vector_start = time.time()
    
    # 计算数据哈希值
    data_hash = calculate_data_hash(documents)
    
    # 尝试加载已存在的embeddings
    embeddings = load_embeddings(data_hash)
    
    if embeddings is not None:
        print("找到缓存的embeddings，直接加载使用")
    else:
        print("未找到缓存的embeddings，重新计算...")
        embeddings = batch_embed(documents)
        # 保存embeddings
        save_embeddings(embeddings, data_hash)
        
    print(f"向量生成完成，用时: {format_time(time.time() - vector_start)}")
    print(f"当前内存使用: {get_memory_usage():.2f} MB")

    # 在生成向量后，使用新的vectorize_data函数
    print("\n[4/5] 创建数据库集合...")
    collection = vectorize_data(documents, embeddings, metadatas)

    total_time = time.time() - start_time
    print(f"\n数据库构建完成！")
    print(f"总用时: {format_time(total_time)}")
    print(f"总条目数: {collection.count()} 条")
    print(f"数据库大小: {os.path.getsize(CHROMA_URI) / 1024 / 1024:.2f} MB")
    print(f"最终内存使用: {get_memory_usage():.2f} MB")
