import json
import torch
from sentence_transformers import SentenceTransformer
from chromadb import Client, Settings,EmbeddingFunction
from tqdm import tqdm
import numpy as np
import os
import psutil
import time
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

CHROMA_URI = "./Data/database"
EMBEDDING_MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
VECTOR_DIM = 768
EMBEDDINGS_DIR = "./Data/Embeddings"



class BioEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.model.to(self.device)
        # 添加dimensionality属性，获取模型输出的嵌入维度
        self.dimensionality = self.model.get_sentence_embedding_dimension()
        # 确保dimensionality在类实例中正确设置
        if not hasattr(self, "dimensionality") or self.dimensionality is None:
            self.dimensionality = VECTOR_DIM
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            input,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

if __name__ == "__main__":
    embedding_func = BioEmbeddingFunction()

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

    # 建立QA索引映射
    print("\n[2/5] 处理文档数据...")
    documents = []
    metadatas = []

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

    client = Client(
        Settings(
            persist_directory=CHROMA_URI,
            anonymized_telemetry=False,
            is_persistent=True
        )
    )

    collection = client.get_or_create_collection(
        name="healthcare_qa",
        embedding_function=embedding_func,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 128,
            "hnsw:M": 64,
        }
    )

    # 分批持久化数据
    PERSIST_BATCH_SIZE = 40000  # 设置小于最大限制 41666
    total_records = len(documents)
    
    print("\n[4/5] 开始持久化数据到向量数据库...")
    
    with tqdm(total=total_records, desc="持久化进度") as pbar:
        for i in range(0, total_records, PERSIST_BATCH_SIZE):
            end_idx = min(i + PERSIST_BATCH_SIZE, total_records)
            
            # 获取当前批次的数据
            batch_ids = [str(j) for j in range(i, end_idx)]
            batch_documents = documents[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
 
            # 添加到持久化集合
            collection.upsert(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            
            pbar.update(end_idx - i)
    
    print("\n[5/5] 完成数据处理和持久化!")
    print(f"总共处理了 {total_records} 条记录")
    print(f"向量维度: {embedding_func.dimensionality}")
    
