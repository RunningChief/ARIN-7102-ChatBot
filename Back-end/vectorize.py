import json
import torch
from transformers import AutoTokenizer, AutoModel
from chromadb import PersistentClient
from chromadb.config import Settings
from tqdm import tqdm
import numpy as np
import os

# 优化参数
CHROMA_URI = "../Data/database"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256
MAX_TOKENS = 512
OVERLAP_RATIO = 0.2
VECTOR_DIM = 384  # 显式定义向量维度

# 初始化Chroma客户端（启用压缩）
client = PersistentClient(path=CHROMA_URI,
                          settings=Settings(
                              anonymized_telemetry=False,
                              allow_reset=True,
                          ))

# 使用自动混合精度的模型进行加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)
if torch.cuda.is_available():
    model = model.half()  # 使用半精度减少内存占用


def chunk_text(text):
    # 对超过最大长度的文本进行分块
    encoding = tokenizer(text,
                         truncation=False,
                         max_length=MAX_TOKENS,
                         return_offsets_mapping=True,
                         add_special_tokens=False)


    tokens = encoding['input_ids']
    offsets = encoding['offset_mapping']

    chunk_size = MAX_TOKENS - int(MAX_TOKENS * OVERLAP_RATIO)
    chunks = []
    start = 0

    if len(tokens) <= MAX_TOKENS:
        return [text]

    while start < len(tokens):
        end = min(start + MAX_TOKENS, len(tokens))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1] if end > 0 else len(text)
        chunks.append(text[char_start:char_end])
        start += chunk_size

    return chunks


def batch_embed(texts):
    #批量嵌入生成
    batch_embeddings = []

    # 自动分块处理
    batched_chunks = []
    chunk_indices = []
    for i, text in enumerate(texts):
        chunks = chunk_text(text)
        batched_chunks.extend(chunks)
        chunk_indices.extend([i] * len(chunks))

    # 批量处理所有chunks
    for i in tqdm(range(0, len(batched_chunks), BATCH_SIZE), desc="Embedding"):
        batch = batched_chunks[i:i + BATCH_SIZE]
        inputs = tokenizer(batch,
                           return_tensors="pt",
                           max_length=MAX_TOKENS,
                           truncation=True,
                           padding=True).to(device)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            outputs = model(**inputs)
            embeddings = torch.nn.functional.normalize(
                outputs.last_hidden_state[:, 0].float(),  # 使用[CLS] token
                p=2, dim=1
            ).cpu().numpy()

        batch_embeddings.append(embeddings)

    # 合并结果并聚合
    chunk_embeddings = np.concatenate(batch_embeddings, axis=0)
    final_embeddings = np.zeros((len(texts), VECTOR_DIM))
    np.add.at(final_embeddings, chunk_indices, chunk_embeddings)
    counts = np.bincount(chunk_indices, minlength=len(texts))
    final_embeddings /= counts[:, np.newaxis]

    return final_embeddings.astype(np.float32)


if __name__ == "__main__":

    # 加载数据
    with open("../Data/Processed/keyword_index.json") as f:
        keyword_index = json.load(f)
    with open("../Data/Processed/qa_database.json") as f:
        qa_database = json.load(f)

    # 构建文档集合
    documents = []
    metadatas = []

    # 建立QA索引映射
    qa_map = {qa["id"]: qa for qa in qa_database}

    for source, item_ids in keyword_index.items():
        for item_id in item_ids:
            qa = qa_map.get(item_id)
            if not qa:
                continue

            # 合并文本内容（包含所有相关信息）
            combined_text = f"Question: {qa['question']}\nAnswer: {qa['answer']}\nKeywords: {', '.join(qa.get('keywords', []))}"

            metadata = {
                "source": source,
                "item_id": item_id,
                # "keywords": qa.get("keywords", []),  # 保持数组格式
                "keywords": ", ".join(qa.get("keywords", [])),  # 转换为字符串
                "type": "qa"
            }

            documents.append(combined_text)
            metadatas.append(metadata)

    # 生成嵌入向量
    embeddings = batch_embed(documents)

    # 创建集合时指定向量维度
    collection = client.get_or_create_collection(
        name="healthcare_qa",
        metadata={"hnsw:space": "cosine"},  # 使用cosine相似度
        embedding_function=None  # 显式禁用默认embedding
    )

    # 批量插入
    total = len(documents)
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Inserting"):
        batch_ids = [str(j) for j in range(i, min(i + BATCH_SIZE, total))]
        batch_embeddings = embeddings[i:i + BATCH_SIZE].tolist()
        batch_metadatas = metadatas[i:i + BATCH_SIZE]
        batch_documents = documents[i:i + BATCH_SIZE]

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_documents
        )

    print(f"Database size: {collection.count()} items")
    print(f"Estimated size: {os.path.getsize(CHROMA_URI) / 1024 / 1024:.2f} MB")
