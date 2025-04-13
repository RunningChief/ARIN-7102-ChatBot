import json
from chromadb import Client, Settings, EmbeddingFunction
from pprint import pprint
import random
import os
from sentence_transformers import SentenceTransformer
import torch

# 配置参数
CHROMA_URI = "./Data/database"
EMBEDDING_MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
VECTOR_DIM = 768  # BioBERT的向量维度

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

def test_database():
    print("="*50)
    print("开始测试数据库")
    print("="*50)
    
    # 初始化客户端
    client = Client(Settings(
        persist_directory=CHROMA_URI,
        anonymized_telemetry=False,
        is_persistent=True
    ))
    

    embedding_func = BioEmbeddingFunction()
    collection = client.get_or_create_collection(
        name="healthcare_qa",
        embedding_function=embedding_func
    )
    
    # 1. 显示基本信息
    print("\n1. 数据库基本信息:")
    print(f"数据库位置: {os.path.abspath(CHROMA_URI)}")
    print(f"数据库大小: {os.path.getsize(CHROMA_URI) / 1024 / 1024:.2f} MB")
    print(f"总条目数: {collection.count()} 条")
    print(f"使用的嵌入模型: {EMBEDDING_MODEL_NAME}")
    
    # 2. 随机获取样本
    print("\n2. 随机样本展示:")
    total_items = collection.count()
    sample_size = min(2, total_items)
    random_indices = random.sample(range(total_items), sample_size)
    
    results = collection.get(
        ids=[str(i) for i in random_indices],
        include=["documents", "metadatas"]
    )
    
    for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas']), 1):
        print(f"\n样本 {i}:")
        print("-" * 40)
        print("文档内容:")
        print(doc)
        print("\n元数据:")
        pprint(metadata)
        print("-" * 40)


    # # 4. 测试 update
    # print("\n4. 测试 update 功能:")
    # # 随机选择一个文档
    # total_items = collection.count()
    # random_index = random.randint(0, total_items - 1)
    # random_id = str(random_index)
    # random_index2 = random.randint(0, total_items - 1)
    # random_id2 = str(random_index2)

    # # 更新文档内容
    # new_content = "更新后的文档内容"
    # collection.update(
    #     ids=[random_id, random_id2],
    #     metadatas=[{"cluster": "new_cluster"}, {"cluster": "new_cluster2"}]
    # )
    
    # # 验证更新
    # results = collection.get(
    #     ids=[random_id],
    #     include=["documents", "metadatas"]
    # )
    # print(f"\n更新后的文档内容: {results['documents'][0]}")
    # print(f"\n更新后的元数据: {results['metadatas'][0]}")   


    # 3. 测试简单查询
    print("\n3. 测试查询功能:")
    query = "diabetes"
    results = collection.query(
        query_texts=[query],
        n_results=1,
        include=["documents", "metadatas", "distances"]
    )
    
    print(f"\n使用查询词 '{query}' 的结果:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"\n结果 {i}:")
        print("-" * 40)
        print(f"相似度得分: {1 - distance:.4f}")
        print("\n文档内容:")
        print(doc)
        print("\n元数据:")
        pprint(metadata)
        print("-" * 40)
    


    

if __name__ == "__main__":
    test_database()
