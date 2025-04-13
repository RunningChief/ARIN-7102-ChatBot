import numpy as np
from chromadb import Client, Settings
from sklearn.decomposition import PCA
import joblib
import os
from datetime import datetime
import warnings
import cupy as cp
from cuml.cluster import KMeans as cuKMeans
from tqdm import tqdm
# 忽略警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class TopicClusterer:
    def __init__(self, chroma_uri: str = "./Data/database"):
        """初始化聚类器
        
        Args:
            chroma_uri: ChromaDB数据库路径
        """
        self.chroma_uri = chroma_uri
        self.client = Client(Settings(
            persist_directory=chroma_uri,
            anonymized_telemetry=False,
            is_persistent=True
        ))
        # 设置维度信息（修改）
        self.vector_dim = 768  # 与vectorize.py中使用的向量维度保持一致
        
        # 获取集合时先检查是否存在
        try:
            self.collection = self.client.get_collection("healthcare_qa")
        except Exception as e:
            print(f"集合不存在")
        
        self.embeddings = None
        self.reduced_embeddings = None
        self.labels = None
        self.document_ids = None
        
    def load_embeddings(self) -> np.ndarray:
        """从数据库加载embeddings"""
        
        embeddings_cache_file = '/home/dyvm6xra/dyvm6xrauser11/workspace/projects/HKU/Chatbot/Data/Embeddings/embeddings_703df19c43bd6565563071b97e7172ce.npy'
        
        # 如果缓存文件存在，直接加载
        if os.path.exists(embeddings_cache_file) and 0:
            print("发现缓存的embeddings，正在加载...")
            try:
                self.embeddings = np.load(embeddings_cache_file)
                self.document_ids = [str(i) for i in range(len(self.embeddings))]
                print(f"从缓存加载完成，数据形状: {self.embeddings.shape}")
                return self.embeddings
            except Exception as e:
                print(f"加载缓存失败: {e}，将从数据库重新加载")
        else:
            print("正在加载embeddings...")
            print(self.collection.count())
            result = self.collection.get(include=["embeddings"])
            self.embeddings = np.array(result["embeddings"])
            self.document_ids = result["ids"]
 
            print(f"加载完成，数据形状: {self.embeddings.shape}")
            return self.embeddings
    
    def reduce_dimensions(self, n_components: int = 2) -> np.ndarray:
        """使用PCA进行降维
        
        Args:
            n_components: 降维后的维度
        """
        if self.embeddings is None:
            self.load_embeddings()
            
        print("使用PCA进行降维...")
        
        # 使用PCA降维
        reducer = PCA(
            n_components=n_components,
            random_state=42,
            svd_solver='randomized'  # 对大数据集更高效
        )
        self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        cumulative_variance = np.cumsum(reducer.explained_variance_ratio_)
        print(f"PCA累积解释方差比: {cumulative_variance[-1]:.4f}")
        
        print(f"降维完成，降维后形状: {self.reduced_embeddings.shape}")
        
        # 保存降维结果到缓存
        cache_dir = os.path.dirname(os.path.dirname(self.chroma_uri)) + '/Embeddings'
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'pca_reduced_{n_components}d.npy')
        np.save(cache_file, self.reduced_embeddings)
        print(f"降维结果已缓存到: {cache_file}")
        
        return self.reduced_embeddings
    
    def cluster_kmeans(self, n_clusters: int = 4) -> np.ndarray:
        """使用KMeans进行聚类
        
        Args:
            n_clusters: 聚类数
        """
        print("使用GPU加速的KMeans进行聚类...")
        
        # 确保已经进行了降维
        if self.reduced_embeddings is None:
            self.reduce_dimensions()
        
        # 将数据转换为GPU上的数据
        data_gpu = cp.array(self.reduced_embeddings)
        
        # 使用cuML的KMeans
        kmeans = cuKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
            verbose=1
        )
        kmeans.fit(data_gpu)
        self.labels = cp.asnumpy(kmeans.labels_)
        
        # 打印聚类信息
        unique_labels = np.unique(self.labels)
        n_clusters = len(unique_labels)
        
        print(f"发现 {n_clusters} 个聚类")
        for label in unique_labels:
            count = np.sum(self.labels == label)
            percentage = count / len(self.labels) * 100
            print(f"簇 {label}: {count} 样本 ({percentage:.2f}%)")
        
        return self.labels
    
    def update_database(self) -> None:
        """将聚类结果写回数据库"""
        if self.labels is None or self.document_ids is None:
            raise ValueError("请先进行聚类")
            
        print("正在更新数据库...")
        
        # 将标签转换为字符串
        label_strings = [f"cluster_{label}" for label in self.labels]
        
        # 批量更新数据库，每批10000个文档
        batch_size = 500 # 减小批量大小
        total_docs = len(self.document_ids)
        
        for i in tqdm(range(0, total_docs, batch_size), desc="批量更新数据库"):
            batch_end = min(i + batch_size, total_docs)
            batch_ids = self.document_ids[i:batch_end]
            batch_labels = label_strings[i:batch_end]
            # print(batch_ids)
            # print(batch_labels)
            continue

            # self.collection.update( # 改回使用 update
            #     ids=batch_ids,
            #     metadatas=[{"cluster": label} for label in batch_labels]
            # )

        print("数据库更新完成")

def main():
    # 示例用法
    clusterer = TopicClusterer()
    
    # 1. 加载embeddings
    clusterer.load_embeddings()
    
    # 2. PCA降维
    clusterer.reduce_dimensions(n_components=2)
    
    # 3. KMeans聚类
    clusterer.cluster_kmeans(n_clusters=4)
    
    # 4. 更新数据库
    clusterer.update_database()

if __name__ == "__main__":
    main()
