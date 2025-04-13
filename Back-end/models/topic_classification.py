import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from chromadb import Client, Settings
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TopicClassifier:
    def __init__(self, chroma_uri: str = "./Data/database"):
        """初始化分类器
        
        Args:
            chroma_uri: ChromaDB数据库路径
        """
        self.chroma_uri = chroma_uri
        self.client = Client(Settings(
            persist_directory=chroma_uri,
            anonymized_telemetry=False,
            is_persistent=True
        ))
        self.collection = self.client.get_collection("healthcare_qa")
        self.model = None
        self.X = None
        self.y = None
        
    def load_data(self):
        """从数据库加载数据和标签"""
        print("正在加载数据...")
        
        # 加载嵌入向量
        result = self.collection.get(include=["embeddings", "metadatas"])
        self.X = np.array(result["embeddings"])
        
        # 从元数据中提取cluster标签
        self.y = []
        for metadata in result["metadatas"]:
            cluster = metadata.get("cluster", "noise")
            # 将cluster_X格式转换为数字标签
            if cluster == "noise":
                self.y.append(-1)
            else:
                self.y.append(int(cluster.split("_")[1]))
        self.y = np.array(self.y)
        
        # 移除噪声点
        mask = self.y != -1
        self.X = self.X[mask]
        self.y = self.y[mask]
        
        print(f"数据加载完成，特征形状: {self.X.shape}")
        print(f"类别数量: {len(np.unique(self.y))}")
        
    def train_and_evaluate(self, n_splits=5):
        """使用5折交叉验证训练和评估模型"""
        if self.X is None or self.y is None:
            self.load_data()
            
        print(f"\n开始{n_splits}折交叉验证...")
        
        # 初始化分层K折交叉验证
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # 存储每折的性能指标
        fold_scores = {
            'accuracy': [],
            'macro_f1': [],
            'weighted_f1': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y), 1):
            print(f"\n第 {fold} 折验证:")
            
            # 划分训练集和验证集
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            # 训练模型
            print("训练模型...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                n_jobs=-1,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # 预测和评估
            y_pred = self.model.predict(X_val)
            
            # 计算性能指标
            accuracy = accuracy_score(y_val, y_pred)
            macro_f1 = f1_score(y_val, y_pred, average='macro')
            weighted_f1 = f1_score(y_val, y_pred, average='weighted')
            
            fold_scores['accuracy'].append(accuracy)
            fold_scores['macro_f1'].append(macro_f1)
            fold_scores['weighted_f1'].append(weighted_f1)
            
            print("\n分类报告:")
            print(classification_report(y_val, y_pred))
            
        # 输出平均性能
        print("\n总体性能:")
        print(f"平均准确率: {np.mean(fold_scores['accuracy']):.4f} ± {np.std(fold_scores['accuracy']):.4f}")
        print(f"平均宏F1分数: {np.mean(fold_scores['macro_f1']):.4f} ± {np.std(fold_scores['macro_f1']):.4f}")
        print(f"平均加权F1分数: {np.mean(fold_scores['weighted_f1']):.4f} ± {np.std(fold_scores['weighted_f1']):.4f}")
        
    def save_model(self, model_dir: str = "./models"):
        """保存最终模型"""
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"topic_classifier_{timestamp}.joblib")
        
        joblib.dump(self.model, model_path)
        print(f"\n模型已保存到: {model_path}")

def main():
    # 示例用法
    classifier = TopicClassifier()
    classifier.train_and_evaluate()
    classifier.save_model()

if __name__ == "__main__":
    main()
