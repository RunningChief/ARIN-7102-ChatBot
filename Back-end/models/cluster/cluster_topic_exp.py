import numpy as np
from chromadb import Client, Settings
from sklearn.decomposition import PCA
import hdbscan
from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import seaborn as sns
from tqdm import tqdm
import joblib
import os
import json
import argparse
from datetime import datetime
import warnings

# 添加 RAPIDS cuML 库导入
import cudf
import cuml
from cuml.cluster import HDBSCAN as cuHDBSCAN
from cuml.cluster import KMeans as cuKMeans
from cuml.manifold import UMAP as cuUMAP
import cupy as cp

# 忽略特定的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class ClusterAnalyzer:
    def __init__(self, chroma_uri: str = "./Data/database", output_dir: str = None, experiment_name: str = None, use_gpu: bool = True):
        self.chroma_uri = chroma_uri
        self.client = Client(Settings(
            persist_directory=chroma_uri,
            anonymized_telemetry=False,
            is_persistent=True
        ))
        self.collection = self.client.get_collection("healthcare_qa")
        self.embeddings = None
        self.reduced_embeddings = None
        self.labels = None
        self.use_gpu = use_gpu
        
        # 创建结果保存目录
        self.results_dir = output_dir if output_dir else "./clustering_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置实验名称
        self.experiment_name = experiment_name if experiment_name else f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 实验结果
        self.experiment_results = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "parameters": {},
            "metrics": {}
        }
        
    def load_embeddings(self) -> np.ndarray:
        """加载数据库中的embeddings"""
        print("正在加载embeddings...")
        cache_file = '/home/dyvm6xra/dyvm6xrauser11/workspace/projects/HKU/Chatbot/Data/Embeddings/embeddings_703df19c43bd6565563071b97e7172ce.npy'
        if os.path.exists(cache_file):
            self.embeddings = np.load(cache_file)
            print(f"从缓存文件加载embeddings，数据形状: {self.embeddings.shape}")
        else:
            result = self.collection.get(include=["embeddings"])
            self.embeddings = np.array(result["embeddings"])
            np.save(cache_file, self.embeddings)
            print(f"从数据库加载embeddings，并保存到缓存文件，数据形状: {self.embeddings.shape}")
        
        self.experiment_results["data_info"] = {
            "embeddings_shape": self.embeddings.shape
        }
        
        return self.embeddings
    
    def reduce_dimensions(self, method: str = "umap", n_components: int = 50, 
                         umap_n_neighbors: int = 50, umap_min_dist: float = 0.2) -> np.ndarray:
        """降维处理
        
        Args:
            method: 降维方法，可选 "umap" 或 "pca"
            n_components: 降维后的维度
            umap_n_neighbors: UMAP的邻居数量参数
            umap_min_dist: UMAP的最小距离参数
        """
        if self.embeddings is None:
            self.load_embeddings()
            
        print(f"使用 {method} 进行降维...")
        
        # 记录降维参数
        self.experiment_results["parameters"]["dimension_reduction"] = {
            "method": method,
            "n_components": n_components
        }
        
        # # 保存降维结果的文件路径
        # reduced_file = os.path.join(self.results_dir, f"{self.experiment_name}_{method}_reduced_embeddings.joblib")
            
        if method.lower() == "umap":
            # 更新实验参数
            self.experiment_results["parameters"]["dimension_reduction"].update({
                "umap_n_neighbors": umap_n_neighbors,
                "umap_min_dist": umap_min_dist
            })
            
            if self.use_gpu:
                print("使用 GPU 加速的 UMAP...")
                # 将 numpy 数组转换为 cupy 数组
                embeddings_gpu = cp.array(self.embeddings)
                
                # 使用 cuML 的 UMAP
                reducer = cuUMAP(
                    n_components=n_components,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    metric='cosine',
                    random_state=42,
                    verbose=True
                )
                
                self.reduced_embeddings = reducer.fit_transform(embeddings_gpu)
                # 将结果转回 CPU
                self.reduced_embeddings = cp.asnumpy(self.reduced_embeddings)
            else:
                # 使用 CPU 版本的 UMAP
                reducer = UMAP(
                    n_components=n_components,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    metric='cosine',
                    random_state=42,
                    n_jobs=-1,
                    low_memory=True,
                    verbose=True
                )
                
                self.reduced_embeddings = reducer.fit_transform(self.embeddings)
                
        elif method.lower() == "pca":
            reducer = PCA(
                n_components=n_components,
                random_state=42,
                svd_solver='randomized'  # 对大数据集更高效
            )
            self.reduced_embeddings = reducer.fit_transform(self.embeddings)
            cumulative_variance = np.cumsum(reducer.explained_variance_ratio_)
            print(f"PCA累积解释方差比: {cumulative_variance[-1]:.4f}")
            
            # 更新实验结果
            self.experiment_results["metrics"]["pca_cumulative_variance"] = float(cumulative_variance[-1])
            
            # 绘制解释方差比曲线
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
            plt.xlabel('主成分数量')
            plt.ylabel('累积解释方差比')
            plt.title('PCA Cumulative Explained Variance Ratio')
            plt.grid(True)
            plt.savefig(os.path.join(self.results_dir, f'{self.experiment_name}_pca_variance_ratio.png'))
            plt.close()
        
        # # 保存降维结果
        # joblib.dump(self.reduced_embeddings, reduced_file)
        # print(f"降维结果已保存至: {reduced_file}")
        
        return self.reduced_embeddings
    
    def cluster_hdbscan(self, min_cluster_size: int = 100, min_samples: int = 10) -> np.ndarray:
        """使用HDBSCAN进行聚类"""
        print("使用HDBSCAN进行聚类...")
        data = self.reduced_embeddings if self.reduced_embeddings is not None else self.embeddings
        
        # 记录聚类参数
        self.experiment_results["parameters"]["clustering"] = {
            "method": "hdbscan",
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples
        }
        
        if self.use_gpu:
            print("使用 GPU 加速的 HDBSCAN...")
            # 将数据转换为 GPU 上的数据
            data_gpu = cp.array(data)
            
            # 使用 cuML 的 HDBSCAN
            clusterer = cuHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_epsilon=0.0,  # cuML HDBSCAN 使用 epsilon 而不是 method
                allow_single_cluster=False,
                verbose=True
            )
            clusterer.fit(data_gpu)
            self.labels = cp.asnumpy(clusterer.labels_)
        else:
            from umap import UMAP
            # 对于大规模数据集的优化参数 (CPU 版本)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True,
                core_dist_n_jobs=64
            )
            self.labels = clusterer.fit_predict(data)
        
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        noise_ratio = n_noise / len(self.labels)
        
        print(f"发现 {n_clusters} 个聚类")
        print(f"噪声点数量: {n_noise} ({noise_ratio:.2%})")
        
        # 计算评估指标
        if n_clusters > 1:  # 需要至少两个簇才能计算
            try:
                silhouette_avg = silhouette_score(data, self.labels, sample_size=10000)
                calinski_avg = calinski_harabasz_score(data, self.labels)
                
                print(f"轮廓系数: {silhouette_avg:.4f}")
                print(f"Calinski-Harabasz指数: {calinski_avg:.4f}")
                
                # 更新实验结果
                self.experiment_results["metrics"].update({
                    "silhouette_score": float(silhouette_avg),
                    "calinski_harabasz_score": float(calinski_avg)
                })
            except Exception as e:
                print(f"计算评估指标时出错: {e}")
        
        # 更新实验结果
        self.experiment_results["metrics"].update({
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": float(noise_ratio)
        })
        
        # 保存聚类结果
        results = {
            'labels': self.labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio
        }
        # joblib.dump(results, os.path.join(self.results_dir, f'{self.experiment_name}_hdbscan_results.joblib'))
        
        return self.labels
    
    def cluster_optics(self, min_samples: int = 50, max_eps: float = 0.5) -> np.ndarray:
        """使用OPTICS进行聚类"""
        print("使用OPTICS进行聚类...")
        data = self.reduced_embeddings if self.reduced_embeddings is not None else self.embeddings
        
        # 记录聚类参数
        self.experiment_results["parameters"]["clustering"] = {
            "method": "optics",
            "min_samples": min_samples,
            "max_eps": max_eps
        }
        
        # 对大规模数据集优化的OPTICS参数
        clustering = OPTICS(
            min_samples=min_samples,
            max_eps=max_eps,
            metric='euclidean',
            n_jobs=-1  # 使用所有CPU核心
        )
        self.labels = clustering.fit_predict(data)
        
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        noise_ratio = n_noise / len(self.labels)
        
        print(f"发现 {n_clusters} 个聚类")
        print(f"噪声点数量: {n_noise} ({noise_ratio:.2%})")
        
        # 计算评估指标
        if n_clusters > 1:  # 需要至少两个簇才能计算
            try:
                silhouette_avg = silhouette_score(data, self.labels, sample_size=10000)
                calinski_avg = calinski_harabasz_score(data, self.labels)
                
                print(f"轮廓系数: {silhouette_avg:.4f}")
                print(f"Calinski-Harabasz指数: {calinski_avg:.4f}")
                
                # 更新实验结果
                self.experiment_results["metrics"].update({
                    "silhouette_score": float(silhouette_avg),
                    "calinski_harabasz_score": float(calinski_avg)
                })
            except Exception as e:
                print(f"计算评估指标时出错: {e}")
        
        # 更新实验结果
        self.experiment_results["metrics"].update({
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": float(noise_ratio)
        })
        
        # 保存结果
        results = {
            'labels': self.labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio
        }
        # joblib.dump(results, os.path.join(self.results_dir, f'{self.experiment_name}_optics_results.joblib'))
        
        return self.labels
    
    def cluster_kmeans(self, n_clusters: int = 100) -> Tuple[np.ndarray, float]:
        """使用K-means进行聚类"""
        print("使用K-means进行聚类...")
        data = self.reduced_embeddings if self.reduced_embeddings is not None else self.embeddings
        
        # 记录聚类参数
        self.experiment_results["parameters"]["clustering"] = {
            "method": "kmeans",
            "n_clusters": n_clusters
        }
        
        if self.use_gpu:
            print("使用 GPU 加速的 KMeans...")
            # 将数据转换为 GPU 上的数据
            data_gpu = cp.array(data)
            
            # 使用 cuML 的 KMeans
            kmeans = cuKMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300,
                verbose=1
            )
            kmeans.fit(data_gpu)
            self.labels = cp.asnumpy(kmeans.labels_)
            inertia = float(kmeans.inertia_)
        else:
            # 对大规模数据优化的 CPU K-means 参数
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300,
                algorithm='elkan',
                n_jobs=-1
            )
            self.labels = kmeans.fit_predict(data)
            inertia = kmeans.inertia_
        
        # 计算评估指标
        try:
            silhouette_avg = silhouette_score(data, self.labels, sample_size=10000)
            calinski_avg = calinski_harabasz_score(data, self.labels)
            
            print(f"聚类数量: {n_clusters}")
            print(f"轮廓系数: {silhouette_avg:.4f}")
            print(f"Calinski-Harabasz指数: {calinski_avg:.4f}")
            
            # 更新实验结果
            self.experiment_results["metrics"].update({
                "silhouette_score": float(silhouette_avg),
                "calinski_harabasz_score": float(calinski_avg),
                "inertia": float(inertia)
            })
        except Exception as e:
            print(f"计算评估指标时出错: {e}")
        
        # 保存结果
        results = {
            'labels': self.labels,
            'inertia': inertia
        }
        # joblib.dump(results, os.path.join(self.results_dir, f'{self.experiment_name}_kmeans_results.joblib'))
        
        return self.labels, silhouette_avg
    
    def find_optimal_k(self, k_range: range) -> int:
        """使用肘部法则和多个评估指标找到最佳的K值"""
        print("寻找最佳K值...")
        data = self.reduced_embeddings if self.reduced_embeddings is not None else self.embeddings
        
        if self.use_gpu:
            # 将数据转换为 GPU 上的数据
            data_gpu = cp.array(data)
        
        results = []
        for k in tqdm(k_range):
            if self.use_gpu:
                kmeans = cuKMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=5,
                    max_iter=300,
                    verbose=0
                )
                kmeans.fit(data_gpu)
                labels = cp.asnumpy(kmeans.labels_)
                inertia = float(kmeans.inertia_)
            else:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=5,
                    algorithm='elkan',
                    n_jobs=-1
                )
                labels = kmeans.fit_predict(data)
                inertia = kmeans.inertia_
            
            # 计算多个评估指标
            silhouette_avg = silhouette_score(data, labels, sample_size=10000)
            calinski_avg = calinski_harabasz_score(data, labels)
            
            results.append({
                'k': k,
                'inertia': float(inertia),
                'silhouette': float(silhouette_avg),
                'calinski': float(calinski_avg)
            })
        
        # 保存结果
        # joblib.dump(results, os.path.join(self.results_dir, f'{self.experiment_name}_kmeans_optimization.joblib'))
        
        # 更新实验结果
        self.experiment_results["kmeans_optimization"] = results
        
        # 绘制评估指标图
        plt.figure(figsize=(15, 5))
        
        # 绘制肘部图
        plt.subplot(1, 3, 1)
        plt.plot([r['k'] for r in results], [r['inertia'] for r in results], 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        
        # 绘制轮廓系数
        plt.subplot(1, 3, 2)
        plt.plot([r['k'] for r in results], [r['silhouette'] for r in results], 'rx-')
        plt.xlabel('k')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        
        # 绘制Calinski-Harabasz指数
        plt.subplot(1, 3, 3)
        plt.plot([r['k'] for r in results], [r['calinski'] for r in results], 'gx-')
        plt.xlabel('k')
        plt.ylabel('Calinski-Harabasz Score')
        plt.title('Calinski-Harabasz Analysis')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.experiment_name}_kmeans_optimization.png'))
        plt.close()
        
        # 基于轮廓系数选择最佳K值
        best_k = max(results, key=lambda x: x['silhouette'])['k']
        
        # 更新实验结果
        self.experiment_results["metrics"]["best_k"] = best_k
        
        return best_k
    
    def visualize_clusters(self, title: str = "Cluster Visualization", sample_size: int = 10000):
        """可视化聚类结果（使用采样来处理大规模数据）"""
        if self.reduced_embeddings is None or self.labels is None:
            print("请先进行降维和聚类")
            return
            
        if self.reduced_embeddings.shape[1] != 2:
            print("只能可视化2维数据，请先使用reduce_dimensions降至2维")
            return
        
        # 对大规模数据进行采样
        if len(self.labels) > sample_size:
            indices = np.random.choice(len(self.labels), sample_size, replace=False)
            reduced_data = self.reduced_embeddings[indices]
            labels = self.labels[indices]
        else:
            reduced_data = self.reduced_embeddings
            labels = self.labels
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=labels,
            cmap='tab20',
            alpha=0.6,
            s=20  # 减小点的大小
        )
        plt.colorbar(scatter)
        plt.title(f"{title}\n(Sampled {sample_size:,} points)")
        
        # 保存图片
        plt.savefig(os.path.join(self.results_dir, f'{self.experiment_name}_cluster_visualization.png'))
        plt.close()
    
    def save_results(self):
        """保存实验结果到JSON文件"""
        # 添加时间戳
        self.experiment_results["end_time"] = datetime.now().isoformat()
        
        # 保存到JSON文件
        results_file = os.path.join(self.results_dir, f'{self.experiment_name}_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.experiment_results, f, indent=2)
        
        print(f"实验结果已保存至: {results_file}")
        
        return results_file

def parse_args():
    parser = argparse.ArgumentParser(description="聚类实验")
    
    # 实验名称
    parser.add_argument("--name", type=str, default=None, help="实验名称")
    
    # 数据库路径
    parser.add_argument("--db_path", type=str, default="./Data/database", help="ChromaDB数据库路径")
    
    # 输出目录
    parser.add_argument("--output_dir", type=str, default="./clustering_results", help="结果输出目录")
    
    # 降维方法
    parser.add_argument("--dim_reduction", type=str, choices=["pca", "umap", "pca_umap"], default="pca_umap", 
                        help="降维方法: pca, umap, 或 pca_umap (两步降维)")
    
    # PCA参数
    parser.add_argument("--pca_components", type=int, default=50, help="PCA降维后的维度")
    
    # UMAP参数
    parser.add_argument("--umap_components", type=int, default=2, help="UMAP降维后的维度")
    parser.add_argument("--umap_neighbors", type=int, default=50, help="UMAP邻居数量")
    parser.add_argument("--umap_min_dist", type=float, default=0.2, help="UMAP最小距离")
    
    # 聚类方法
    parser.add_argument("--clustering", type=str, choices=["hdbscan", "kmeans", "optics"], default="hdbscan", 
                       help="聚类方法: hdbscan, kmeans, 或 optics")
    
    # HDBSCAN参数
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=100, help="HDBSCAN最小簇大小")
    parser.add_argument("--hdbscan_min_samples", type=int, default=10, help="HDBSCAN最小样本数")
    
    # OPTICS参数
    parser.add_argument("--optics_min_samples", type=int, default=50, help="OPTICS最小样本数")
    parser.add_argument("--optics_max_eps", type=float, default=0.5, help="OPTICS最大邻域距离")
    
    # KMeans参数
    parser.add_argument("--kmeans_clusters", type=int, default=0, 
                       help="KMeans聚类数量 (0表示自动寻找最佳K值)")
    parser.add_argument("--kmeans_min_k", type=int, default=50, help="寻找最佳K值的最小K")
    parser.add_argument("--kmeans_max_k", type=int, default=200, help="寻找最佳K值的最大K")
    parser.add_argument("--kmeans_step", type=int, default=10, help="寻找最佳K值的步长")
    
    # 添加 GPU 选项
    parser.add_argument("--use_gpu", action="store_true", help="是否使用 GPU 加速")
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建实验名称
    if not args.name:
        gpu_tag = "gpu" if args.use_gpu else "cpu"
        args.name = f"{args.dim_reduction}_{args.clustering}_{gpu_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 初始化聚类器
    analyzer = ClusterAnalyzer(
        chroma_uri=args.db_path, 
        output_dir=args.output_dir, 
        experiment_name=args.name,
        use_gpu=args.use_gpu
    )
    
    # 1. 加载embeddings
    analyzer.load_embeddings()
    
    # 2. 降维处理
    if args.dim_reduction == "pca":
        # 仅使用PCA
        analyzer.reduce_dimensions(method="pca", n_components=args.pca_components)
    elif args.dim_reduction == "umap":
        # 仅使用UMAP
        analyzer.reduce_dimensions(
            method="umap", 
            n_components=args.umap_components,
            umap_n_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist
        )
    elif args.dim_reduction == "pca_umap":
        # 两步降维: PCA + UMAP
        print("\n=== 第一阶段降维(PCA) ===")
        analyzer.reduce_dimensions(method="pca", n_components=args.pca_components)
        
        print("\n=== 第二阶段降维(UMAP) ===")
        analyzer.reduce_dimensions(
            method="umap", 
            n_components=args.umap_components,
            umap_n_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist
        )
    
    # 3. 聚类
    if args.clustering == "hdbscan":
        print("\n=== HDBSCAN聚类 ===")
        analyzer.cluster_hdbscan(
            min_cluster_size=args.hdbscan_min_cluster_size, 
            min_samples=args.hdbscan_min_samples
        )
    elif args.clustering == "optics":
        print("\n=== OPTICS聚类 ===")
        analyzer.cluster_optics(
            min_samples=args.optics_min_samples,
            max_eps=args.optics_max_eps
        )
    elif args.clustering == "kmeans":
        if args.kmeans_clusters > 0:
            # 使用指定的K值
            print(f"\n=== K-means聚类 (K={args.kmeans_clusters}) ===")
            analyzer.cluster_kmeans(n_clusters=args.kmeans_clusters)
        else:
            # 自动寻找最佳K值
            print("\n=== 寻找最佳K值 ===")
            k_range = range(args.kmeans_min_k, args.kmeans_max_k + 1, args.kmeans_step)
            best_k = analyzer.find_optimal_k(k_range)
            print(f"最佳聚类数量: {best_k}")
            
            print("\n=== K-means聚类 (最佳K) ===")
            analyzer.cluster_kmeans(n_clusters=best_k)
    
    # 4. 可视化结果 (如果降维到了2维)
    if args.umap_components == 2 or (args.dim_reduction == "pca" and args.pca_components == 2):
        analyzer.visualize_clusters(f"{args.clustering.upper()} Clustering Results")
    
    # 5. 保存实验结果
    analyzer.save_results()

if __name__ == "__main__":
    main()
