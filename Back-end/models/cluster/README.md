# 主题聚类实验框架

本框架用于评估不同的降维和聚类算法组合在主题聚类任务上的表现。

## 实验架构

实验框架主要由以下部分组成：

1. `cluster_topic_exp.py`：实验主程序，支持通过命令行参数配置各种实验参数
2. `run_experiments.sh`：批量运行多组实验的脚本
3. `clustering_results/analyze_results.py`：结果分析脚本，将在实验结束后自动生成

## 评估指标

实验使用以下指标评估聚类质量：

- **轮廓系数（Silhouette Score）**：衡量簇内紧密度和簇间分离度，取值范围[-1,1]，越大越好
- **Calinski-Harabasz指数**：簇间方差与簇内方差的比值，越大越好
- **聚类数量**：发现的聚类数量
- **噪声点比例**：被识别为噪声的数据比例（仅适用于HDBSCAN和OPTICS）

## 实验参数

以下是可配置的主要参数：

### 降维方法
- **PCA**：主成分分析
  - `pca_components`：降维后的维度

- **UMAP**：Uniform Manifold Approximation and Projection
  - `umap_components`：降维后的维度
  - `umap_neighbors`：邻居数量（影响全局结构保存程度）
  - `umap_min_dist`：最小距离（影响局部结构保存程度）

### 聚类算法
- **HDBSCAN**：基于密度的层次聚类
  - `hdbscan_min_cluster_size`：最小簇大小
  - `hdbscan_min_samples`：最小样本数（影响噪声点识别）

- **K-means**：基于距离的聚类
  - `kmeans_clusters`：聚类数量（设为0时自动寻找最佳K值）
  - `kmeans_min_k`/`kmeans_max_k`/`kmeans_step`：寻找最佳K值的范围和步长

- **OPTICS**：有序点以识别聚类结构
  - `optics_min_samples`：最小样本数
  - `optics_max_eps`：最大邻域距离

## 使用方法

### 运行单个实验

```bash
python cluster_topic_exp.py --name 实验名称 --dim_reduction 降维方法 --clustering 聚类方法 [其他参数]
```

例如，运行PCA降维+HDBSCAN聚类实验：

```bash
python cluster_topic_exp.py --name pca50_hdbscan --dim_reduction pca --pca_components 50 --clustering hdbscan
```

### 批量运行实验

```bash
./run_experiments.sh
```

运行后，所有实验结果将保存在`clustering_results`目录下，包含：
- 降维后的嵌入向量
- 聚类结果和标签
- 评估指标和可视化结果
- JSON格式的完整实验记录

### 分析实验结果

实验完成后，运行分析脚本：

```bash
python clustering_results/analyze_results.py
```

该脚本将生成：
- 实验结果概览表格
- Excel格式的详细结果
- 不同实验的轮廓系数和CH指数对比图

## 定制实验

要添加新的实验组合，可以编辑`run_experiments.sh`脚本，添加新的实验配置。 