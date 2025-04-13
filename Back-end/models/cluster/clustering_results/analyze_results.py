#!/usr/bin/env python
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# 结果目录
results_dir = "./clustering_results"

# 加载所有实验结果
results = []
for filename in os.listdir(results_dir):
    if filename.endswith("_results.json"):
        with open(os.path.join(results_dir, filename), 'r') as f:
            try:
                data = json.load(f)
                results.append(data)
            except json.JSONDecodeError:
                print(f"无法解析: {filename}")

if not results:
    print("未找到任何结果文件")
    exit(1)

# 提取关键指标到DataFrame
data = []
for res in results:
    row = {
        "实验名称": res.get("experiment_name", "未知"),
        "降维方法": res.get("parameters", {}).get("dimension_reduction", {}).get("method", "未知"),
        "降维维度": res.get("parameters", {}).get("dimension_reduction", {}).get("n_components", "未知"),
        "聚类方法": res.get("parameters", {}).get("clustering", {}).get("method", "未知"),
    }
    
    # 添加聚类特定参数
    if row["聚类方法"] == "hdbscan":
        row["min_cluster_size"] = res.get("parameters", {}).get("clustering", {}).get("min_cluster_size", "未知")
        row["min_samples"] = res.get("parameters", {}).get("clustering", {}).get("min_samples", "未知")
    elif row["聚类方法"] == "kmeans":
        row["n_clusters"] = res.get("parameters", {}).get("clustering", {}).get("n_clusters", "未知")
    
    # 添加UMAP特定参数
    if row["降维方法"] == "umap":
        row["umap_n_neighbors"] = res.get("parameters", {}).get("dimension_reduction", {}).get("umap_n_neighbors", "未知")
        row["umap_min_dist"] = res.get("parameters", {}).get("dimension_reduction", {}).get("umap_min_dist", "未知")
    
    # 添加指标
    row["聚类数量"] = res.get("metrics", {}).get("n_clusters", "未知")
    row["噪声比例"] = res.get("metrics", {}).get("noise_ratio", "未知")
    row["轮廓系数"] = res.get("metrics", {}).get("silhouette_score", "未知")
    row["CH指数"] = res.get("metrics", {}).get("calinski_harabasz_score", "未知")
    row["最佳K值"] = res.get("metrics", {}).get("best_k", "未适用")
    
    data.append(row)

df = pd.DataFrame(data)

# 生成结果报告
print("=" * 80)
print("实验结果分析")
print("=" * 80)

# 打印表格
print("\n实验结果概览:")
print(tabulate(df, headers="keys", tablefmt="pipe", showindex=False))

# 保存到Excel
excel_file = os.path.join(results_dir, "实验结果分析.xlsx")
df.to_excel(excel_file, index=False)
print(f"\n详细结果已保存到: {excel_file}")

# 绘制轮廓系数对比
plt.figure(figsize=(12, 6))
sns.barplot(x="实验名称", y="轮廓系数", data=df)
plt.xticks(rotation=90)
plt.title("不同实验的轮廓系数对比")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "轮廓系数对比.png"))

# 绘制CH指数对比
plt.figure(figsize=(12, 6))
sns.barplot(x="实验名称", y="CH指数", data=df)
plt.xticks(rotation=90)
plt.title("不同实验的Calinski-Harabasz指数对比")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "CH指数对比.png"))

print("\n分析图表已生成")
