#!/bin/bash

# 创建结果目录
RESULTS_DIR="./clustering_results"
mkdir -p $RESULTS_DIR

# 数据库路径
DB_PATH="/home/dyvm6xra/dyvm6xrauser11/workspace/projects/HKU/Chatbot/Data/database"

# 定义日志文件
LOG_FILE="${RESULTS_DIR}/experiments_$(date +%Y%m%d_%H%M%S).log"

# 函数：运行实验并记录日志
run_experiment() {
    echo "运行实验: $1" | tee -a $LOG_FILE
    echo "命令: $2" | tee -a $LOG_FILE
    echo "开始时间: $(date)" | tee -a $LOG_FILE
    
    # 运行命令并将输出同时写入日志
    eval $2 | tee -a $LOG_FILE
    
    echo "结束时间: $(date)" | tee -a $LOG_FILE
    echo "==========================================" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
}

echo "================ 聚类实验 ================" | tee -a $LOG_FILE
echo "开始时间: $(date)" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# ==== 实验1：单独PCA降维 + 不同聚类方法 ====
# # PCA(50) + HDBSCAN
# run_experiment "PCA(50) + HDBSCAN" \
#     "python cluster_topic_exp.py --name pca50_hdbscan --dim_reduction pca --pca_components 50 --clustering hdbscan --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu"

# PCA(8) + KMEANS(自动寻找最佳K)
run_experiment "PCA(4) + KMEANS(自动寻找最佳K)" \
    "python cluster_topic_exp.py --name pca4_kmeans_auto --dim_reduction pca --pca_components 4 --clustering kmeans --kmeans_min_k 4 --kmeans_max_k 31 --kmeans_step 2 --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu"


# # # ==== 实验2：单独UMAP降维 + 不同聚类方法 ====
# # UMAP(2, n_neighbors=50) + HDBSCAN
# run_experiment "UMAP(2, n_neighbors=50) + HDBSCAN" \
#     "python cluster_topic_exp.py --name umap2_nn50_hdbscan --dim_reduction umap --umap_components 2 --umap_neighbors 50 --umap_min_dist 0.2 --clustering hdbscan --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu"

# # UMAP(2, n_neighbors=30) + HDBSCAN
# run_experiment "UMAP(2, n_neighbors=30) + HDBSCAN" \
#     "python cluster_topic_exp.py --name umap2_nn30_hdbscan --dim_reduction umap --umap_components 2 --umap_neighbors 30 --umap_min_dist 0.2 --clustering hdbscan --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu"

# # # UMAP(2, n_neighbors=50) + KMEANS(自动寻找最佳K)
# run_experiment "UMAP(2, n_neighbors=50) + KMEANS(自动寻找最佳K)" \
#     "python cluster_topic_exp.py --name umap2_nn50_kmeans_auto --dim_reduction umap --umap_components 2 --umap_neighbors 50 --umap_min_dist 0.2 --clustering kmeans --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu --kmeans_min_k 10 --kmeans_max_k 210 --kmeans_step 20"


# run_experiment "UMAP(4, n_neighbors=50) + KMEANS(自动寻找最佳K)" \
#     "python cluster_topic_exp.py --name umap4_nn50_kmeans_auto --dim_reduction umap --umap_components 4 --umap_neighbors 50 --umap_min_dist 0.2 --clustering kmeans --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu --kmeans_min_k 10 --kmeans_max_k 210 --kmeans_step 20"

# run_experiment "UMAP(16, n_neighbors=50) + KMEANS(自动寻找最佳K)" \
#     "python cluster_topic_exp.py --name umap16_nn50_kmeans_auto --dim_reduction umap --umap_components 16 --umap_neighbors 50 --umap_min_dist 0.2 --clustering kmeans --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu --kmeans_min_k 10 --kmeans_max_k 210 --kmeans_step 20"

# run_experiment "UMAP(32, n_neighbors=50) + KMEANS(自动寻找最佳K)" \
#     "python cluster_topic_exp.py --name umap32_nn50_kmeans_auto --dim_reduction umap --umap_components 32 --umap_neighbors 50 --umap_min_dist 0.2 --clustering kmeans --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu --kmeans_min_k 10 --kmeans_max_k 210 --kmeans_step 20"


# # # UMAP(100, n_neighbors=50) + KMEANS(自动寻找最佳K)
# run_experiment "UMAP(100, n_neighbors=50) + KMEANS(自动寻找最佳K)" \
#     "python cluster_topic_exp.py --name umap100_nn50_kmeans_auto --dim_reduction umap --umap_components 100 --umap_neighbors 50 --umap_min_dist 0.2 --clustering kmeans --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu"

# # UMAP(64, n_neighbors=50) + KMEANS(自动寻找最佳K)
# run_experiment "UMAP(64, n_neighbors=50) + KMEANS(自动寻找最佳K)" \
#     "python cluster_topic_exp.py --name umap64_nn50_kmeans_auto --dim_reduction umap --umap_components 64 --umap_neighbors 50 --umap_min_dist 0.2 --clustering kmeans --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu"

# # ==== 实验3：两步降维 PCA+UMAP + 不同聚类方法 ====
# PCA(50) + UMAP(2) + HDBSCAN(min_cluster_size=100)
# run_experiment "PCA(50) + UMAP(2) + HDBSCAN(min_cluster_size=100)" \
#     "python cluster_topic_exp.py --name pca50_umap2_hdbscan100 --dim_reduction pca_umap --pca_components 50 --umap_components 2 --clustering hdbscan --hdbscan_min_cluster_size 100 --hdbscan_min_samples 10 --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu"

# # PCA(50) + UMAP(2) + HDBSCAN(min_cluster_size=50)
# run_experiment "PCA(50) + UMAP(2) + HDBSCAN(min_cluster_size=50)" \
#     "python cluster_topic_exp.py --name pca50_umap2_hdbscan50 --dim_reduction pca_umap --pca_components 50 --umap_components 2 --clustering hdbscan --hdbscan_min_cluster_size 50 --hdbscan_min_samples 5 --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu"

# # PCA(50) + UMAP(2) + KMEANS(自动寻找最佳K)
# run_experiment "PCA(50) + UMAP(2) + KMEANS(自动寻找最佳K)" \
#     "python cluster_topic_exp.py --name pca50_umap2_kmeans_auto --dim_reduction pca_umap --pca_components 50 --umap_components 2 --clustering kmeans --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu"

# # PCA(100) + UMAP(2) + HDBSCAN
# run_experiment "PCA(100) + UMAP(2) + HDBSCAN" \
#     "python cluster_topic_exp.py --name pca100_umap2_hdbscan --dim_reduction pca_umap --pca_components 100 --umap_components 2 --clustering hdbscan --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu"

# # ==== 实验4：不同UMAP参数 + HDBSCAN ====
# # PCA(50) + UMAP(2, min_dist=0.1) + HDBSCAN
# run_experiment "PCA(50) + UMAP(2, min_dist=0.1) + HDBSCAN" \
#     "python cluster_topic_exp.py --name pca50_umap2_md01_hdbscan --dim_reduction pca_umap --pca_components 50 --umap_components 2 --umap_min_dist 0.1 --clustering hdbscan --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu"

# # PCA(50) + UMAP(2, min_dist=0.5) + HDBSCAN
# run_experiment "PCA(50) + UMAP(2, min_dist=0.5) + HDBSCAN" \
#     "python cluster_topic_exp.py --name pca50_umap2_md05_hdbscan --dim_reduction pca_umap --pca_components 50 --umap_components 2 --umap_min_dist 0.5 --clustering hdbscan --db_path $DB_PATH --output_dir $RESULTS_DIR --use_gpu"

# ==== 实验5：结果分析脚本 ====
echo "所有实验完成，生成分析报告..." | tee -a $LOG_FILE

cat > ${RESULTS_DIR}/analyze_results.py << 'EOL'
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
EOL

# 设置脚本执行权限
chmod +x ${RESULTS_DIR}/analyze_results.py

echo "实验全部完成！" | tee -a $LOG_FILE
echo "总结果保存在: ${RESULTS_DIR}" | tee -a $LOG_FILE
echo "您可以运行以下命令分析结果:" | tee -a $LOG_FILE
echo "python ${RESULTS_DIR}/analyze_results.py" | tee -a $LOG_FILE 