## 1. 数据预处理：将原始医疗文本数据转换为标准化的问答对
```bash
python Back-end/data_processing/process_qa.py
```

### 数据预处理说明

#### 功能描述
- 将原始的医疗文本数据清洗并转换为标准化的问答(QA)对格式
- 支持处理多个来源的数据（Healthline文章、MedQA数据集等）
- 使用TF-IDF算法提取关键词
- 生成问答对的倒排索引，便于后续检索
- 自动去除重复和无效的问答对

#### 数据要求
数据目录结构应如下：
```
./Data/
├── Raw/
│   ├── Healthline/
│   │   └── healthline_articles_text.csv
│   └── MedQA/
│       └── *.csv
└── Processed/
    └── cleaned_qa/
```

#### 输入文件格式
1. Healthline文章数据 (CSV格式):
   - 必需列：'title'（作为问题）, 'content'（作为答案）

2. MedQA数据 (CSV格式):
   - 必需列：'Question'/'question', 'Answer'/'answer'

#### 输出文件
处理后的数据将保存在 `./Data/Processed/cleaned_qa/` 以及 `keywords/` 目录下：
1. `qa_database.json`: 包含所有处理后的问答对
   ```json
   [
     {
       "id": "unique_id",
       "source": "数据来源",
       "question": "清洗后的问题",
       "answer": "清洗后的答案",
       "keywords": ["关键词1", "关键词2", ...]
     },
     ...
   ]
   ```
2. `keyword_index.json`: 关键词倒排索引
   ```json
   {
     "关键词1": ["qa_id1", "qa_id2", ...],
     "关键词2": ["qa_id3", "qa_id4", ...],
     ...
   }
   ```

#### 数据处理步骤
1. 文本清理：
   - 移除HTML标签
   - 规范化标点符号
   - 去除多余空白字符
   - 转换为小写

2. 关键词提取：
   - 使用NLTK进行分词
   - 去除停用词（包含通用停用词和医疗领域特定停用词）
   - 使用TF-IDF算法提取重要关键词

3. 质量控制：
   - 过滤无效的问答对
   - 去除重复内容
   - 生成唯一标识符

#### 依赖安装
```bash
pip install nltk scikit-learn pandas tqdm
```

## 2. 文本向量化：将问答对转换为高维向量并建立向量数据库
```bash
python Back-end/models/vectorize.py
```

### 向量化处理说明

#### 功能描述
- 使用医疗领域预训练模型 BioBERT-MNLI 将问答对转换为高维向量
- 构建高效的向量数据库，支持语义相似度检索
- 实现向量缓存机制，避免重复计算
- 多线程并行处理，提高数据处理效率
- 支持增量更新和数据持久化

#### 系统架构
1. 模型配置：
   - 预训练模型：`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`
   - 向量维度：768维
   - 设备支持：自动检测并使用GPU/CPU
   - 内存优化：批处理机制

2. 向量化流程：
   - 数据加载和预处理
   - 文本向量生成（支持缓存）
   - 向量标准化（L2正则化）
   - 批量处理（BATCH_SIZE=1024）
   - 多线程并行插入（MAX_WORKERS=8）

3. 数据库优化：
   - 使用ChromaDB作为向量存储
   - HNSW索引配置：
     - 空间度量：cosine
     - 构建参数：ef=100
     - 搜索参数：ef=128
     - 图连接度：M=32/64
   - 分批持久化（40000条/批）

#### 数据目录结构
```
./Data/
├── database/          # ChromaDB持久化存储
├── Embeddings/        # 向量缓存目录
└── Processed/
    ├── keywords/      # 关键词索引
    └── cleaned_qa/    # 预处理后的QA数据
```

#### 向量化过程
1. 数据准备：
   - 加载问答数据和关键词索引
   - 合并问题、答案和关键词为统一文本
   - 构建元数据（来源、ID、关键词等）

2. 向量生成：
   - 计算数据哈希值用于缓存验证
   - 检查并加载已缓存的向量
   - 批量生成新的文本向量
   - 自动保存向量到缓存目录

3. 数据库构建：
   - 创建临时内存数据库
   - 多线程并行插入数据
   - 分批持久化到磁盘
   - 自动进度跟踪和内存监控

#### 性能指标
- 批处理大小：1024条/批
- 插入批次：1024条/批
- 持久化批次：40000条/批
- 并行线程数：8
- 向量维度：768
- 内存占用监控：实时跟踪

#### 依赖要求
```bash
pip install torch sentence-transformers chromadb tqdm numpy psutil
```

#### 硬件建议
- 推荐配置：NVIDIA GPU（8GB+ VRAM）
- 最低配置：8GB系统内存（CPU模式）

#### 使用说明
1. 运行向量化处理：
```bash
python Back-end/models/vectorize.py
```

2. 测试数据库：
```bash
python Back-end/models/test_db.py
```

测试程序将：
- 验证数据库完整性
- 展示随机样本数据
- 执行示例查询
- 显示相似度得分

示例查询结果：
```
使用查询词 'diabetes' 的结果:

结果 1:
----------------------------------------
相似度得分: 0.6597

文档内容:
Question: what are the treatments for diabetes
Answer: diabetes is a very serious disease over time diabetes that is not well managed causes serious damage to the eyes kidneys nerves and heart gums and teeth if you have diabetes you are more likely than someone who does not have diabetes to have heart disease or a stroke people with diabetes also tend to develop heart disease or stroke at an earlier age than others the best way to protect yourself from the serious complications of diabetes is to manage your blood glucose blood pressure and cholesterol and avoid smoking it is not always easy but people who make an ongoing effort to manage their diabetes can greatly improve their overall health
Keywords: diabetes, heart, serious, blood, manage, people, stroke, best, complication, damage

元数据:
{
    'item_id': 'MedicalQuestionAnswering_5480',
    'keywords': 'diabetes, heart, serious, blood, manage, people, stroke, best, complication, damage',
    'source': 'diabetes',
    'type': 'qa'
}
```

这个示例展示了：
- 语义相似度搜索的效果
- 结果包含完整的问答对
- 相关的元数据信息
- 关键词提取结果

## 3. 主题聚类：对向量化后的问答对进行主题聚类
```bash
python Back-end/models/cluster_topic.py
```

### 主题聚类说明

#### 功能描述
- 使用 UMAP 进行高维向量降维
- 采用 HDBSCAN 算法进行密度聚类
- 支持 GPU 加速（如果可用）
- 自动将聚类结果更新到向量数据库
- 聚类结果持久化和缓存机制

#### 系统架构
1. 降维配置：
   - 算法：UMAP（Uniform Manifold Approximation and Projection）
   - 输出维度：50
   - 邻居数：50
   - 最小距离：0.2
   - 距离度量：cosine
   - GPU 支持：自动检测

2. 聚类配置：
   - 算法：HDBSCAN
   - 最小簇大小：100
   - 最小样本数：10
   - 距离度量：euclidean
   - 簇选择方法：EOM (Excess of Mass)
   - 并行处理：支持多核心

3. 数据流程：
   - 从 ChromaDB 加载向量
   - UMAP 降维处理
   - HDBSCAN 聚类
   - 结果写回数据库

#### 数据目录结构
```
./Data/
├── database/          # ChromaDB存储
├── Embeddings/        # 降维结果缓存
└── Processed/
    └── clusters/      # 聚类结果
```

#### 性能优化
- 自动 GPU 加速支持
- 降维结果缓存
- 并行计算优化
- 内存使用优化

#### 依赖要求
```bash
pip install umap-learn hdbscan joblib
# GPU加速（可选）
pip install cupy cuml
```

#### 使用说明
1. 运行聚类处理：
```bash
python Back-end/models/cluster_topic.py
```

2. 聚类结果：
- 每个文档将被分配一个聚类标签
- 标签格式：`cluster_N`（N为聚类编号）
- 噪声点标记为 `noise`
- 结果存储在向量数据库的元数据中

示例查询结果：
```json
{
    "id": "doc_id",
    "metadata": {
        "cluster": "cluster_1",
        "source": "原始来源",
        "keywords": ["关键词1", "关键词2", ...]
    }
}
```

## 4. 主题分类：训练主题分类器并进行预测
```bash
python Back-end/models/topic_classification.py
```

### 主题分类说明

#### 功能描述
- 基于聚类结果训练随机森林分类器
- 使用5折交叉验证评估模型性能
- 支持模型持久化和版本管理
- 提供新文档的主题预测功能
- 自动整合到向量数据库系统

#### 系统架构
1. 分类器配置：
   - 算法：随机森林（RandomForest）
   - 树的数量：100
   - 并行处理：支持多核心
   - 评估指标：准确率、宏F1、加权F1
   - 交叉验证：5折分层验证

2. 训练流程：
   - 从ChromaDB加载向量和标签
   - 数据预处理和清洗
   - 模型训练和验证
   - 性能评估和报告
   - 最优模型保存

3. 预测流程：
   - 加载训练好的模型
   - 新文档向量化
   - 主题预测
   - 结果更新到数据库

#### 数据目录结构
```
./Data/
├── database/          # ChromaDB存储
└── models/            # 模型存储目录
    └── topic_classifier_*.joblib  # 带时间戳的模型文件
```

#### 性能指标
- 交叉验证折数：5
- 评估指标：
  - 准确率（Accuracy）
  - 宏平均F1（Macro-F1）
  - 加权平均F1（Weighted-F1）
- 模型版本控制：时间戳命名

#### 依赖要求
```bash
pip install scikit-learn joblib numpy
```

#### 使用说明
1. 训练分类器：
```bash
python Back-end/models/topic_classification.py
```

2. 模型输出：
- 训练过程中会显示每折的详细性能报告
- 最终模型保存在`./models/`目录
- 文件名格式：`topic_classifier_YYYYMMDD_HHMMSS.joblib`

示例输出：
```
正在加载数据...
数据加载完成，特征形状: (N, 768)
类别数量: K

开始5折交叉验证...
第 1 折验证:
分类报告:
              precision    recall  f1-score 
   cluster_0       0.85      0.83      0.84       
   cluster_1       0.82      0.80      0.81      
...

总体性能:
平均准确率: 0.8234 ± 0.0256
平均宏F1分数: 0.8156 ± 0.0278
平均加权F1分数: 0.8245 ± 0.0267

模型已保存到: ./models/topic_classifier_20240315_143022.joblib
```

这个分类系统将：
- 自动处理新增的医疗问答文档
- 保持与现有聚类体系的一致性
- 支持增量更新和模型迭代
- 提供可靠的主题预测服务

