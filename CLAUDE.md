# Intention Correction 项目文档

## 项目概述

**意图纠正服务（Intention Correction Service）** 是一个基于 RAG（Retrieval-Augmented Generation）架构的智能意图识别系统，专门用于数据资产平台的用户查询意图分析和槽位提取。

### 核心功能
- **意图识别**：分析用户自然语言查询，识别用户意图类型
- **槽位提取**：从用户查询中提取关键信息参数
- **RAG 增强**：通过向量检索召回相似样本，提升识别准确率
- **多意图支持**：支持单次查询中包含多个独立意图的场景

---

## 项目结构

```
intention_correction/
├── llm_milvuslite.py      # RAG服务核心逻辑（主要业务文件）
├── flask_api.py           # Flask REST API接口
├── init_milvus.py         # Milvus向量数据库初始化脚本
├── milvuslite_test.py     # Milvus连接测试脚本
├── config.ini             # 配置文件
├── data.json              # 意图识别训练样本数据（85条）
├── start_flask_api.sh     # Gunicorn启动脚本
├── curl.txt               # API调用示例
├── milvus.db              # Milvus Lite数据库文件
├── logs/                  # 日志目录
├── responses/             # RAG响应结果存储目录（约560个历史响应）
├── pymilvus2.6.3/         # pymilvus依赖包
├── milvuslite2.5.1/       # milvus-lite依赖包
├── flask_cors-6.0.1-py3-none-any.whl    # Flask-CORS依赖
└── gunicorn-23.0.0-py3-none-any.whl     # Gunicorn依赖
```

---

## 技术架构

### 技术栈
| 组件 | 技术选型 | 说明 |
|------|----------|------|
| Web框架 | Flask + Flask-CORS | REST API服务 |
| WSGI服务器 | Gunicorn | 生产环境部署，4 workers |
| 向量数据库 | Milvus Lite | 轻量级向量存储与检索 |
| 大语言模型 | Qwen3-32B | 意图识别与生成 |
| Embedding模型 | bge-m3 | 文本向量化（1024维） |
| Rerank模型 | bge-rerank-v2-m3 | 召回结果重排序 |

### 系统架构流程

```
用户查询 → Flask API → RAG服务
                          ↓
                    1. Embedding生成（bge-m3）
                          ↓
                    2. Milvus向量检索（top_k=10）
                          ↓
                    3. Rerank重排序（取top 5）
                          ↓
                    4. 构建RAG Prompt
                          ↓
                    5. LLM意图识别（Qwen3-32B）
                          ↓
                    返回JSON结果
```

---

## 配置说明

### config.ini 配置项

```ini
[DEFAULT]
# 大语言模型配置
MODEL_API_URL = http://localhost:8891/v1/chat/completions
MODEL_NAME = Qwen3-32B
TIMEOUT = 300

# Embedding模型配置
EMBEDDING_API_URL = http://localhost:54114/v1/embeddings
EMBEDDING_MODEL_NAME = bge-m3

# Rerank模型配置
RERANK_API_URL = http://localhost:54113/v1/rerank
RERANK_MODEL_NAME = bge-rerank-v2-m3

# Milvus配置
MILVUS_DB_PATH = ./milvus.db
MILVUS_COLLECTION_NAME = intention

# 训练数据
INTENTION_EXAMPLE = data.json

# 日志配置
LOG_DIR = logs
LOG_FILE = app.log

# 默认响应
DEFAULT_RESPONSE = 对不起，纠正服务暂时不可用。请稍后再试。
```

---

## 核心模块详解

### 1. RAGService 类 (`llm_milvuslite.py`)

核心服务类，提供完整的 RAG 查询流程。

**主要方法**：

| 方法 | 功能 |
|------|------|
| `__init__()` | 初始化配置、日志、Milvus连接 |
| `_get_embedding(text)` | 调用Embedding API获取文本向量 |
| `_search_similar_documents(query, top_k)` | Milvus向量相似度检索 |
| `call_llm(prompt, temperature, enable_thinking)` | 调用大语言模型 |
| `call_rerank(query, documents)` | 调用Rerank模型重排序 |
| `rag_query(query, top_k, temperature)` | 完整RAG查询流程 |
| `_save_response(query, response)` | 保存响应结果到文件 |

**向量搜索参数**：
- `metric_type`: COSINE（余弦相似度）
- `nprobe`: 10
- `limit`: 默认10条

### 2. Flask API (`flask_api.py`)

**API 端点**：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/status` | GET | 服务状态检查（包含各组件状态） |
| `/rag_query` | POST | 单次RAG查询 |
| `/rag_query_batch` | POST | 批量RAG查询（最多10条） |

**请求示例**：
```bash
curl -X POST http://localhost:8890/rag_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "M域的所有标签资产",
    "top_k": 5,
    "temperature": 0.7,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

**响应格式**：
```json
{
  "result": {
    "text": "{\"intent\": \"31\", \"slots\": {\"BusinessDomain\": \"M域\", \"AssetType\": \"标签\"}, \"query\": \"M域的所有标签资产\"}"
  }
}
```

### 3. Milvus初始化 (`init_milvus.py`)

`MilvusIntentionIngestor` 类负责：
- 创建/重置 Milvus 集合
- 加载训练数据 (`data.json`)
- 批量生成 Embedding 并插入向量数据库

**集合Schema**：
```python
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="input", dtype=DataType.VARCHAR, max_length=65535),  # 用户查询
    FieldSchema(name="output", dtype=DataType.JSON),                       # 意图+槽位
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)  # 向量
]
```

**索引配置**：
- `index_type`: IVF_FLAT
- `metric_type`: COSINE
- `nlist`: 128

---

## 意图体系

### 意图编码表

| 编码 | 意图名称 | 功能描述 |
|------|----------|----------|
| 31 | 资产基础检索 | 基于名称/ID/域/类型等条件查找资产 |
| 32 | 资产元数据查询 | 查询业务口径/技术口径/负责人等元数据 |
| 33 | 资产质量与价值查询 | 查询价值评分/星级/质量稽核 |
| 34 | 资产血缘关系查询 | 查询上游依赖/下游应用/血缘图 |
| 35 | 资产使用与工单查询 | 查询订阅/收藏/工单进度/审批 |
| 36 | 场景与标签推荐 | 基于业务专区/场景推荐资产 |
| 37 | 资产复合对比与筛选 | 对比两个资产或多条件筛选 |
| 38 | 平台规则与帮助 | 平台操作/帮助/名词解释 |
| 39 | 助手能力与帮助 | 助手自身能力范围、功能清单 |
| 40 | OOD兜底 | 用户查询与意图清单不匹配时 |
| 50 | 多意图并发查询 | 一句话包含多个独立查询意图 |

### 槽位定义

| 槽位 | 说明 | 示例 |
|------|------|------|
| AssetName | 资产完整名称 | `[在线公司]终端激活信息(日)` |
| AssetId | 资产ID（纯数字） | `1169315655200010241` |
| TableEnglishName | 表英文名 | `ESD_D_CUS_HOME_DEVICE_B` |
| TableChineseName | 表中文名 | `客户星级数据月全量加工表` |
| MetadataItem | 元数据项 | `业务口径`/`技术口径`/`负责人` |
| FieldName | 字段名称 | `user_id`/`order_id` |
| CoreDataItem | 业务概念 | `5G登网`/`移网用户是否活跃` |
| BusinessDomain | 业务域 | `M域`/`O域`/`B域` |
| AssetType | 资产类型 | `标签`/`数据表`/`模型资产`/`指标` |
| BusinessZone | 业务专区 | `公众智慧运营`/`一线赋能专区` |
| FilterCondition | 筛选条件 | `五星`/`高价值`/`最近一周更新的` |
| LineageDirection | 血缘方向 | `上游`/`下游`/`血缘图` |
| OwnerTenant | 归属租户 | `总部`/`分公司A` |
| AssetRanking | 综合排行 | `最新`/`本周上新`/`热门` |
| AssertAdmin | 资产管理员 | `张三` |
| AssertPublisher | 资产发布人 | `李四` |
| DataDomain | 归属数据域 | `固网视图`/`客户视图` |
| DataLayer | 归属数据层 | `ESD`/`DM`/`SRC`/`DWD` |
| AssetOpenScope | 资产开放范围 | `公共`/`私有`/`保护` |

---

## 输出格式示例

### 单意图
```json
{"intent": "31", "slots": {"BusinessDomain": "M域", "AssetType": "标签"}, "query": "M域的所有标签资产"}
{"intent": "32", "slots": {"AssetName": "宽带提质速率(月)", "MetadataItem": "业务口径"}, "query": "宽带提质速率(月)的业务口径是什么？"}
{"intent": "40", "slots": {}, "query": "今天天气怎么样？"}
```

### 多意图（Intent 50）
```json
{
  "intent": "50",
  "slots": [
    {"intent": "31", "slots": {"AssetName": "家庭圈模型"}},
    {"intent": "32", "slots": {"AssetName": "智慧家庭工程师信息", "MetadataItem": "负责人"}}
  ],
  "query": "查询家庭圈模型的详情,以及智慧家庭工程师信息的负责人"
}
```

---

## 部署与运行

### 启动服务

**开发模式**：
```bash
python flask_api.py
# 服务监听: 0.0.0.0:8890
```

**生产模式（Gunicorn）**：
```bash
./start_flask_api.sh
# 或直接执行：
nohup gunicorn --workers 4 --bind 0.0.0.0:8890 --timeout 300 --log-level info flask_api:app > ./logs/gunicorn.log 2>&1 &
```

### 初始化向量数据库

```bash
python init_milvus.py
```
此脚本会：
1. 连接/创建 Milvus Lite 数据库
2. 删除并重建 `intention` 集合
3. 加载 `data.json` 中的训练数据
4. 批量生成 Embedding 并插入

### 测试连接

```bash
python milvuslite_test.py
```

---

## 依赖要求

### Python 包
```
flask
flask-cors
gunicorn
requests
pymilvus>=2.6.3
milvus-lite>=2.5.1
configparser
numpy
```

### 外部服务依赖
- **LLM API**: `http://localhost:8891/v1/chat/completions` (Qwen3-32B)
- **Embedding API**: `http://localhost:54114/v1/embeddings` (bge-m3)
- **Rerank API**: `http://localhost:54113/v1/rerank` (bge-rerank-v2-m3)

---

## 已知问题与注意事项

### 1. Rerank 模型名称问题
日志显示 `bge-reranker-v2-m3` 不存在，配置文件中的名称与实际部署可能不一致。
- 配置值: `bge-rerank-v2-m3`
- 错误提示: `The model 'bge-reranker-v2-m3' does not exist`

**解决方案**：确认 Rerank 服务实际支持的模型名称，更新 `config.ini`。

### 2. 布尔值语法问题（已修复）
历史日志显示 `enable_thinking: false` 使用了 JavaScript 语法，应为 Python 的 `False`。

### 3. 数据量
当前 Milvus 集合包含 **85 个实体**（训练样本）。

---

## 历史记录

### 2025-11-26
- Claude 完成项目全面分析
- 创建 CLAUDE.md 项目文档
- 项目核心功能：意图识别 + 槽位提取 + RAG增强
- 技术栈：Flask + Milvus Lite + Qwen3-32B + bge-m3

---

## 文件说明快速索引

| 文件 | 行数 | 核心功能 |
|------|------|----------|
| `llm_milvuslite.py` | ~837行 | RAGService类，包含完整的Prompt模板 |
| `flask_api.py` | ~392行 | Flask API，单例模式管理RAG服务 |
| `init_milvus.py` | ~313行 | MilvusIntentionIngestor类 |
| `data.json` | ~1217行 | 85条意图识别训练样本 |
| `config.ini` | 15行 | 所有外部服务配置 |
