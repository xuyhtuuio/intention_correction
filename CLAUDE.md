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

### 2025-11-26（第三次更新）
- **修复 RAG 召回 bug**：向量召回只返回 `input` 字段，缺少 `output`（意图+槽位）导致 few-shot 学习失效
- 修改 `_search_similar_documents` 方法：
  - `output_fields=["input"]` → `output_fields=["input", "output"]`
  - 返回类型从 `List[str]` 改为 `List[Dict[str, Any]]`
- 修改 `rag_query` 方法：
  - Rerank 调用适配新的数据结构（提取 input 文本进行排序，通过 index 映射回完整文档）
  - 重构上下文构建逻辑，生成 few-shot 示例格式：
    ```
    示例 1:
    用户查询: 查找标签资产
    正确输出: {"intent": "31", "slots": {"AssetType": "标签"}, "query": "查找标签资产"}
    ```
- 优化 RAG 提示词，明确告知模型参考示例的作用

### 2025-11-26（第二次更新）
- 设计并完善**自动校准系统方案**
- 核心模块：反馈收集器、评估引擎、自动校准器、报告生成器
- 真实标签获取策略：业务API反馈、用户行为信号、LLM交叉验证、人工抽样
- 评估指标：意图准确率、槽位精确率/召回率、业务转化率、置信度校准(ECE)
- 自动校准：高质量样本自动入库、问题样本自动移除、样本库备份与回滚

### 2025-11-26（第一次更新）
- Claude 完成项目全面分析
- 创建 CLAUDE.md 项目文档
- 项目核心功能：意图识别 + 槽位提取 + RAG增强
- 技术栈：Flask + Milvus Lite + Qwen3-32B + bge-m3
- 分析 Embedding 模型(bge-m3)和 Rerank 模型(bge-rerank-v2-m3)的作用

---

## 文件说明快速索引

| 文件 | 行数 | 核心功能 |
|------|------|----------|
| `llm_milvuslite.py` | ~837行 | RAGService类，包含完整的Prompt模板 |
| `flask_api.py` | ~392行 | Flask API，单例模式管理RAG服务 |
| `init_milvus.py` | ~313行 | MilvusIntentionIngestor类 |
| `data.json` | ~1217行 | 85条意图识别训练样本 |
| `config.ini` | 15行 | 所有外部服务配置 |

---

## 自动校准系统设计方案

### 一、系统概述

**意图识别自动校准系统（Intent Calibration System）** 是一个用于动态监测意图识别准确度、自动优化样本库、并生成评估报告的闭环系统。

#### 设计目标
- **动态观测**：实时或定期评估意图识别的准确度
- **自动校准**：根据评估结果自动调整训练样本库
- **报告输出**：生成多维度的准确度评估报告

#### 核心设计原则

| 原则 | 说明 |
|------|------|
| **无需人工标注** | 通过业务闭环和用户行为自动获取真实标签 |
| **非侵入式** | 异步处理，不影响主服务性能 |
| **渐进式校准** | 小步迭代，避免大幅波动 |
| **可回滚** | 每次校准保留快照，支持回退 |
| **多维度评估** | 意图准确率 + 槽位精确率 + 业务转化率 |

---

### 二、系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        意图识别自动校准系统 (Intent Calibration System)        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   数据采集层   │───▶│   评估引擎层   │───▶│   校准执行层   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  反馈存储库   │    │   指标计算器   │    │  样本管理器   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                          │
│         └───────────────────┴───────────────────┘                          │
│                             │                                              │
│                             ▼                                              │
│                    ┌──────────────┐                                        │
│                    │   报告生成器   │                                        │
│                    └──────────────┘                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 三、真实标签获取策略（核心难点）

在没有人工标注的情况下，系统通过以下四种来源自动获取"真实标签"：

#### 1. 业务执行反馈（最可靠）
```
用户查询 → 意图识别 → 调用下游API → API执行成功/失败
                                        ↓
                                 成功 = 意图正确
                                 失败 = 可能意图错误
```

#### 2. 用户行为信号（隐式反馈）
| 行为 | 解读 |
|------|------|
| 用户重新提问(rephrase) | 上次识别可能有误 |
| 用户点击"换一个回答" | 当前结果不满意 |
| 用户完成业务流程 | 识别正确 |
| 会话轮次过多 | 体验不佳 |

#### 3. LLM自校验（Cross-Validation）
```
同一查询用不同温度/Prompt多次推理
结果一致性高 → 置信度高
结果分歧大 → 标记为"待审核"
```

#### 4. 人工抽样审核（定期）
每日/每周抽取低置信度样本进行人工校验，作为评估基准的"金标准"。

---

### 四、数据模型设计

#### 4.1 预测记录模型

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum
from datetime import datetime
import uuid


class FeedbackSource(Enum):
    """反馈来源"""
    BUSINESS_API = "business_api"      # 业务API执行结果
    USER_BEHAVIOR = "user_behavior"    # 用户行为信号
    LLM_CROSS_CHECK = "llm_cross"      # LLM交叉验证
    HUMAN_REVIEW = "human_review"      # 人工审核


class FeedbackSignal(Enum):
    """反馈信号类型"""
    POSITIVE = "positive"              # 正向反馈（识别正确）
    NEGATIVE = "negative"              # 负向反馈（识别错误）
    UNCERTAIN = "uncertain"            # 不确定


@dataclass
class PredictionRecord:
    """单次预测记录"""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # 输入
    query: str = ""

    # 预测输出
    predicted_intent: str = ""
    predicted_slots: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0      # 模型置信度

    # RAG上下文
    retrieved_docs: List[str] = field(default_factory=list)
    rerank_scores: List[float] = field(default_factory=list)

    # 真实标签（后续填充）
    actual_intent: Optional[str] = None
    actual_slots: Optional[Dict[str, Any]] = None

    # 反馈信息
    feedback_source: Optional[FeedbackSource] = None
    feedback_signal: Optional[FeedbackSignal] = None
    feedback_detail: Optional[str] = None
    feedback_timestamp: Optional[datetime] = None

    # 业务结果
    downstream_api_called: Optional[str] = None
    downstream_api_success: Optional[bool] = None
    business_conversion: Optional[bool] = None  # 是否完成业务目标
```

#### 4.2 评估指标模型

```python
@dataclass
class EvaluationMetrics:
    """评估指标"""
    eval_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    eval_timestamp: datetime = field(default_factory=datetime.now)
    eval_period_start: datetime = None
    eval_period_end: datetime = None
    sample_count: int = 0

    # 意图级别指标
    intent_accuracy: float = 0.0                    # 整体准确率
    intent_precision: Dict[str, float] = field(default_factory=dict)  # 各意图精确率
    intent_recall: Dict[str, float] = field(default_factory=dict)     # 各意图召回率
    intent_f1: Dict[str, float] = field(default_factory=dict)         # 各意图F1
    intent_confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # 槽位级别指标
    slot_precision: float = 0.0                     # 槽位精确率
    slot_recall: float = 0.0                        # 槽位召回率
    slot_f1: float = 0.0
    slot_exact_match: float = 0.0                   # 槽位完全匹配率

    # 业务级别指标
    business_success_rate: float = 0.0              # 业务转化率
    avg_session_turns: float = 0.0                  # 平均会话轮次
    rephrase_rate: float = 0.0                      # 用户重述率

    # 置信度校准指标
    calibration_error: float = 0.0                  # ECE (Expected Calibration Error)
    confidence_histogram: Dict[str, int] = field(default_factory=dict)

    # 异常检测
    low_confidence_count: int = 0                   # 低置信度样本数
    ood_detection_rate: float = 0.0                 # OOD检测率
```

---

### 五、核心模块实现

#### 5.1 反馈收集器 (FeedbackCollector)

```python
import json
import logging
from typing import Optional, Dict
from datetime import datetime
import threading
import queue


class FeedbackCollector:
    """
    反馈收集器 - 异步收集各类反馈信号

    设计原则：
    1. 非阻塞：不影响主服务响应
    2. 批量写入：减少IO压力
    3. 多源融合：整合不同来源的反馈
    """

    def __init__(self, storage_path: str = "./calibration_data"):
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)

        # 异步队列
        self._feedback_queue = queue.Queue(maxsize=10000)
        self._prediction_cache: Dict[str, PredictionRecord] = {}
        self._cache_lock = threading.Lock()

        # 启动后台写入线程
        self._start_background_writer()

    def record_prediction(self, record: PredictionRecord) -> str:
        """
        记录一次预测（主服务调用）

        Returns:
            record_id: 用于后续关联反馈
        """
        with self._cache_lock:
            self._prediction_cache[record.record_id] = record

        # 设置过期时间（24小时后如果没有反馈则持久化）
        threading.Timer(
            86400,
            self._expire_record,
            args=[record.record_id]
        ).start()

        return record.record_id

    def collect_business_feedback(
        self,
        record_id: str,
        api_name: str,
        api_success: bool,
        error_msg: Optional[str] = None
    ):
        """
        收集业务API执行反馈

        这是最可靠的反馈来源：
        - API执行成功 → 意图识别大概率正确
        - API执行失败(参数错误) → 槽位提取可能有误
        - API执行失败(无结果) → 意图可能错误
        """
        with self._cache_lock:
            if record_id not in self._prediction_cache:
                self.logger.warning(f"Record {record_id} not found")
                return

            record = self._prediction_cache[record_id]
            record.downstream_api_called = api_name
            record.downstream_api_success = api_success
            record.feedback_source = FeedbackSource.BUSINESS_API
            record.feedback_timestamp = datetime.now()

            # 推断反馈信号
            if api_success:
                record.feedback_signal = FeedbackSignal.POSITIVE
            else:
                if error_msg and "参数" in error_msg:
                    record.feedback_signal = FeedbackSignal.NEGATIVE
                    record.feedback_detail = f"槽位提取可能有误: {error_msg}"
                else:
                    record.feedback_signal = FeedbackSignal.UNCERTAIN
                    record.feedback_detail = error_msg

        self._feedback_queue.put(record_id)

    def collect_user_behavior(
        self,
        record_id: str,
        behavior_type: str,  # "rephrase", "click_retry", "complete_flow", "abandon"
        detail: Optional[Dict] = None
    ):
        """
        收集用户行为信号

        行为信号解读：
        - rephrase: 用户换了一种说法重新提问 → 上次识别可能不准
        - click_retry: 用户点击重试/换一个 → 当前结果不满意
        - complete_flow: 用户完成了整个业务流程 → 识别正确
        - abandon: 用户放弃/离开 → 体验不佳
        """
        with self._cache_lock:
            if record_id not in self._prediction_cache:
                return

            record = self._prediction_cache[record_id]
            record.feedback_source = FeedbackSource.USER_BEHAVIOR
            record.feedback_timestamp = datetime.now()

            if behavior_type == "complete_flow":
                record.feedback_signal = FeedbackSignal.POSITIVE
                record.business_conversion = True
            elif behavior_type in ["rephrase", "click_retry"]:
                record.feedback_signal = FeedbackSignal.NEGATIVE
                record.feedback_detail = f"用户行为: {behavior_type}"
            elif behavior_type == "abandon":
                record.feedback_signal = FeedbackSignal.UNCERTAIN
                record.feedback_detail = "用户放弃"

        self._feedback_queue.put(record_id)

    def collect_llm_cross_check(
        self,
        record_id: str,
        alternative_results: List[Dict],
        consistency_score: float
    ):
        """
        收集LLM交叉验证结果

        同一查询用不同参数多次推理，检查结果一致性
        """
        with self._cache_lock:
            if record_id not in self._prediction_cache:
                return

            record = self._prediction_cache[record_id]
            record.feedback_source = FeedbackSource.LLM_CROSS_CHECK
            record.confidence_score = consistency_score

            if consistency_score >= 0.9:
                record.feedback_signal = FeedbackSignal.POSITIVE
            elif consistency_score >= 0.7:
                record.feedback_signal = FeedbackSignal.UNCERTAIN
            else:
                record.feedback_signal = FeedbackSignal.NEGATIVE
                record.feedback_detail = f"LLM结果不一致: {alternative_results}"

        self._feedback_queue.put(record_id)

    def _start_background_writer(self):
        """启动后台写入线程"""
        def writer_loop():
            batch = []
            while True:
                try:
                    record_id = self._feedback_queue.get(timeout=60)
                    with self._cache_lock:
                        if record_id in self._prediction_cache:
                            batch.append(self._prediction_cache.pop(record_id))

                    if len(batch) >= 100:
                        self._persist_batch(batch)
                        batch = []
                except queue.Empty:
                    if batch:
                        self._persist_batch(batch)
                        batch = []

        thread = threading.Thread(target=writer_loop, daemon=True)
        thread.start()

    def _persist_batch(self, records: List[PredictionRecord]):
        """批量持久化到存储"""
        # 实现存储逻辑（文件/数据库）
        pass

    def _expire_record(self, record_id: str):
        """过期处理：没有收到反馈的记录"""
        with self._cache_lock:
            if record_id in self._prediction_cache:
                record = self._prediction_cache.pop(record_id)
                record.feedback_signal = FeedbackSignal.UNCERTAIN
                record.feedback_detail = "未收到反馈，已过期"
                self._feedback_queue.put_nowait(record_id)
```

#### 5.2 评估引擎 (EvaluationEngine)

```python
import numpy as np
from collections import defaultdict
from typing import List, Tuple


class EvaluationEngine:
    """
    评估引擎 - 计算各维度指标

    评估维度：
    1. 意图分类准确率
    2. 槽位提取准确率
    3. 业务转化指标
    4. 置信度校准
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.intent_list = ["31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "50"]

    def evaluate(
        self,
        records: List[PredictionRecord],
        period_start: datetime = None,
        period_end: datetime = None
    ) -> EvaluationMetrics:
        """执行完整评估"""
        metrics = EvaluationMetrics(
            eval_period_start=period_start,
            eval_period_end=period_end,
            sample_count=len(records)
        )

        valid_records = [r for r in records if r.actual_intent is not None]
        feedback_records = [r for r in records if r.feedback_signal is not None]

        if not valid_records and not feedback_records:
            self.logger.warning("没有有效评估样本")
            return metrics

        # 1. 意图级别评估
        if valid_records:
            self._evaluate_intent(valid_records, metrics)
            self._evaluate_slots(valid_records, metrics)

        # 2. 基于反馈的评估
        if feedback_records:
            self._evaluate_from_feedback(feedback_records, metrics)

        # 3. 业务指标评估
        self._evaluate_business(records, metrics)

        # 4. 置信度校准评估
        self._evaluate_calibration(records, metrics)

        return metrics

    def _evaluate_intent(self, records: List[PredictionRecord], metrics: EvaluationMetrics):
        """意图分类评估"""
        y_true = [r.actual_intent for r in records]
        y_pred = [r.predicted_intent for r in records]

        # 整体准确率
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        metrics.intent_accuracy = correct / len(records)

        # 混淆矩阵
        confusion = defaultdict(lambda: defaultdict(int))
        for t, p in zip(y_true, y_pred):
            confusion[t][p] += 1
        metrics.intent_confusion_matrix = dict(confusion)

        # 各意图的精确率、召回率、F1
        for intent in self.intent_list:
            tp = confusion[intent][intent]
            fp = sum(confusion[other][intent] for other in self.intent_list if other != intent)
            fn = sum(confusion[intent][other] for other in self.intent_list if other != intent)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics.intent_precision[intent] = precision
            metrics.intent_recall[intent] = recall
            metrics.intent_f1[intent] = f1

    def _evaluate_slots(self, records: List[PredictionRecord], metrics: EvaluationMetrics):
        """槽位提取评估"""
        total_tp, total_fp, total_fn = 0, 0, 0
        exact_match_count = 0

        for record in records:
            if record.actual_slots is None:
                continue

            pred_slots = set(record.predicted_slots.items())
            actual_slots = set(record.actual_slots.items())

            tp = len(pred_slots & actual_slots)
            fp = len(pred_slots - actual_slots)
            fn = len(actual_slots - pred_slots)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            if pred_slots == actual_slots:
                exact_match_count += 1

        metrics.slot_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        metrics.slot_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        metrics.slot_f1 = (2 * metrics.slot_precision * metrics.slot_recall /
                          (metrics.slot_precision + metrics.slot_recall)
                          if (metrics.slot_precision + metrics.slot_recall) > 0 else 0)
        metrics.slot_exact_match = exact_match_count / len(records) if records else 0

    def _evaluate_from_feedback(self, records: List[PredictionRecord], metrics: EvaluationMetrics):
        """基于反馈信号的评估（无真实标签时的替代方案）"""
        positive_count = sum(1 for r in records if r.feedback_signal == FeedbackSignal.POSITIVE)
        negative_count = sum(1 for r in records if r.feedback_signal == FeedbackSignal.NEGATIVE)

        total = positive_count + negative_count
        if total > 0:
            feedback_accuracy = positive_count / total
            if metrics.intent_accuracy == 0:
                metrics.intent_accuracy = feedback_accuracy

    def _evaluate_business(self, records: List[PredictionRecord], metrics: EvaluationMetrics):
        """业务指标评估"""
        conversion_records = [r for r in records if r.business_conversion is not None]
        if conversion_records:
            success_count = sum(1 for r in conversion_records if r.business_conversion)
            metrics.business_success_rate = success_count / len(conversion_records)

        rephrase_records = [r for r in records
                          if r.feedback_source == FeedbackSource.USER_BEHAVIOR
                          and r.feedback_detail and "rephrase" in r.feedback_detail]
        if records:
            metrics.rephrase_rate = len(rephrase_records) / len(records)

    def _evaluate_calibration(self, records: List[PredictionRecord], metrics: EvaluationMetrics):
        """
        置信度校准评估

        ECE (Expected Calibration Error): 置信度与实际准确率的差距
        理想情况：置信度90%的样本，准确率应该接近90%
        """
        bins = defaultdict(list)
        for record in records:
            if record.confidence_score > 0:
                bin_idx = int(record.confidence_score * 10)
                is_correct = (record.feedback_signal == FeedbackSignal.POSITIVE or
                             (record.actual_intent and record.actual_intent == record.predicted_intent))
                bins[bin_idx].append((record.confidence_score, is_correct))

        total_samples = sum(len(b) for b in bins.values())
        ece = 0
        for bin_idx, samples in bins.items():
            if samples:
                avg_confidence = np.mean([s[0] for s in samples])
                avg_accuracy = np.mean([s[1] for s in samples])
                ece += len(samples) / total_samples * abs(avg_confidence - avg_accuracy)

        metrics.calibration_error = ece
        metrics.low_confidence_count = sum(1 for r in records if r.confidence_score < 0.7)
```

#### 5.3 自动校准器 (AutoCalibrator)

```python
import shutil
from pathlib import Path


class AutoCalibrator:
    """
    自动校准器 - 根据评估结果调整系统

    校准策略：
    1. 样本库动态调整：添加高质量样本、移除噪声样本
    2. 阈值调整：调整置信度阈值
    3. 触发告警：准确率下降时告警
    """

    def __init__(
        self,
        data_json_path: str = "data.json",
        backup_dir: str = "./calibration_backups",
        min_accuracy_threshold: float = 0.85,
        max_sample_size: int = 500
    ):
        self.data_json_path = data_json_path
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.min_accuracy_threshold = min_accuracy_threshold
        self.max_sample_size = max_sample_size
        self.logger = logging.getLogger(__name__)

    def calibrate(
        self,
        metrics: EvaluationMetrics,
        feedback_records: List[PredictionRecord]
    ) -> Dict[str, Any]:
        """执行校准，返回校准报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "actions_taken": [],
            "alerts": [],
            "recommendations": []
        }

        # 1. 检查是否需要告警
        self._check_alerts(metrics, report)

        # 2. 识别高质量新样本
        new_samples = self._identify_quality_samples(feedback_records)

        # 3. 识别问题样本
        problem_samples = self._identify_problem_samples(feedback_records, metrics)

        # 4. 执行样本库更新
        if new_samples or problem_samples:
            self._update_sample_library(new_samples, problem_samples, report)

        # 5. 生成优化建议
        self._generate_recommendations(metrics, report)

        return report

    def _check_alerts(self, metrics: EvaluationMetrics, report: Dict):
        """检查是否需要告警"""
        if metrics.intent_accuracy < self.min_accuracy_threshold:
            report["alerts"].append({
                "level": "critical",
                "type": "accuracy_drop",
                "message": f"意图准确率降至 {metrics.intent_accuracy:.2%}，低于阈值 {self.min_accuracy_threshold:.2%}",
                "metric_value": metrics.intent_accuracy
            })

        for intent, f1 in metrics.intent_f1.items():
            if f1 < 0.7:
                report["alerts"].append({
                    "level": "warning",
                    "type": "intent_performance",
                    "message": f"意图 {intent} 的F1分数较低: {f1:.2%}",
                    "intent": intent,
                    "metric_value": f1
                })

        if metrics.calibration_error > 0.15:
            report["alerts"].append({
                "level": "warning",
                "type": "calibration",
                "message": f"置信度校准误差过大: {metrics.calibration_error:.2%}",
                "metric_value": metrics.calibration_error
            })

    def _identify_quality_samples(self, records: List[PredictionRecord]) -> List[Dict]:
        """
        识别高质量样本（可加入训练集）

        标准：
        1. 有正向反馈
        2. 置信度高
        3. 业务执行成功
        """
        quality_samples = []

        for record in records:
            score = 0

            if record.feedback_signal == FeedbackSignal.POSITIVE:
                score += 2
            if record.downstream_api_success:
                score += 2
            if record.confidence_score >= 0.9:
                score += 1
            if record.feedback_source == FeedbackSource.BUSINESS_API:
                score += 1

            if score >= 4:
                quality_samples.append({
                    "input": record.query,
                    "output": {
                        "intent": record.predicted_intent,
                        "slots": record.predicted_slots,
                        "query": record.query
                    },
                    "quality_score": score,
                    "source": "auto_calibration"
                })

        return quality_samples

    def _identify_problem_samples(
        self,
        records: List[PredictionRecord],
        metrics: EvaluationMetrics
    ) -> List[str]:
        """识别问题样本（需要从训练集移除或修正）"""
        problem_queries = []

        for record in records:
            if record.feedback_signal == FeedbackSignal.NEGATIVE:
                problem_queries.append(record.query)

        return problem_queries

    def _update_sample_library(
        self,
        new_samples: List[Dict],
        problem_queries: List[str],
        report: Dict
    ):
        """更新样本库"""
        # 1. 备份当前样本库
        backup_path = self.backup_dir / f"data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy(self.data_json_path, backup_path)
        report["actions_taken"].append(f"已备份样本库到 {backup_path}")

        # 2. 加载当前样本
        with open(self.data_json_path, 'r', encoding='utf-8') as f:
            current_samples = json.load(f)

        original_count = len(current_samples)

        # 3. 移除问题样本
        if problem_queries:
            current_samples = [
                s for s in current_samples
                if s.get("input") not in problem_queries
            ]
            removed_count = original_count - len(current_samples)
            if removed_count > 0:
                report["actions_taken"].append(f"移除 {removed_count} 个问题样本")

        # 4. 添加新样本（去重）
        existing_queries = {s.get("input") for s in current_samples}
        added_count = 0
        for sample in new_samples:
            if sample["input"] not in existing_queries:
                current_samples.append({
                    "input": sample["input"],
                    "output": sample["output"]
                })
                existing_queries.add(sample["input"])
                added_count += 1

                if len(current_samples) >= self.max_sample_size:
                    break

        if added_count > 0:
            report["actions_taken"].append(f"添加 {added_count} 个高质量样本")

        # 5. 保存更新后的样本库
        with open(self.data_json_path, 'w', encoding='utf-8') as f:
            json.dump(current_samples, f, ensure_ascii=False, indent=2)

        report["actions_taken"].append(f"样本库已更新: {original_count} → {len(current_samples)}")

    def _generate_recommendations(self, metrics: EvaluationMetrics, report: Dict):
        """生成优化建议"""
        for intent, f1 in metrics.intent_f1.items():
            if f1 < 0.8:
                precision = metrics.intent_precision.get(intent, 0)
                recall = metrics.intent_recall.get(intent, 0)

                if precision < recall:
                    report["recommendations"].append({
                        "intent": intent,
                        "issue": "精确率低",
                        "suggestion": f"意图 {intent} 容易被误判，建议添加更多边界样本或调整Prompt中的区分规则"
                    })
                else:
                    report["recommendations"].append({
                        "intent": intent,
                        "issue": "召回率低",
                        "suggestion": f"意图 {intent} 容易漏判，建议添加更多该意图的多样化表述样本"
                    })

        if metrics.ood_detection_rate < 0.9:
            report["recommendations"].append({
                "intent": "40",
                "issue": "OOD检测率低",
                "suggestion": "建议添加更多OOD样本，或在Prompt中强调OOD判断规则"
            })
```

#### 5.4 报告生成器 (ReportGenerator)

```python
class ReportGenerator:
    """报告生成器 - 生成多维度评估报告"""

    def generate_report(
        self,
        metrics: EvaluationMetrics,
        calibration_report: Dict[str, Any],
        output_format: str = "markdown"
    ) -> str:
        """生成评估报告"""
        if output_format == "markdown":
            return self._generate_markdown_report(metrics, calibration_report)
        elif output_format == "json":
            return self._generate_json_report(metrics, calibration_report)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def _generate_markdown_report(
        self,
        metrics: EvaluationMetrics,
        calibration_report: Dict[str, Any]
    ) -> str:
        """生成Markdown格式报告"""
        report = f"""
# 意图识别评估报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**评估周期**: {metrics.eval_period_start} ~ {metrics.eval_period_end}
**样本数量**: {metrics.sample_count}

---

## 一、核心指标概览

| 指标 | 数值 | 状态 |
|------|------|------|
| 意图准确率 | {metrics.intent_accuracy:.2%} | {self._status_icon(metrics.intent_accuracy, 0.85)} |
| 槽位精确率 | {metrics.slot_precision:.2%} | {self._status_icon(metrics.slot_precision, 0.80)} |
| 槽位召回率 | {metrics.slot_recall:.2%} | {self._status_icon(metrics.slot_recall, 0.80)} |
| 槽位完全匹配 | {metrics.slot_exact_match:.2%} | {self._status_icon(metrics.slot_exact_match, 0.70)} |
| 业务转化率 | {metrics.business_success_rate:.2%} | {self._status_icon(metrics.business_success_rate, 0.80)} |
| 置信度校准误差 | {metrics.calibration_error:.2%} | {self._status_icon(1 - metrics.calibration_error, 0.85)} |

---

## 二、各意图表现详情

| 意图 | 精确率 | 召回率 | F1 | 状态 |
|------|--------|--------|-----|------|
"""
        for intent in sorted(metrics.intent_f1.keys()):
            p = metrics.intent_precision.get(intent, 0)
            r = metrics.intent_recall.get(intent, 0)
            f1 = metrics.intent_f1.get(intent, 0)
            report += f"| {intent} | {p:.2%} | {r:.2%} | {f1:.2%} | {self._status_icon(f1, 0.80)} |\n"

        # 告警部分
        if calibration_report.get("alerts"):
            report += "\n---\n\n## 三、告警信息\n\n"
            for alert in calibration_report["alerts"]:
                icon = "🔴" if alert["level"] == "critical" else "🟡"
                report += f"{icon} **{alert['type']}**: {alert['message']}\n\n"

        # 校准动作
        if calibration_report.get("actions_taken"):
            report += "\n---\n\n## 四、自动校准动作\n\n"
            for action in calibration_report["actions_taken"]:
                report += f"- {action}\n"

        # 优化建议
        if calibration_report.get("recommendations"):
            report += "\n---\n\n## 五、优化建议\n\n"
            for rec in calibration_report["recommendations"]:
                report += f"### 意图 {rec['intent']}\n"
                report += f"- **问题**: {rec['issue']}\n"
                report += f"- **建议**: {rec['suggestion']}\n\n"

        return report

    def _status_icon(self, value: float, threshold: float) -> str:
        """根据阈值返回状态图标"""
        if value >= threshold:
            return "✅"
        elif value >= threshold * 0.9:
            return "⚠️"
        else:
            return "❌"

    def _generate_json_report(
        self,
        metrics: EvaluationMetrics,
        calibration_report: Dict[str, Any]
    ) -> str:
        """生成JSON格式报告"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "metrics": {
                "intent_accuracy": metrics.intent_accuracy,
                "slot_precision": metrics.slot_precision,
                "slot_recall": metrics.slot_recall,
                "slot_f1": metrics.slot_f1,
                "business_success_rate": metrics.business_success_rate,
                "calibration_error": metrics.calibration_error,
                "intent_details": {
                    intent: {
                        "precision": metrics.intent_precision.get(intent, 0),
                        "recall": metrics.intent_recall.get(intent, 0),
                        "f1": metrics.intent_f1.get(intent, 0)
                    }
                    for intent in metrics.intent_f1.keys()
                }
            },
            "calibration": calibration_report
        }
        return json.dumps(report, ensure_ascii=False, indent=2)
```

---

### 六、系统集成

#### 6.1 Flask API 集成

```python
# 在 flask_api.py 中集成

from calibration import FeedbackCollector, EvaluationEngine, AutoCalibrator, ReportGenerator

# 初始化校准系统
feedback_collector = FeedbackCollector()
evaluation_engine = EvaluationEngine()
auto_calibrator = AutoCalibrator()
report_generator = ReportGenerator()


@app.route('/rag_query', methods=['POST'])
def rag_query():
    # ... 原有逻辑 ...

    # 记录预测（新增）
    record = PredictionRecord(
        query=query,
        predicted_intent=response_dict.get("intent"),
        predicted_slots=response_dict.get("slots", {}),
        confidence_score=calculate_confidence(response)
    )
    record_id = feedback_collector.record_prediction(record)

    # 在响应中返回record_id，供前端回传反馈
    response["_record_id"] = record_id

    return response


@app.route('/feedback', methods=['POST'])
def collect_feedback():
    """接收反馈的接口"""
    data = request.get_json()
    record_id = data.get("record_id")
    feedback_type = data.get("type")

    if feedback_type == "business_result":
        feedback_collector.collect_business_feedback(
            record_id=record_id,
            api_name=data.get("api_name"),
            api_success=data.get("success"),
            error_msg=data.get("error")
        )
    elif feedback_type == "user_behavior":
        feedback_collector.collect_user_behavior(
            record_id=record_id,
            behavior_type=data.get("behavior"),
            detail=data.get("detail")
        )

    return {"status": "ok"}


@app.route('/calibration/report', methods=['GET'])
def get_calibration_report():
    """获取校准报告"""
    records = load_feedback_records()
    metrics = evaluation_engine.evaluate(records)
    calibration_result = auto_calibrator.calibrate(metrics, records)
    report = report_generator.generate_report(
        metrics,
        calibration_result,
        output_format=request.args.get("format", "markdown")
    )
    return {"report": report}
```

#### 6.2 定时任务配置

```python
# calibration_scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler


def setup_calibration_scheduler():
    """设置定时校准任务"""
    scheduler = BackgroundScheduler()

    # 每天凌晨2点执行评估和校准
    scheduler.add_job(
        run_daily_calibration,
        'cron',
        hour=2,
        minute=0
    )

    # 每小时检查告警
    scheduler.add_job(
        check_alerts,
        'interval',
        hours=1
    )

    scheduler.start()


def run_daily_calibration():
    """每日校准任务"""
    # 1. 加载过去24小时的记录
    records = load_recent_records(hours=24)

    # 2. 执行评估
    metrics = evaluation_engine.evaluate(records)

    # 3. 执行校准
    calibration_result = auto_calibrator.calibrate(metrics, records)

    # 4. 生成并保存报告
    report = report_generator.generate_report(metrics, calibration_result)
    save_report(report)

    # 5. 如果有严重告警，发送通知
    critical_alerts = [a for a in calibration_result["alerts"] if a["level"] == "critical"]
    if critical_alerts:
        send_alert_notification(critical_alerts)

    # 6. 如果样本库有更新，触发Milvus重建
    if any("样本库已更新" in action for action in calibration_result["actions_taken"]):
        trigger_milvus_rebuild()
```

---

### 七、新增文件结构

```
intention_correction/
├── calibration/                    # 校准系统模块（新增）
│   ├── __init__.py
│   ├── models.py                   # 数据模型定义
│   ├── feedback_collector.py       # 反馈收集器
│   ├── evaluation_engine.py        # 评估引擎
│   ├── auto_calibrator.py          # 自动校准器
│   ├── report_generator.py         # 报告生成器
│   └── scheduler.py                # 定时任务
├── calibration_data/               # 校准数据存储（新增）
│   └── feedback_records/           # 反馈记录
├── calibration_backups/            # 样本库备份（新增）
├── calibration_reports/            # 评估报告（新增）
└── ... (原有文件)
```

---

### 八、系统优势总结

| 特性 | 说明 |
|------|------|
| **无需人工标注** | 通过业务闭环和用户行为自动获取真实标签 |
| **非侵入式** | 异步处理，不影响主服务性能 |
| **多维度评估** | 意图准确率 + 槽位准确率 + 业务指标 + 置信度校准 |
| **自动校准** | 自动添加高质量样本、移除噪声样本 |
| **可回滚** | 每次校准前备份，支持快速回退 |
| **告警机制** | 准确率下降自动告警 |
| **报告可视化** | 支持Markdown/JSON多种格式 |

---

### 九、报告示例

```markdown
# 意图识别评估报告

**生成时间**: 2025-11-26 02:00:00
**评估周期**: 2025-11-25 02:00:00 ~ 2025-11-26 02:00:00
**样本数量**: 1250

---

## 一、核心指标概览

| 指标 | 数值 | 状态 |
|------|------|------|
| 意图准确率 | 91.2% | ✅ |
| 槽位精确率 | 87.5% | ✅ |
| 槽位召回率 | 82.3% | ✅ |
| 槽位完全匹配 | 76.8% | ✅ |
| 业务转化率 | 85.2% | ✅ |
| 置信度校准误差 | 8.3% | ✅ |

---

## 二、各意图表现详情

| 意图 | 精确率 | 召回率 | F1 | 状态 |
|------|--------|--------|-----|------|
| 31 | 94.2% | 92.1% | 93.1% | ✅ |
| 32 | 89.5% | 87.3% | 88.4% | ✅ |
| 33 | 85.2% | 78.6% | 81.8% | ✅ |
| 34 | 91.0% | 88.5% | 89.7% | ✅ |
| 37 | 72.3% | 68.9% | 70.5% | ⚠️ |
| 40 | 95.8% | 93.2% | 94.5% | ✅ |
| 50 | 68.5% | 71.2% | 69.8% | ⚠️ |

---

## 三、告警信息

🟡 **intent_performance**: 意图 37 的F1分数较低: 70.5%

🟡 **intent_performance**: 意图 50 的F1分数较低: 69.8%

---

## 四、自动校准动作

- 已备份样本库到 calibration_backups/data_backup_20251126_020000.json
- 移除 3 个问题样本
- 添加 12 个高质量样本
- 样本库已更新: 85 → 94

---

## 五、优化建议

### 意图 37
- **问题**: 精确率低
- **建议**: 意图 37 容易被误判，建议添加更多边界样本或调整Prompt中的区分规则

### 意图 50
- **问题**: 召回率低
- **建议**: 意图 50 容易漏判，建议添加更多该意图的多样化表述样本
```
