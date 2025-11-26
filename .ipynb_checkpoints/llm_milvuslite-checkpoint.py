"""
RAG服务 - 从Milvus Lite召回文档并进行重排序

安装依赖:
pip install requests pymilvus milvus-lite

主要功能:
- 从Milvus Lite数据库召回相关文档
- Rerank模型对文档进行重排序
"""

import requests
import json
import configparser
import os
import logging
from typing import Optional, Dict, Any, List
import datetime
import threading

try:
    from pymilvus import connections, Collection, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    print("警告: pymilvus未安装，向量召回功能将不可用。请运行: pip install pymilvus milvus-lite")

class RAGService:
    def __init__(self, config_path: str = "config.ini"):
        """
        初始化RAG服务
        
        Args:
            config_path: 配置文件路径
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='utf-8')
        
        # 读取配置
        self.model_api_url = self.config.get('DEFAULT', 'MODEL_API_URL')
        self.timeout = self.config.getint('DEFAULT', 'TIMEOUT')
        self.default_response = self.config.get('DEFAULT', 'DEFAULT_RESPONSE')
        self.log_dir = self.config.get('DEFAULT', 'LOG_DIR')
        self.log_file = self.config.get('DEFAULT', 'LOG_FILE')
        self.model_name = self.config.get('DEFAULT', 'MODEL_NAME')
        
        # Rerank配置
        self.rerank_api_url = self.config.get('DEFAULT', 'RERANK_API_URL')
        self.rerank_model_name = self.config.get('DEFAULT', 'RERANK_MODEL_NAME')

        # 设置日志
        self._setup_logging()

        # Milvus Lite配置
        if MILVUS_AVAILABLE:
            try:
                self.milvus_db_path = self.config.get('DEFAULT', 'MILVUS_DB_PATH', fallback='./milvus.db')
                self.milvus_collection_name = self.config.get('DEFAULT', 'MILVUS_COLLECTION_NAME')
                self.embedding_model_url = self.config.get('DEFAULT', 'EMBEDDING_API_URL')
                self.embedding_model_name = self.config.get('DEFAULT', 'EMBEDDING_MODEL_NAME')

                # 初始化Milvus Lite连接
                self._init_milvus_lite_connection()
            except configparser.NoOptionError:
                self.logger.warning("Milvus Lite配置未找到，向量召回功能将不可用")
                self.milvus_collection = None
        else:
            self.logger.warning("pymilvus或milvus-lite未安装，向量召回功能将不可用")
            self.milvus_collection = None
        
        # 设置回复存储目录
        self.responses_dir = "responses"
        if not os.path.exists(self.responses_dir):
            os.makedirs(self.responses_dir)

    def _setup_logging(self):
        """设置日志配置"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        log_path = os.path.join(self.log_dir, self.log_file)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _init_milvus_lite_connection(self):
        """初始化Milvus Lite连接"""
        try:
            self.logger.info(f"正在连接Milvus Lite数据库: {self.milvus_db_path}")

            # 连接到 Milvus Lite 数据库
            connections.connect(uri=self.milvus_db_path)

            # 检查集合是否存在
            if not utility.has_collection(self.milvus_collection_name):
                self.logger.error(f"集合 '{self.milvus_collection_name}' 不存在")
                self.milvus_collection = None
                return

            self.milvus_collection = Collection(self.milvus_collection_name)

            # 检查集合中是否有数据
            self.logger.info(f"集合包含 {self.milvus_collection.num_entities} 个实体")

            # 加载集合
            self.logger.info(f"成功连接到Milvus Lite集合: {self.milvus_collection_name}")

        except Exception as e:
            self.logger.error(f"连接Milvus Lite失败: {e}")
            self.milvus_collection = None

    def test_milvus_lite_connection(self) -> bool:
        """测试Milvus Lite连接是否正常"""
        try:
            if not self.milvus_collection:
                self.logger.error("Milvus Lite集合未初始化")
                return False

            # 测试基本连接
            if not connections.has_connection("default"):
                self.logger.error("Milvus Lite连接不存在")
                return False

            # 检查集合是否已加载
            num_entities = self.milvus_collection.num_entities
            self.logger.info(f"集合包含 {num_entities} 个实体")

            # 检查向量字段
            schema = self.milvus_collection.schema
            has_embedding_field = any(field.name == "embedding" for field in schema.fields)
            if not has_embedding_field:
                self.logger.error("集合中缺少 'embedding' 字段")
                return False

            self.logger.info("Milvus Lite连接测试成功")
            return True

        except Exception as e:
            self.logger.error(f"Milvus Lite连接测试失败: {e}")
            return False

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """获取文本的向量嵌入"""
        try:
            headers = {'Content-Type': 'application/json'}
            payload = {
                "model": self.embedding_model_name,
                "input": text
            }

            response = requests.post(
                self.embedding_model_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'data' in result and len(result['data']) > 0:
                    return result['data'][0]['embedding']

            self.logger.error(f"获取嵌入向量失败: {response.text}")
            return None
        except Exception as e:
            self.logger.error(f"获取嵌入向量异常: {e}")
            return None

    def _search_similar_documents(self, query: str, top_k: int = 10) -> List[str]:
        """
        从Milvus Lite搜索相似文档

        Args:
            query: 查询文本
            top_k: 返回文档数量

        Returns:
            相似文档列表
        """
        if not self.milvus_collection:
            self.logger.warning("Milvus Lite集合未初始化，无法进行向量搜索")
            return []

        try:
            # 获取查询向量
            self.logger.info(f"正在获取查询向量...")
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                self.logger.error("无法获取查询向量")
                return []

            # 检查集合是否为空
            if self.milvus_collection.num_entities == 0:
                self.logger.warning("集合为空，无法进行向量搜索")
                return []

            # 获取集合信息用于调试
            self.logger.info(f"集合包含 {self.milvus_collection.num_entities} 个实体")

            # 执行向量搜索
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }

            self.logger.info(f"开始向量搜索，返回 {top_k} 个结果...")
            results = self.milvus_collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["input"]  # 假设文档内容字段名为content
            )

            # 提取文档内容
            documents = []
            for hits in results:
                for hit in hits:
                    if hasattr(hit, 'entity') and hasattr(hit.entity, 'get'):
                        content = hit.entity.get('input', '')
                        if content:
                            documents.append(content)
                        else:
                            self.logger.warning("命中结果中没有content字段")
                    elif hasattr(hit, 'entity') and 'input' in hit.entity:
                        # 兼容不同版本的pymilvus
                        content = hit.entity['input']
                        if content:
                            documents.append(content)

            self.logger.info(f"向量搜索返回 {len(documents)} 个文档")
            return documents

        except Exception as e:
            self.logger.error(f"向量搜索失败: {e}")
            self.logger.error(f"错误详情: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """
        调用大模型API
        
        Args:
            prompt: 提示词
            temperature: 温度参数
            
        Returns:
            模型响应文本
        """
        headers = {
            'Content-Type': 'application/json'
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature
        }
        
        try:
            self.logger.info(f"调用大模型API: {self.model_api_url}")
            self.logger.info(f"请求内容: {prompt[:200]}...")  # 只记录前200个字符
            
            response = requests.post(
                self.model_api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    self.logger.info("大模型调用成功")
                    return content
                else:
                    self.logger.error(f"API响应格式异常: {result}")
                    return self.default_response
            else:
                self.logger.error(f"API调用失败，状态码: {response.status_code}, 响应: {response.text}")
                return self.default_response
                
        except requests.exceptions.Timeout:
            self.logger.error("API调用超时")
            return self.default_response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API调用异常: {e}")
            return self.default_response
        except Exception as e:
            self.logger.error(f"调用大模型时发生未知错误: {e}")
            return self.default_response
    
    def call_rerank(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """
        调用rerank模型对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            
        Returns:
            重排序后的文档列表，包含相关性分数
        """
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = {
            "query": query,
            "documents": documents,
            "model": self.rerank_model_name
        }
        
        try:
            self.logger.info(f"调用rerank API: {self.rerank_api_url}")
            self.logger.info(f"查询内容: {query[:100]}...")
            
            response = requests.post(
                self.rerank_api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'results' in result:
                    reranked_results = result['results']
                    self.logger.info(f"Rerank调用成功，返回 {len(reranked_results)} 个结果")
                    
                    # 按照score降序排序
                    reranked_results.sort(key=lambda x: x.get('score', 0), reverse=True)
                    return reranked_results
                else:
                    self.logger.error(f"Rerank API响应格式异常: {result}")
                    return []
            else:
                self.logger.error(f"Rerank API调用失败，状态码: {response.status_code}, 响应: {response.text}")
                return []
                
        except requests.exceptions.Timeout:
            self.logger.error("Rerank API调用超时")
            return []
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Rerank API调用异常: {e}")
            return []
        except Exception as e:
            self.logger.error(f"调用rerank模型时发生未知错误: {e}")
            return []
    
    def _save_response(self, query: str, response: str, filename: str = None):
        """
        保存单次查询的响应结果
        
        Args:
            query: 查询内容
            response: 响应内容
            filename: 指定保存文件名，默认为当前时间戳
        """
        try:
            if not filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rag_response_{timestamp}.json"
            filepath = os.path.join(self.responses_dir, filename)
            
            result = {
                "query": query,
                "response": response,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self.logger.info(f"响应已保存到: {filepath}")
        except Exception as e:
            self.logger.error(f"保存响应失败: {e}")

    def _save_all_responses(self, all_responses: List[Dict[str, Any]], filename: str = None):
        """
        保存所有并发查询的响应结果
        
        Args:
            all_responses: 所有响应列表
            filename: 指定保存文件名，默认为当前时间戳
        """
        try:
            if not filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"all_rag_responses_{timestamp}.json"
            filepath = os.path.join(self.responses_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(all_responses, f, ensure_ascii=False, indent=2)
            self.logger.info(f"所有响应已保存到: {filepath}")
        except Exception as e:
            self.logger.error(f"保存所有响应失败: {e}")

    def rag_query(self, query: str, top_k: int = 10, temperature: float = 0.7) -> str:
        """
        执行RAG查询：召回 -> 重排序 -> 生成回答
        
        Args:
            query: 用户查询
            top_k: 召回文档数量
            temperature: 大模型温度参数
            
        Returns:
            生成的回答
        """
        self.logger.info(f"开始RAG查询: {query}")
        
        # 1. 从Milvus Lite召回文档
        retrieved_docs = self._search_similar_documents(query, top_k=top_k)
        
        if not retrieved_docs:
            self.logger.warning("未召回到相关文档，直接生成回答")
            # 使用简单提示词生成回答
            simple_prompt = f"请回答：{query}"
            return self.call_llm(simple_prompt, temperature)
        
        self.logger.info(f"召回到 {len(retrieved_docs)} 个文档")
        
        # 2. 使用Rerank对召回文档进行重排序
        rerank_results = self.call_rerank(query, retrieved_docs)
        
        # 3. 选择最相关的文档
        if rerank_results:
            # 取前5个最相关的文档
            top_docs = [result['document'] for result in rerank_results[:5]]
            self.logger.info(f"重排序后选择 {len(top_docs)} 个最相关文档")
        else:
            # 如果重排序失败，使用原始召回结果
            top_docs = retrieved_docs[:5]
            self.logger.info(f"重排序失败，使用前5个召回文档")
        
        # 4. 构建RAG提示词
        context = "\n\n".join([f"参考信息 {i+1}: {doc}" for i, doc in enumerate(top_docs)])
        single_query = """        【指令】
  作为数据资产助手的意图识别专家，请分析用户查询，识别意图并提取槽位。
  若用户问题与意图清单都不相符，则返回 {intent: "40", slots: {}, query: "用户查询的原始问题"}。
  若用户询问助手本身的能力范围、功能清单、使用方法、身份定义等相关问题，则返回 {intent: "39", slots: {}, query: "用户查询的原始问题"}。
  若用户询问平台操作/帮助/名词解释/资产推荐，则返回 {intent: "38", slots: {}, query: "用户查询的原始问题"}。
  无需对识别结果进行解释或说明。

  【输出要求】
  1. 格式：严格遵循JSON结构 {"intent": "意图编码", "slots": {"槽位类型": "槽位值"}, "query": "用户查询的原始问题"}
  2. 约束：只输出JSON，不要包含任何解释、思考过程或其他文字
  3. 规则：无法提取的槽位直接省略，不要返回null或空字符串

  ═══════════════════════════════════════
  【意图清单】
  ═══════════════════════════════════════

  31 - 资产基础检索
    功能：基于名称/字段/域/类型/租户/排行/核心数据项/归属数据域/归属数据层/资产开放范围等条件查找资产
    核心槽位：AssetName, AssetType, BusinessDomain, FilterCondition, AssetRanking, OwnerTenant, FieldName, CoreDataItem, DataDomain, DataLayer, AssetOpenScope

  32 - 资产元数据查询
    功能：查询资产的业务口径/技术口径/负责人/数据存储周期/数据域/数据层/资产开放范围等元数据
    核心槽位：AssetName, MetadataItem（必填）, AssetType, BusinessDomain, FilterCondition, FieldName, AssertAdmin, AssertPublisher
    注意：MetadataItem支持同义词（"业务解释"→"业务口径"、"是干什么的"→"简介/用途"）

  33 - 资产质量与价值查询
    功能：查询资产的价值评分/星级/质量稽核
    核心槽位：AssetName, FilterCondition

  34 - 资产血缘关系查询
    功能：查询资产的上游依赖/下游应用/血缘图/依赖数量
    核心槽位：AssetName（必填）, LineageDirection, MetadataItem

  35 - 资产使用与工单查询
    功能：查询订阅/收藏/工单进度/审批/账期/API服务
    核心槽位：UserStatus, MetadataItem, AssetName, AssetType, FilterCondition, OrgName/UserName

  36 - 场景与标签推荐
    功能：基于业务专区/场景/业务概念推荐资产
    核心槽位：BusinessZone, CoreDataItem, AssetType

  37 - 资产复合对比与筛选
    功能：对比两个资产或进行多条件筛选
    核心槽位：AssetName（多个）, MetadataItem, AssetType, BusinessDomain, FilterCondition

  38 - 平台规则与帮助
    功能：询问平台操作/帮助/名词解释/资产推荐
    核心槽位：无

  39 - 助手能力与帮助
  功能：询问助手自身的能力范围、功能清单、使用方法、身份定义等相关问题
  核心槽位：无
  注意：仅针对“助手本身”的咨询，不包含平台规则、业务名词等非助手相关问题

  40 - OOD兜底
    功能：当用户查询与意图清单都不相符时，返回OOD兜底回答
    核心槽位：无

  ═══════════════════════════════════════
  【槽位定义】
  ═══════════════════════════════════════

  AssetName：资产完整名称（如"[在线公司]终端激活信息(日)"）
  MetadataItem：元数据项（业务口径/技术口径/负责人/血缘图/依赖数量/审批/账期/API服务）
  FieldName：字段名称（user_id/order_id）
  CoreDataItem：业务概念（5G登网/数据资产/移网用户是否活跃, 宽带用户是否活跃, 宽带下是否有异网号码等具体描述）
  BusinessDomain：业务域（M域/O域/B域/API来源/数据库同步）
  AssetType：资产类型（标签/数据表/模型资产/指标/API服务）
  BusinessZone：业务专区（公众智慧运营/一线赋能专区）
  FilterCondition：筛选条件（五星/高价值/最近一周更新的）
  LineageDirection：血缘方向（上游/下游/血缘图）
  OwnerTenant：归属租户（总部/分公司A/数据部）
  AssetRanking：综合排行（最新/本周上新/热门/订阅最多）
  AssertAdmin：资产管理员、负责人（张三）
  AssertPublisher：资产发布人（李四）
  DataDomain：归属数据域（固网视图/客户视图/公众客户信息/营销活动）
  DataLayer：归属数据层（ESD/DM/SRC）
  AssetOpenScope：资产开放范围（公共/私有/保护）

  ═══════════════════════════════════════
  【识别规则】
  ═══════════════════════════════════════

  1. AssetName必须识别完整名称（包括前缀[]和周期()），不要只识别部分
  2. MetadataItem需要识别同义词并规范化
  3. 多个同类型槽位使用数组：{"AssetName": ["资产1", "资产2"]}
  4. 无法提取的槽位直接省略，不返回null
  5. intent必须是31-38之一，无法识别时返回"38"
  6. slots是对象格式{"SlotType": "value"}，不是数组
  7. 输出纯JSON，不包含任何解释或说明
  8. 归属数据层识别为ESD、DM、SRC等
  9. query是用户查询的原始问题

  ═══════════════════════════════════════
  【输出格式示例】
  ═══════════════════════════════════════

  {"intent": "31", "slots": {"BusinessDomain": "M域", "AssetType": "标签"}, "query":"M域的所有标签资产"}
  {"intent": "32", "slots": {"AssetName": "宽带提质速率(月)", "MetadataItem": "业务口径"}, "query": "宽带提质速率(月)的业务口径是什么？"}
  {"intent": "34", "slots": {"AssetName": "HR系统", "LineageDirection": "上游"}, "query": "HR系统上游依赖有哪些？"}
  {"intent": "38", "slots": {}, "query": "平台有什么功能？"}
  {"intent": "38", "slots": {}, "query": "业务口径和技术口径的区别"}
  {"intent": "40", "slots": {}, "query": "今天天气怎么样？"}
"""
        rag_prompt = single_query + f"""
        请根据以下参考信息回答用户的问题：

        参考信息：
        {context}

        用户问题：{query}

        请基于参考信息给出准确、简洁的回答。
        """
        # 5. 调用大模型生成回答
        response = self.call_llm(rag_prompt, temperature)
        
        # 6. 保存结果
        self._save_response(query, response)
        
        return response

    # def rag_query_concurrent(self, queries: List[str], top_k: int = 10, temperature: float = 0.7, max_workers: int = 5) -> List[Dict[str, Any]]:
    #     """
    #     并发执行多个RAG查询
        
    #     Args:
    #         queries: 查询列表
    #         top_k: 召回文档数量
    #         temperature: 大模型温度参数
    #         max_workers: 最大并发数
            
    #     Returns:
    #         包含查询和响应的字典列表
    #     """
    #     all_responses = []
        
    #     self.logger.info(f"开始并发RAG查询，共 {len(queries)} 个查询，最大并发数: {max_workers}")
        
    #     # 使用ThreadPoolExecutor进行并发处理
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         # 提交所有任务
    #         future_to_query = {
    #             executor.submit(self.rag_query, query, top_k, temperature): query 
    #             for query in queries
    #         }
            
    #         # 收集结果
    #         for future in concurrent.futures.as_completed(future_to_query):
    #             query = future_to_query[future]
    #             try:
    #                 response = future.result()
    #                 all_responses.append({
    #                     "query": query,
    #                     "response": response
    #                 })
    #                 self.logger.info(f"查询完成: {query[:50]}...")
    #             except Exception as e:
    #                 self.logger.error(f"查询 '{query}' 时发生错误: {e}")
    #                 all_responses.append({
    #                     "query": query,
    #                     "response": f"查询失败: {str(e)}"
    #                 })

    #     # 保存所有结果
    #     self._save_all_responses(all_responses)
        
    #     self.logger.info(f"并发RAG查询完成，共处理 {len(all_responses)} 个查询")
    #     return all_responses

# 使用示例
def main():
    # 创建RAG服务实例
    rag_service = RAGService("config.ini")

    # 测试Milvus Lite连接
    print("正在测试Milvus Lite连接...")
    if rag_service.test_milvus_lite_connection():
        print("✅ Milvus Lite连接测试成功")
        
        # 单次查询示例
        print("\n单次RAG查询示例:")
        single_query = """【指令】
  作为数据资产助手的意图识别专家，请分析用户查询，识别意图并提取槽位。
  若用户问题与意图清单都不相符，则返回 {intent: "40", slots: {}, query: "用户查询的原始问题"}。
  若用户询问助手本身的能力范围、功能清单、使用方法、身份定义等相关问题，则返回 {intent: "39", slots: {}, query: "用户查询的原始问题"}。
  若用户询问平台操作/帮助/名词解释/资产推荐，则返回 {intent: "38", slots: {}, query: "用户查询的原始问题"}。
  无需对识别结果进行解释或说明。

  【输出要求】
  1. 格式：严格遵循JSON结构 {"intent": "意图编码", "slots": {"槽位类型": "槽位值"}, "query": "用户查询的原始问题"}
  2. 约束：只输出JSON，不要包含任何解释、思考过程或其他文字
  3. 规则：无法提取的槽位直接省略，不要返回null或空字符串

  ═══════════════════════════════════════
  【意图清单】
  ═══════════════════════════════════════

  31 - 资产基础检索
    功能：基于名称/字段/域/类型/租户/排行/核心数据项/归属数据域/归属数据层/资产开放范围等条件查找资产
    核心槽位：AssetName, AssetType, BusinessDomain, FilterCondition, AssetRanking, OwnerTenant, FieldName, CoreDataItem, DataDomain, DataLayer, AssetOpenScope

  32 - 资产元数据查询
    功能：查询资产的业务口径/技术口径/负责人/数据存储周期/数据域/数据层/资产开放范围等元数据
    核心槽位：AssetName, MetadataItem（必填）, AssetType, BusinessDomain, FilterCondition, FieldName, AssertAdmin, AssertPublisher
    注意：MetadataItem支持同义词（"业务解释"→"业务口径"、"是干什么的"→"简介/用途"）

  33 - 资产质量与价值查询
    功能：查询资产的价值评分/星级/质量稽核
    核心槽位：AssetName, FilterCondition

  34 - 资产血缘关系查询
    功能：查询资产的上游依赖/下游应用/血缘图/依赖数量
    核心槽位：AssetName（必填）, LineageDirection, MetadataItem

  35 - 资产使用与工单查询
    功能：查询订阅/收藏/工单进度/审批/账期/API服务
    核心槽位：UserStatus, MetadataItem, AssetName, AssetType, FilterCondition, OrgName/UserName

  36 - 场景与标签推荐
    功能：基于业务专区/场景/业务概念推荐资产
    核心槽位：BusinessZone, CoreDataItem, AssetType

  37 - 资产复合对比与筛选
    功能：对比两个资产或进行多条件筛选
    核心槽位：AssetName（多个）, MetadataItem, AssetType, BusinessDomain, FilterCondition

  38 - 平台规则与帮助
    功能：询问平台操作/帮助/名词解释/资产推荐
    核心槽位：无

  39 - 助手能力与帮助
  功能：询问助手自身的能力范围、功能清单、使用方法、身份定义等相关问题
  核心槽位：无
  注意：仅针对“助手本身”的咨询，不包含平台规则、业务名词等非助手相关问题

  40 - OOD兜底
    功能：当用户查询与意图清单都不相符时，返回OOD兜底回答
    核心槽位：无

  ═══════════════════════════════════════
  【槽位定义】
  ═══════════════════════════════════════

  AssetName：资产完整名称（如"[在线公司]终端激活信息(日)"）
  MetadataItem：元数据项（业务口径/技术口径/负责人/血缘图/依赖数量/审批/账期/API服务）
  FieldName：字段名称（user_id/order_id）
  CoreDataItem：业务概念（5G登网/数据资产/移网用户是否活跃, 宽带用户是否活跃, 宽带下是否有异网号码等具体描述）
  BusinessDomain：业务域（M域/O域/B域/API来源/数据库同步）
  AssetType：资产类型（标签/数据表/模型资产/指标/API服务）
  BusinessZone：业务专区（公众智慧运营/一线赋能专区）
  FilterCondition：筛选条件（五星/高价值/最近一周更新的）
  LineageDirection：血缘方向（上游/下游/血缘图）
  OwnerTenant：归属租户（总部/分公司A/数据部）
  AssetRanking：综合排行（最新/本周上新/热门/订阅最多）
  AssertAdmin：资产管理员、负责人（张三）
  AssertPublisher：资产发布人（李四）
  DataDomain：归属数据域（固网视图/客户视图/公众客户信息/营销活动）
  DataLayer：归属数据层（ESD/DM/SRC）
  AssetOpenScope：资产开放范围（公共/私有/保护）

  ═══════════════════════════════════════
  【识别规则】
  ═══════════════════════════════════════

  1. AssetName必须识别完整名称（包括前缀[]和周期()），不要只识别部分
  2. MetadataItem需要识别同义词并规范化
  3. 多个同类型槽位使用数组：{"AssetName": ["资产1", "资产2"]}
  4. 无法提取的槽位直接省略，不返回null
  5. intent必须是31-38之一，无法识别时返回"38"
  6. slots是对象格式{"SlotType": "value"}，不是数组
  7. 输出纯JSON，不包含任何解释或说明
  8. 归属数据层识别为ESD、DM、SRC等
  9. query是用户查询的原始问题

  ═══════════════════════════════════════
  【输出格式示例】
  ═══════════════════════════════════════

  {"intent": "31", "slots": {"BusinessDomain": "M域", "AssetType": "标签"}, "query":"M域的所有标签资产"}
  {"intent": "32", "slots": {"AssetName": "宽带提质速率(月)", "MetadataItem": "业务口径"}, "query": "宽带提质速率(月)的业务口径是什么？"}
  {"intent": "34", "slots": {"AssetName": "HR系统", "LineageDirection": "上游"}, "query": "HR系统上游依赖有哪些？"}
  {"intent": "38", "slots": {}, "query": "平台有什么功能？"}
  {"intent": "38", "slots": {}, "query": "业务口径和技术口径的区别"}
  {"intent": "40", "slots": {}, "query": "今天天气怎么样？"}
  """
        result = rag_service.rag_query(single_query)
        print(f"查询: {single_query}")
        print(f"回答: {result}")
        
    else:
        print("❌ Milvus Lite连接测试失败，请检查配置")
        print("请检查以下配置:")
        print("1. Milvus Lite数据库文件是否存在")
        print("2. 集合是否已创建")
        print("3. pymilvus和milvus-lite是否正确安装")

if __name__ == "__main__":
    main()