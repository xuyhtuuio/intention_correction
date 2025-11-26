import json
import logging
import time
import os
from typing import List, Dict, Any
import configparser
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import requests
import numpy as np

class MilvusIntentionIngestor:
    def __init__(self, config_path: str = "config.ini"):
        """
        初始化Milvus意图数据导入器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='utf-8')
        
        # 读取配置
        self.embedding_api_url = self.config.get('DEFAULT', 'EMBEDDING_API_URL')
        self.embedding_model_name = self.config.get('DEFAULT', 'EMBEDDING_MODEL_NAME')
        self.intention_example_path = self.config.get('DEFAULT', 'INTENTION_EXAMPLE')
        self.milvus_db_path = self.config.get('DEFAULT', 'MILVUS_DB_PATH')
        self.collection_name = self.config.get('DEFAULT', 'MILVUS_COLLECTION_NAME')
        
        # 设置日志
        self._setup_logging()
        
        # 连接MilvusLite
        self._connect_milvus()
        
        # 清空现有集合（如果存在）并创建新集合
        self._reset_collection()

    def _setup_logging(self):
        """设置日志配置"""
        log_dir = self.config.get('DEFAULT', 'LOG_DIR', fallback='logs')
        log_file = self.config.get('DEFAULT', 'LOG_FILE', fallback='app.log')
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_path = os.path.join(log_dir, log_file)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _connect_milvus(self):
        """连接到MilvusLite数据库"""
        try:
            connections.connect(
                alias="default",
                uri=self.milvus_db_path
            )
            self.logger.info(f"成功连接到MilvusLite: {self.milvus_db_path}")
        except Exception as e:
            self.logger.error(f"连接MilvusLite失败: {e}")
            raise

    def _reset_collection(self):
        """清空现有集合（如果存在）并创建新集合"""
        try:
            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                self.logger.info(f"集合 {self.collection_name} 已存在，正在删除...")
                
                # 获取现有集合实例
                existing_collection = Collection(self.collection_name)
                
                # 删除现有集合
                existing_collection.drop()
                self.logger.info(f"集合 {self.collection_name} 已删除")
            
            # 创建新集合
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="input", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="output", dtype=DataType.JSON),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)  # 假设embedding维度为1024
            ]
            
            # 创建集合模式
            schema = CollectionSchema(fields, description="意图识别训练数据向量集合")
            
            # 创建集合
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # 创建索引
            self._create_index()
            
            self.logger.info(f"新集合 {self.collection_name} 创建成功")
            
        except Exception as e:
            self.logger.error(f"重置集合失败: {e}")
            raise

    def _create_index(self):
        """创建索引"""
        try:
            # 创建索引 - 使用支持的metric_type
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",  # 使用COSINE作为度量类型，适合文本相似度
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            self.logger.info("索引创建成功")
        except Exception as e:
            self.logger.error(f"创建索引失败: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的embedding向量
        
        Args:
            text: 输入文本
            
        Returns:
            embedding向量
        """
        try:
            headers = {
                'Content-Type': 'application/json'
            }
            
            payload = {
                "model": self.embedding_model_name,
                "input": text
            }
            
            response = requests.post(
                self.embedding_api_url,
                headers=headers,
                json=payload,
                timeout=30  # 30秒超时
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'data' in result and len(result['data']) > 0:
                    embedding = result['data'][0]['embedding']
                    self.logger.info(f"成功获取文本embedding，维度: {len(embedding)}")
                    return embedding
                else:
                    self.logger.error(f"embedding API响应格式异常: {result}")
                    return []
            else:
                self.logger.error(f"embedding API调用失败，状态码: {response.status_code}, 响应: {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"获取embedding时发生错误: {e}")
            return []

    def load_intention_examples(self) -> List[Dict[str, Any]]:
        """
        加载意图识别训练数据
        
        Returns:
            意图识别训练数据列表
        """
        try:
            with open(self.intention_example_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                self.logger.info(f"成功加载 {len(data)} 条意图识别训练数据")
                return data
            else:
                self.logger.error("意图识别训练数据格式错误，应为列表格式")
                return []
                
        except Exception as e:
            self.logger.error(f"加载意图识别训练数据失败: {e}")
            return []

    def insert_to_milvus(self, intention_examples: List[Dict[str, Any]], batch_size: int = 100):
        """
        批量插入数据到Milvus
        
        Args:
            intention_examples: 意图识别训练数据列表
            batch_size: 批次大小
        """
        total_records = len(intention_examples)
        self.logger.info(f"开始批量插入数据到Milvus，总记录数: {total_records}，批次大小: {batch_size}")
        
        for i in range(0, total_records, batch_size):
            batch = intention_examples[i:i + batch_size]
            self.logger.info(f"处理批次 {i//batch_size + 1}，记录数: {len(batch)}")
            
            # 准备数据
            inputs = []
            outputs = []
            embeddings = []
            
            for example in batch:
                input_text = example.get("input", "")
                output_data = example.get("output", {})
                
                if not input_text:
                    self.logger.warning(f"输入文本为空，跳过")
                    continue
                
                # 获取embedding（使用input文本）
                embedding = self.get_embedding(input_text)
                if not embedding:
                    self.logger.warning(f"输入 '{input_text}' 的embedding获取失败，跳过")
                    continue
                
                inputs.append(input_text)
                outputs.append(output_data)
                embeddings.append(embedding)
            
            if inputs:
                # 插入数据到Milvus
                try:
                    entities = [
                        inputs,
                        outputs,
                        embeddings
                    ]
                    
                    insert_result = self.collection.insert(entities)
                    self.logger.info(f"成功插入 {len(inputs)} 条记录，插入ID: {insert_result.insert_count}")
                    
                    # 刷新集合以确保数据立即可用
                    self.collection.flush()
                    
                except Exception as e:
                    self.logger.error(f"插入批次数据失败: {e}")
                    continue
            
            self.logger.info(f"批次 {i//batch_size + 1} 处理完成")
        
        self.logger.info("所有数据插入完成")

    def create_partition(self, partition_name: str):
        """
        创建分区
        
        Args:
            partition_name: 分区名称
        """
        raise NotImplementedError("此方法在清空模式下不适用，因为整个集合会被删除")

    def load_collection(self):
        """加载集合到内存"""
        try:
            # 等待索引构建完成
            self.logger.info("等待索引构建完成...")
            self.collection.load()
            self.logger.info(f"集合 {self.collection_name} 已加载到内存")
        except Exception as e:
            self.logger.error(f"加载集合失败: {e}")
            # 尝试等待一段时间后重试
            try:
                time.sleep(5)  # 等待5秒
                self.collection.load()
                self.logger.info(f"集合 {self.collection_name} 已加载到内存")
            except Exception as e2:
                self.logger.error(f"重试加载集合失败: {e2}")

    def run(self, batch_size: int = 100):
        """
        执行完整的导入流程
        
        Args:
            batch_size: 批次大小
        """
        self.logger.info("开始导入意图识别训练数据到Milvus")
        
        # 加载意图识别训练数据
        intention_examples = self.load_intention_examples()
        if not intention_examples:
            self.logger.error("没有意图识别训练数据可导入")
            return
        
        # 批量插入数据
        self.insert_to_milvus(intention_examples, batch_size)
        
        # 刷新集合确保数据持久化
        self.collection.flush()
        
        # 加载集合到内存以供查询
        self.load_collection()
        
        self.logger.info("意图识别训练数据导入完成")


# 使用示例
def main():
    # 创建Milvus意图数据导入器
    ingester = MilvusIntentionIngestor("config.ini")
    
    # 执行导入（可以调整批次大小）
    ingester.run(batch_size=50)  # 可以根据系统性能调整批次大小


if __name__ == "__main__":
    main()