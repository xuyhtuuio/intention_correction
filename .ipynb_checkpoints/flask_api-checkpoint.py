"""
Flask API服务 - 暴露RAG查询接口

安装依赖:
pip install flask flask-cors

启动服务:
python flask_app.py

API接口:
POST /rag_query - 执行RAG查询
GET /health - 健康检查
GET /status - 服务状态检查
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
import traceback
from datetime import datetime
import threading
import os

# 导入RAG服务
from llm_milvuslite import RAGService

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局RAG服务实例
rag_service = None
service_lock = threading.Lock()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/flask_app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_rag_service():
    """获取RAG服务实例（单例模式）"""
    global rag_service
    if rag_service is None:
        with service_lock:
            if rag_service is None:
                try:
                    logger.info("初始化RAG服务...")
                    rag_service = RAGService("config.ini")
                    logger.info("RAG服务初始化成功")
                except Exception as e:
                    logger.error(f"RAG服务初始化失败: {e}")
                    raise
    return rag_service

def create_response(success=True, data=None, message="", error_code=None):
    """创建标准化的API响应"""
    response = {
        "success": success,
        "timestamp": datetime.now().isoformat(),
        "message": message
    }

    if data is not None:
        response["data"] = data

    if error_code is not None:
        response["error_code"] = error_code

    return response

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify(create_response(
        success=True,
        message="服务运行正常",
        data={"status": "healthy", "service": "RAG API"}
    ))

@app.route('/status', methods=['GET'])
def status_check():
    """服务状态检查接口"""
    try:
        service = get_rag_service()

        # 检查各个服务的状态
        status = {
            "rag_service": "initialized",
            "milvus_lite": "connected" if service.milvus_collection else "disconnected",
            "embedding_api": "configured" if service.embedding_model_url else "not_configured",
            "rerank_api": "configured" if service.rerank_api_url else "not_configured",
            "llm_api": "configured" if service.model_api_url else "not_configured"
        }

        # 测试Milvus连接
        if service.milvus_collection:
            milvus_status = service.test_milvus_lite_connection()
            status["milvus_lite"] = "connected" if milvus_status else "connection_failed"

        return jsonify(create_response(
            success=True,
            message="状态检查完成",
            data=status
        ))

    except Exception as e:
        logger.error(f"状态检查失败: {e}")
        return jsonify(create_response(
            success=False,
            message=f"状态检查失败: {str(e)}",
            error_code="STATUS_CHECK_ERROR"
        )), 500

@app.route('/rag_query', methods=['POST'])
def rag_query():
    """
    RAG查询接口

    请求参数:
    {
        "query": "用户查询内容",
        "top_k": 10,  // 可选，召回文档数量，默认10
        "temperature": 0.7  // 可选，模型温度参数，默认0.7
    }

    响应:
    {
        "success": true,
        "data": {
            "query": "用户查询内容",
            "response": "模型回答"
        },
        "message": "查询完成",
        "timestamp": "2024-xx-xx xx:xx:xx"
    }
    """
    try:
        # 获取请求数据
        request_data = request.get_json()

        if not request_data:
            return jsonify(create_response(
                success=False,
                message="请求数据为空",
                error_code="EMPTY_REQUEST"
            )), 400

        # 验证必需参数
        query = request_data.get('query')
        if not query or not query.strip():
            return jsonify(create_response(
                success=False,
                message="查询内容不能为空",
                error_code="MISSING_QUERY"
            )), 400

        # 获取可选参数
        top_k = request_data.get('top_k', 10)
        temperature = request_data.get('temperature', 0.7)

        # 参数验证
        if not isinstance(top_k, int) or top_k <= 0:
            return jsonify(create_response(
                success=False,
                message="top_k必须是正整数",
                error_code="INVALID_TOP_K"
            )), 400

        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            return jsonify(create_response(
                success=False,
                message="temperature必须在0-2之间",
                error_code="INVALID_TEMPERATURE"
            )), 400

        logger.info(f"收到RAG查询请求: {query[:100]}...")

        # 获取RAG服务并执行查询
        service = get_rag_service()

        # 重要：rag_query方法内部已经调用了Milvus Lite，不需要重复调用
        response = service.rag_query(
            query=query.strip(),
            top_k=top_k,
            temperature=temperature
        )

        logger.info(f"RAG查询完成，回答长度: {len(response)} 字符")

        return jsonify(create_response(
            success=True,
            message="查询完成",
            data={
                "query": query,
                "response": response,
                "parameters": {
                    "top_k": top_k,
                    "temperature": temperature
                }
            }
        ))

    except Exception as e:
        logger.error(f"RAG查询失败: {e}")
        logger.error(f"错误详情: {traceback.format_exc()}")

        return jsonify(create_response(
            success=False,
            message=f"查询失败: {str(e)}",
            error_code="QUERY_ERROR"
        )), 500

@app.route('/rag_query_batch', methods=['POST'])
def rag_query_batch():
    """
    批量RAG查询接口

    请求参数:
    {
        "queries": ["查询1", "查询2", "查询3"],
        "top_k": 10,  // 可选
        "temperature": 0.7  // 可选
    }
    """
    try:
        request_data = request.get_json()

        if not request_data:
            return jsonify(create_response(
                success=False,
                message="请求数据为空",
                error_code="EMPTY_REQUEST"
            )), 400

        queries = request_data.get('queries')
        if not queries or not isinstance(queries, list):
            return jsonify(create_response(
                success=False,
                message="queries参数必须是数组",
                error_code="INVALID_QUERIES"
            )), 400

        # 验证查询内容
        valid_queries = []
        for i, query in enumerate(queries):
            if isinstance(query, str) and query.strip():
                valid_queries.append(query.strip())
            else:
                logger.warning(f"跳过无效查询 {i}: {query}")

        if not valid_queries:
            return jsonify(create_response(
                success=False,
                message="没有有效的查询内容",
                error_code="NO_VALID_QUERIES"
            )), 400

        # 限制批量查询数量
        if len(valid_queries) > 10:
            return jsonify(create_response(
                success=False,
                message="批量查询数量不能超过10个",
                error_code="TOO_MANY_QUERIES"
            )), 400

        top_k = request_data.get('top_k', 10)
        temperature = request_data.get('temperature', 0.7)

        logger.info(f"收到批量RAG查询请求，共 {len(valid_queries)} 个查询")

        service = get_rag_service()
        results = []

        # 逐个处理查询
        for i, query in enumerate(valid_queries):
            try:
                logger.info(f"处理查询 {i+1}/{len(valid_queries)}: {query[:50]}...")
                response = service.rag_query(
                    query=query,
                    top_k=top_k,
                    temperature=temperature
                )

                results.append({
                    "query": query,
                    "response": response,
                    "success": True
                })

            except Exception as e:
                logger.error(f"查询 '{query}' 失败: {e}")
                results.append({
                    "query": query,
                    "response": f"查询失败: {str(e)}",
                    "success": False
                })

        success_count = sum(1 for r in results if r["success"])
        logger.info(f"批量查询完成: {success_count}/{len(results)} 成功")

        return jsonify(create_response(
            success=True,
            message=f"批量查询完成: {success_count}/{len(results)} 成功",
            data={
                "results": results,
                "summary": {
                    "total": len(results),
                    "success": success_count,
                    "failed": len(results) - success_count
                },
                "parameters": {
                    "top_k": top_k,
                    "temperature": temperature
                }
            }
        ))

    except Exception as e:
        logger.error(f"批量RAG查询失败: {e}")
        return jsonify(create_response(
            success=False,
            message=f"批量查询失败: {str(e)}",
            error_code="BATCH_QUERY_ERROR"
        )), 500

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify(create_response(
        success=False,
        message="接口不存在",
        error_code="NOT_FOUND"
    )), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    logger.error(f"服务器内部错误: {error}")
    return jsonify(create_response(
        success=False,
        message="服务器内部错误",
        error_code="INTERNAL_ERROR"
    )), 500

@app.errorhandler(405)
def method_not_allowed(error):
    """405错误处理"""
    return jsonify(create_response(
        success=False,
        message="请求方法不允许",
        error_code="METHOD_NOT_ALLOWED"
    )), 405

if __name__ == '__main__':
    # 确保日志目录存在
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logger.info("启动Flask RAG API服务...")

    try:
        # 预初始化RAG服务
        service = get_rag_service()

        if service.milvus_collection:
            logger.info("Milvus Lite连接正常")
        else:
            logger.warning("Milvus Lite连接失败，将使用基础LLM模式")

        # 启动Flask服务
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # 生产环境设为False
            threaded=True
        )

    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        logger.error(f"请检查配置文件和环境依赖")
        exit(1)