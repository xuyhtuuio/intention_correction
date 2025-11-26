from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# 连接 Milvus Lite
connections.connect(uri="./milvus.db")

# 查看所有集合
collections = utility.list_collections()
print("现有集合:", collections)

# 获取您要检查的 'intention' 集合
collection_name = "intention"
if utility.has_collection(collection_name):
    collection = Collection(collection_name)
    print(f"\n成功加载集合: {collection_name}")
    
    # 获取集合的 schema 信息
    collection_info = collection.schema
    print("\nCollection Fields:")
    
    primary_key_field = None
    vector_field = None
    
    for field in collection_info.fields:
        print(f"  - {field.name} (type: {field.dtype}, is_primary: {field.is_primary}, auto_id: {field.auto_id})")
        
        # 记录主键字段
        if field.is_primary:
            primary_key_field = field.name
            
        # 记录向量字段
        if field.dtype == DataType.FLOAT_VECTOR or field.dtype == DataType.BINARY_VECTOR:
            vector_field = field.name

    print(f"\nPrimary Key Field: {primary_key_field}")
    if vector_field:
        print(f"Vector Field: {vector_field}")
    else:
        print("Vector Field: Not found")
    
    # （可选）检查集合中的实体数量
    print(f"\n集合实体数量: {collection.num_entities}")

else:
    print(f"\n集合 '{collection_name}' 不存在。")

# 断开连接
connections.disconnect("default")