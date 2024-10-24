import requests
import json

OPENSEARCH_URL = "<http://00.00.00.00:9200>" # 보안을 위해 임의로 설정

# 인덱스 생성
def create_index(index_name, mappings=None, settings=None):
    body = {}
    if mappings:
        body["mappings"] = mappings
    if settings:
        body['settings'] = settings

    responses = requests.put(f"{OPENSEARCH_URL}/{index_name}", data=json.dumps(body), headers={"Content-Type": "application/json"})
    return response.json()

# 인덱스 삭제
def delete_index(index_name):
    response = requests.delete(f"{OPENSEARCH_URL}/{index_name}")
    return response.json()

# 인덱스 목록
def list_index():
    response = requests.get(f"{OPENSEARCH_URL}/_cat/indices?v=true", headers={"Content-Type": "application/json"})
    return response.text

# 문서 ID를 지정해서 문서를 인덱싱
def index_document_with_id(index_name, doc_id, document):
    
    response = requests.put(f"{OPENSEARCH_URL}/_doc/{doc_id}", data=json.dumps({"doc": document}), headers={"Content-Type": "application/json"})
    return response.json()

# 문서 ID를 지정해서 문서 인덱스를 갱신
def index_update_document(index_name, doc_id, document):

    response = requests.post(f"{OPENSEARCH_URL}/{index_name}/_update/{doc_id}", data=json.dumps({"doc": document}), headers={"Content-Type": "application/json"})
    return response.json()

# 문서 ID를 지정해서 문서에 대한 벡터를 인덱싱
def update_document_vector(index_name, doc_id, vector):
    update_data = {
        "doc": {
            "vector": vector
        }
    }
    
    response = requests.post(f"{OPENSEARCH_URL}/{index_name}/_update/{doc_id}", data=json.dumps(update_data), headers={"Content-Type": "application/json"})
    return response.json()

# full_text와 page_texts를 사용해서 문서를 검색
def search_text_document(index_name, query):
    search_data = {
        "query": {
            "multi-match": {
                "query": query,
                "fields": ["full_text", "page_texts"]
            }
        }
    }
    
    response = requests.post(f"{OPENSEARCH_URL}/{index_name}/_search", data=json.dumps(search_data), headers={"Content-Type": "application/json"})
    return response.json()

# text 및 문서 속성들을 사용해서 문서를 검색
def search_full_document(index_name, query):
    search_data = {
        'query': query
    }

    response = requests.post(f"{OPENSEARCH_URL}/{index_name}/_search", data=json.dumps(search_data), headers={"Content-Type": "application/json"})
    return response.json()

# 문서 벡터를 사용해 유사 문서를 검색
def search_similar_document(index_name, vector, size=5):
    search_data = {
        'size': size,
        'query': {
            'script_score': {
                'query': {"match_all": {}},
                'script': {
                    'source': "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                    'params': {
                        'query_vector': vector
                    }
                }
            }
        }
    }
    
    response = requests.post(f"{OPENSEARCH_URL}/{index_name}/_search", data=json.dumps(search_data), headers={"Content-Type": "application/json"})
    return response.json()

# 필드 목록 획득
def get_fields_from_mapping(mapping, parent_name=None):
    fields = []
    properties = mapping.get('properties', {})

    for field_name, details in properties.items():
        full_field_name = f"{parent_name}.{field_name}" if parent_name else field_name
        if "properties" in details:
            fields.extend(get_fields_from_mapping(details, full_field_name))
        
        else:
            fields.append(full_field_name)

    return fields

# 인덱스에 대한 정보를 획득
def get_index_statistics(index_name):
    response = requests.get(f"{OPENSEARCH_URL}/{index_name}_stats")
    data = response.json()

    # 문서 개수, 전체 크기
    doc_count = data["indices"][index_name]["primaries"]["docs"]["count"]
    total_size_in_bytes = data["indices"][index_name]["primaries"]["store"]["size_in_bytes"]
    total_size_in_megabytes = total_size_in_bytes / (1024*1024)

    # 필드 목록 가져오기
    mapping_response = requests.get(f"{OPENSEARCH_URL}/{index_name}/_mapping")
    mappings = mapping_response.json()[index_name]["mappings"]
    field_list = get_fields_from_mapping(mappings)

    return {
        "Document Count": doc_count,
        "Total Size (MB)": total_size_in_megabytes,
        "Fields": field_list
    }