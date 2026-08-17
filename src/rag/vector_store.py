from langchain_milvus import Milvus
from pymilvus import MilvusClient

from src.core.config import settings


def get_milvus_client():
    return MilvusClient(uri=settings.milvus_uri)


def build_vectorstore(lc_documents, embeddings, collection_name):
    return Milvus.from_documents(
        documents=lc_documents,
        embedding=embeddings,
        collection_name=collection_name,
        connection_args={"uri": settings.milvus_uri},
        index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}},
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        drop_old=True,
    )
