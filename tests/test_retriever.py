from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.rag.retriever import build_hybrid_retriever


class _FakeDenseRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content="apple banana smoothie", metadata={"src": "dense"})]


class _FakeVectorStore:
    def as_retriever(self, search_kwargs=None):
        return _FakeDenseRetriever()


def test_build_hybrid_retriever_blends_dense_and_keyword_results():
    lc_documents = [
        Document(page_content="apple banana smoothie", metadata={}),
        Document(page_content="quarterly revenue table", metadata={}),
    ]

    retriever = build_hybrid_retriever(_FakeVectorStore(), lc_documents)
    results = retriever.invoke("revenue")

    assert len(results) > 0
    assert any("revenue" in d.page_content for d in results)
