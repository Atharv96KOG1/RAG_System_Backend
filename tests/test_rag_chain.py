import pytest

from src.core.errors import MissingAPIKeyError
from src.rag import rag_chain

META = {"total_pages": 12, "total_tables": 3, "total_pictures": 2, "total_text_blocks": 40}


@pytest.mark.parametrize(
    "question,expected",
    [
        ("How many pages does this have?", "The document has 12 pages."),
        ("What's the page count?", "The document has 12 pages."),
        ("How many tables are there?", "The document has 3 tables."),
        ("Number of figures?", "The document has 2 pictures."),
        ("how many paragraphs", "The document has 40 text blocks."),
    ],
)
def test_answer_meta_question_matches_known_patterns(question, expected):
    assert rag_chain.answer_meta_question(question, META) == expected


def test_answer_meta_question_returns_none_for_content_questions():
    assert rag_chain.answer_meta_question("What is the termination clause?", META) is None


class _FakeDocument:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


def test_format_docs_tags_table_figure_and_text():
    docs = [
        _FakeDocument("row1|row2", {"has_table": True, "has_picture": False, "page": 3, "headings": "Sec 1"}),
        _FakeDocument("a chart", {"has_table": False, "has_picture": True, "page": 4, "headings": ""}),
        _FakeDocument("plain prose", {"has_table": False, "has_picture": False, "page": 5, "headings": ""}),
    ]

    out = rag_chain.format_docs(docs)

    assert "[TABLE | page 3 | Sec 1]\nrow1|row2" in out
    assert "[FIGURE | page 4 | ]\na chart" in out
    assert "[TEXT | page 5 | ]\nplain prose" in out
    assert out.count("\n\n---\n\n") == 2


def test_answer_query_empty_question_short_circuits_before_touching_chain():
    class ExplodingChain:
        def invoke(self, q):
            raise AssertionError("chain should never be invoked for an empty question")

    assert rag_chain.answer_query("", ExplodingChain(), META) == "Please enter a question."
    assert rag_chain.answer_query("   ", ExplodingChain(), META) == "Please enter a question."


def test_answer_query_meta_shortcut_skips_the_chain_entirely():
    class ExplodingChain:
        def invoke(self, q):
            raise AssertionError("chain should never be invoked for a meta question")

    assert rag_chain.answer_query("how many pages", ExplodingChain(), META) == "The document has 12 pages."


def test_answer_query_runs_full_chain_for_content_questions():
    class StubChain:
        def invoke(self, q):
            return f"answer to: {q}"

    assert (
        rag_chain.answer_query("what is the refund policy", StubChain(), META) == "answer to: what is the refund policy"
    )


def test_answer_query_handles_rate_limit_error(monkeypatch):
    class DummyRateLimitError(Exception):
        pass

    monkeypatch.setattr(rag_chain, "RateLimitError", DummyRateLimitError)

    class FailingChain:
        def invoke(self, q):
            raise DummyRateLimitError("429")

    result = rag_chain.answer_query("what is x", FailingChain(), META)
    assert "rate-limited" in result.lower()


def test_answer_query_handles_timeout_error(monkeypatch):
    class DummyTimeoutError(Exception):
        pass

    monkeypatch.setattr(rag_chain, "APITimeoutError", DummyTimeoutError)

    class FailingChain:
        def invoke(self, q):
            raise DummyTimeoutError("timed out")

    result = rag_chain.answer_query("what is x", FailingChain(), META)
    assert "timed out" in result.lower()


def test_answer_query_handles_generic_api_error(monkeypatch):
    class DummyAPIError(Exception):
        pass

    monkeypatch.setattr(rag_chain, "APIError", DummyAPIError)

    class FailingChain:
        def invoke(self, q):
            raise DummyAPIError("500")

    result = rag_chain.answer_query("what is x", FailingChain(), META)
    assert "error" in result.lower()


def test_load_llm_raises_clear_error_when_api_key_missing(monkeypatch):
    # settings is a singleton read once at process startup (src/core/config.py), so
    # tests patch the singleton's attribute directly rather than the env var — setting
    # the env var post-import would never reach an object that's already constructed.
    monkeypatch.setattr(rag_chain.settings, "openai_api_key", None)
    with pytest.raises(MissingAPIKeyError):
        rag_chain.load_llm()


def test_load_llm_succeeds_when_api_key_present(monkeypatch):
    monkeypatch.setattr(rag_chain.settings, "openai_api_key", "sk-test-key")
    llm = rag_chain.load_llm()
    assert llm is not None
