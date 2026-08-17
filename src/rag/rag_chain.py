import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from openai import APIError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.errors import MissingAPIKeyError

PROMPT_SYSTEM = """You are a precise document Q&A assistant. You answer strictly from the context you are given —
never from general knowledge, and never by guessing to fill a gap.

Context format: each chunk is tagged "[TYPE | document name | page N | section heading]" followed by its text.
TYPE is TEXT, TABLE, FIGURE, or LIST — a FIGURE chunk's text is a caption/description/OCR read of an image, not
the image itself, a TABLE chunk's text is a full extracted table, not a summary of it, and a LIST chunk is a
bullet/numbered list kept intact as one unit — treat every item in it as belonging to that same list.

Rules:
- Use only facts present in the context. If the context doesn't contain the answer, say so plainly — do not
  hedge into a partial guess.
- Every claim in your answer must be traceable to a specific chunk. Cite as "(document name, page N)" inline,
  right after the sentence it supports — not bundled into one citation at the end.
- If the context mixes multiple source documents, keep facts from different documents clearly separated and
  never blend them into one unattributed statement.
- If a fact comes from a TABLE or FIGURE chunk, say so explicitly (e.g. "per the table on page 12...").
- Be complete: if the question asks for a list or all instances of something, include every one the context
  supports — do not truncate or say "and more" to save space.
- Be concise otherwise: no restating the question, no filler preamble like "Based on the context provided".
"""

PROMPT_TEMPLATE = """Context:
{context}

Question: {question}
"""

TABLE_PROMPT_TEMPLATE = """Act like best pdf Q&A expert. Using ONLY the context below, extract a table that answers
the question. Every row label, column header, and cell value must come entirely from the context and the question
below — never reuse items, names, or domains from any other document or prior example you may have seen.

A table has two dimensions: whatever varies down the ROWS, and whatever varies across the COLUMNS. Work out which
dimension is which from this specific question and context — there is no fixed rule, and it changes per question:

- If the question explicitly names which dimension it wants as rows (e.g. "with the features, not the approaches"),
  honor that literally: that named dimension is ROWS, the other is COLUMNS.
- Otherwise, put whichever dimension has MORE distinct values down the ROWS, and whichever has FEWER as COLUMNS.
  This keeps the table tall and readable instead of wide (e.g. a list of 20 leave categories each with a "days
  allocated" value should be 20 rows x 1-2 columns, not the reverse).

Per-row grounding — this is the part that most often goes wrong, so follow it exactly:
- Before filling in a cell, find the specific chunk that discusses THAT row's own entity. Only use a number or
  fact from that chunk for that row.
- Never borrow a value from a DIFFERENT row's entity just because it looks plausible or is nearby in the context —
  e.g. if leave type A's chunk never states a day-count but leave type B's chunk mentions "10 days" for something
  specific to B, that "10 days" belongs ONLY in B's row, never in A's, even if A's cell would otherwise be empty.
- If the row's own entity has no chunk supporting a value for a given column, write "Not specified" for that cell —
  an empty-looking table is correct if the source document genuinely doesn't state it that way; do not fill the
  gap with a number that belongs to something else.
- For each row, record which page(s) that row's OWN data actually came from in row_sources — not the citation for
  the table as a whole, the citation for that specific row.

Context:
{context}

Question: {question}
"""

META_QUESTION_PATTERNS = {
    "total_pages": [r"how many pages", r"number of pages", r"page count"],
    "total_tables": [r"how many tables", r"number of tables", r"table count"],
    "total_pictures": [r"how many (pictures|images|figures)", r"number of (pictures|images|figures)"],
    "total_text_blocks": [r"how many (text blocks|paragraphs)", r"number of (text blocks|paragraphs)"],
}


class ExtractedTable(BaseModel):
    """Structured shape for a table answer. Forcing this schema (instead of letting the
    LLM freeform a markdown table) is what guarantees a clean, consistent grid — a plain
    prompt instruction is easy for the model to ignore once it starts mixing prose and
    table together. Row/column orientation itself is NOT fixed here; the prompt decides
    which dimension goes where per-question, this schema just captures the result."""

    row_label: str = Field(description="What the rows represent, e.g. 'Leave Type', 'Feature', 'Vendor Section'.")
    rows: list[str] = Field(description="The row values for that dimension, grounded in the context.")
    columns: list[str] = Field(description="The column headers (the other dimension), grounded in the context.")
    cells: list[list[str]] = Field(
        description="cells[i][j] is the value for rows[i] under columns[j]. Must have len(rows) entries, each a list of len(columns) values."
    )
    row_sources: list[str] = Field(
        description="row_sources[i] is the page number(s) that ROW i's own data actually came from — not the "
        "table's overall citation, the specific chunk(s) that mention rows[i] by name. Must have len(rows) entries. "
        "This is what catches a value accidentally borrowed from a different row's chunk."
    )
    citation: str = Field(description="Source document name the table was built from.")


TABLE_INTENT_PATTERN = re.compile(r"\btables?\b|\bcompar(e|ison)\b|\bvs\.?\b|\bversus\b", re.I)


def is_table_request(question):
    return bool(TABLE_INTENT_PATTERN.search(question))


def _render_table(table):
    if not table.rows or not table.columns or not table.cells:
        return "The context didn't contain enough structured data to build a table."

    row_sources = list(table.row_sources) + ["?"] * (len(table.rows) - len(table.row_sources))

    header = f"| {table.row_label} | " + " | ".join(table.columns) + " | Row source (page) |"
    separator = "|---" * (len(table.columns) + 2) + "|"
    rendered_rows = []
    # strict=False: an LLM-returned table can have mismatched rows/cells/row_sources
    # lengths (e.g. it forgot a row's source) — truncate to the shortest rather than error.
    for row_label, row, source in zip(table.rows, table.cells, row_sources, strict=False):
        values = list(row) + ["Not specified"] * (len(table.columns) - len(row))
        rendered_rows.append(f"| {row_label} | " + " | ".join(values[: len(table.columns)]) + f" | {source} |")

    markdown = "\n".join([header, separator, *rendered_rows])
    return f"{markdown}\n\n_Source: {table.citation}_"


def build_table_chain(retriever, llm):
    # Structured JSON (entities + features + full cell grid + citation) needs more
    # headroom than a short prose answer, or the model truncates mid-table.
    structured_llm = llm.bind(max_tokens=900).with_structured_output(ExtractedTable)
    prompt = ChatPromptTemplate.from_template(TABLE_PROMPT_TEMPLATE)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | structured_llm
        | RunnableLambda(_render_table)
    )


def load_llm():
    api_key = settings.openai_api_key
    if not api_key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set. Add it to a .env file or export it before starting the app."
        )
    return ChatOpenAI(
        model=settings.llm_model_name,
        api_key=api_key,
        max_tokens=650,  # overridden per-chain via .bind() where a different budget is needed
        temperature=0,
    )


_CONTENT_TYPE_TAGS = {"table": "TABLE", "picture": "FIGURE", "list": "LIST", "text": "TEXT"}


def _content_tag(metadata):
    if "content_type" in metadata:
        return _CONTENT_TYPE_TAGS.get(metadata.get("content_type"), "TEXT")
    # Fallback for any caller still only setting the older has_table/has_picture flags.
    return "TABLE" if metadata.get("has_table") else ("FIGURE" if metadata.get("has_picture") else "TEXT")


def format_docs(docs):
    parts = []
    for d in docs:
        tag = _content_tag(d.metadata)
        source = d.metadata.get("source_file")
        source_tag = f"{source} | " if source else ""
        parts.append(
            f"[{tag} | {source_tag}page {d.metadata.get('page')} | {d.metadata.get('headings')}]\n{d.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def build_rag_chain(retriever, llm):
    prose_prompt = ChatPromptTemplate.from_messages([("system", PROMPT_SYSTEM), ("human", PROMPT_TEMPLATE)])
    prose_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prose_prompt
        # 300 tokens truncated multi-item list answers mid-sentence; prose answers need
        # more headroom than a one-line fact but not as much as a full structured table.
        | llm.bind(max_tokens=650)
        | StrOutputParser()
    )
    table_chain = build_table_chain(retriever, llm)

    def route(question):
        chain = table_chain if is_table_request(question) else prose_chain
        return chain.invoke(question)

    return RunnableLambda(route)


def answer_meta_question(question, doc_metadata):
    q = question.lower()
    for field, patterns in META_QUESTION_PATTERNS.items():
        if any(re.search(p, q) for p in patterns):
            label = field.replace("total_", "").replace("_", " ")
            return f"The document has {doc_metadata[field]} {label}."
    return None


def answer_query(question, rag_chain, doc_metadata):
    if not question or not question.strip():
        return "Please enter a question."

    meta_answer = answer_meta_question(question, doc_metadata)
    if meta_answer:
        return meta_answer

    try:
        return rag_chain.invoke(question)
    except RateLimitError:
        return "The language model is rate-limited right now. Please wait a moment and try again."
    except APITimeoutError:
        return "The language model timed out. Please try again."
    except APIError:
        return "The language model returned an error. Please try again in a moment."
