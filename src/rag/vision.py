import base64
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.core.config import settings

logger = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "Describe this figure precisely and literally: name every labeled box/element, describe how they "
    "connect or flow into each other (arrows, order, hierarchy), and transcribe any visible numbers or "
    "text. Do not generalize or guess at meaning beyond what the image actually shows."
)

ANSWER_SYSTEM = """You are a precise document Q&A assistant with the actual source image attached, alongside
retrieved text context. Answer strictly from what the image shows and the given text context — never from
unrelated general knowledge, and never by guessing at details neither of them actually contains.

Context format: each chunk is tagged "[TYPE | document name | page N | section heading]" — cite claims as
"(document name, page N)" right after the sentence they support.

Rules:
- If neither the image nor the text context actually states the answer, say so plainly — do not construct a
  "likely" or "probably" interpretation by inferring from adjacent, superficially-related content. A rating
  scale, an acronym on an unrelated logo, or a nearby table are not evidence for what an unrelated term means
  unless the context explicitly says so.
- If you notice the image doesn't actually support answering the question, say that directly and stop —
  do not follow it with a speculative answer anyway. Noticing the mismatch and then answering around it is
  still a hallucination.
- Every claim must be traceable to a specific chunk or the image itself, cited as above. No claim should rest
  on "in many contexts, X typically means Y" reasoning — that is general knowledge, not this document's content.
"""

ANSWER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"


def _b64_data_url(image_bytes):
    return f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"


def load_vision_llm():
    if not settings.openrouter_api_key:
        return None
    return ChatOpenAI(
        model=settings.vision_model_name,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        max_tokens=500,
        temperature=0,
    )


def caption_picture(vision_llm, image_bytes):
    if vision_llm is None:
        return None
    message = HumanMessage(
        content=[
            {"type": "text", "text": CAPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": _b64_data_url(image_bytes)}},
        ]
    )
    try:
        return vision_llm.invoke([message]).content.strip()
    except Exception:
        logger.exception("Qwen3-VL captioning failed for a picture; continuing without it")
        return None


def answer_with_images(vision_llm, question, context_text, images_bytes):
    content = [{"type": "text", "text": ANSWER_TEMPLATE.format(context=context_text, question=question)}]
    for image_bytes in images_bytes:
        content.append({"type": "image_url", "image_url": {"url": _b64_data_url(image_bytes)}})
    messages = [SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=content)]
    return vision_llm.invoke(messages).content
