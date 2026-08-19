import os

import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from src.core.config import settings


def get_device():
    if settings.torch_thread_limit:
        torch.set_num_threads(settings.torch_thread_limit)
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_embeddings(device=None):
    device = device or get_device()
    kwargs = dict(
        model_name=settings.embed_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        query_instruction="",
    )

    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        return HuggingFaceBgeEmbeddings(**kwargs)
    except Exception:
        os.environ.pop("HF_HUB_OFFLINE", None)
        return HuggingFaceBgeEmbeddings(**kwargs)
