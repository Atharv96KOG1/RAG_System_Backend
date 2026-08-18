import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from src.core.config import settings


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_embeddings(device=None):
    device = device or get_device()
    return HuggingFaceBgeEmbeddings(
        model_name=settings.embed_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        query_instruction="",
    )
