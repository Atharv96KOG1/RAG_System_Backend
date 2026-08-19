import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import chat, documents, graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s][%(name)s]: %(message)s")

app = FastAPI(title="GraphRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Range", "Accept-Ranges", "Content-Disposition", "ETag"],
)

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(graph.router)
