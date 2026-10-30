import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, EmailStr, Field, StringConstraints, field_validator

from nevis.embeddings import Embeddings, INDEX_VERSION, SIMILARITY_THRESHOLD
from nevis.store import Store
from nevis.reranker import RERANK_THRESHOLD

logger = logging.getLogger(__name__)
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
SocialLink = Annotated[str, StringConstraints(max_length=2048)]


class ClientInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    first_name: Name
    last_name: Name
    email: EmailStr
    description: str | None = Field(default=None, max_length=10_000)
    social_links: list[SocialLink] = Field(default_factory=list, max_length=20)


class Client(ClientInput):
    id: str


class DocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    content: str = Field(min_length=1, max_length=100_000)

    @field_validator("content")
    @classmethod
    def non_blank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Content must not be blank")
        return value


class Document(DocumentInput):
    id: str
    client_id: str
    created_at: datetime


class ClientResult(Client):
    type: Literal["client"]
    match_type: Literal["literal"]


class DocumentResult(Document):
    type: Literal["document"]
    match_type: Literal["literal", "semantic"]


SearchResult = Annotated[ClientResult | DocumentResult, Field(discriminator="type")]


class ErrorDetail(BaseModel):
    detail: str


UNAVAILABLE = {503: {"model": ErrorDetail, "description": "Storage or semantic search is unavailable"}}


def create_app(database_path: str | Path | None = None, embeddings=None) -> FastAPI:
    path = Path(database_path or os.environ.get("DATABASE_PATH", "data/nevis.sqlite3"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.embeddings = embeddings if embeddings is not None else Embeddings()
        path.parent.mkdir(parents=True, exist_ok=True)
        app.state.store = Store(path, INDEX_VERSION)
        yield

    app = FastAPI(title="Nevis Search API", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(sqlite3.Error)
    async def storage_error(request, error):
        logger.exception("Database operation failed", exc_info=error)
        return JSONResponse(status_code=503, content={"detail": "Storage temporarily unavailable"})

    @app.get("/health", tags=["Operations"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/clients", status_code=201, response_model=Client,
        responses={409: {"model": ErrorDetail, "description": "Email already belongs to a client"}, **UNAVAILABLE},
        tags=["Clients"],
    )
    def create_client(payload: ClientInput):
        try:
            return app.state.store.create_client(payload.model_dump())
        except sqlite3.IntegrityError as error:
            if error.sqlite_errorname != "SQLITE_CONSTRAINT_UNIQUE":
                raise
            raise HTTPException(status_code=409, detail="A client with this email already exists") from None

    @app.post(
        "/clients/{id}/documents", status_code=201, response_model=Document,
        responses={404: {"model": ErrorDetail, "description": "Client not found"}, **UNAVAILABLE},
        tags=["Documents"],
    )
    def create_document(id: str, payload: DocumentInput):
        if not app.state.store.client_exists(id):
            raise HTTPException(status_code=404, detail="Client not found")
        try:
            vectors = app.state.embeddings.document(payload.content)
        except ValueError as error:
            raise RequestValidationError([
                {"type": "value_error", "loc": ("body", "content"), "msg": str(error), "input": payload.content}
            ]) from None
        except Exception:
            logger.exception("Document embedding failed")
            raise HTTPException(status_code=503, detail="Semantic search temporarily unavailable") from None
        try:
            return app.state.store.create_document(id, payload.model_dump(), vectors)
        except LookupError:
            raise HTTPException(status_code=404, detail="Client not found") from None

    @app.get("/search", response_model=list[SearchResult], responses=UNAVAILABLE, tags=["Search"])
    def search(q: Annotated[str, Query(), StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]):
        try:
            vector = app.state.embeddings.query(q)
        except ValueError as error:
            raise RequestValidationError([
                {"type": "value_error", "loc": ("query", "q"), "msg": str(error), "input": q}
            ]) from None
        except Exception:
            logger.exception("Query embedding failed")
            raise HTTPException(status_code=503, detail="Semantic search temporarily unavailable") from None
        candidates = app.state.store.search(q, vector, SIMILARITY_THRESHOLD)
        semantic = [item for item in candidates if item['match_type'] == 'semantic']
        try:
            scores = app.state.embeddings.rerank(q, [item['content'] for item in semantic])
            matches = [(score, item) for score, item in zip(scores, semantic, strict=True)
                       if score >= RERANK_THRESHOLD]
        except Exception:
            logger.exception('Document relevance scoring failed')
            raise HTTPException(status_code=503, detail='Semantic search temporarily unavailable') from None
        matches.sort(key=lambda match: (-match[0], match[1]['id']))
        return ([item for item in candidates if item['match_type'] == 'literal']
                + [item for _, item in matches])

    return app


app = create_app()
