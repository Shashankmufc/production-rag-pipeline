"""
FastAPI entrypoint. This version adds minimal JWT bearer auth so endpoints
require a valid token before they can ingest or query documents.
"""

from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from dotenv import load_dotenv

from auth.security import create_access_token, decode_access_token, get_password_hash, verify_password
from ingestion.loader import PDFLoader
from chunking.chunker import RecursiveChunker
from embeddings.embed import SentenceTransformerEmbedder
from vector_store.chromaDBstore import ChromaDBStore
from retrieval.retriever import VectorRetriever
from prompt.prompt_temp import RAGPromptTemplate
from llm.model import OpenAILLM
from rag_pipeline import RAGPipeline

load_dotenv()

pipeline: RAGPipeline | None = None
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

FAKE_USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": get_password_hash("admin123"),
    }
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    embedder = SentenceTransformerEmbedder()
    vector_store = ChromaDBStore()
    pipeline = RAGPipeline(
        loader=PDFLoader(),
        chunker=RecursiveChunker(chunk_size=500, chunk_overlap=50),
        embedder=embedder,
        vector_store=vector_store,
        retriever=VectorRetriever(embedder=embedder, vector_store=vector_store),
        prompt_template=RAGPromptTemplate(),
        llm=OpenAILLM(),
    )
    yield
    pipeline = None


app = FastAPI(title="Production RAG Service", version="0.1.0", lifespan=lifespan)


class IngestRequest(BaseModel):
    file_paths: list[str]


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]


def authenticate_user(username: str, password: str):
    user = FAKE_USERS_DB.get(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return username


@app.get("/health")
def health():
    return {"status": "ok", "documents_indexed": pipeline.vector_store.count()}


@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    token = create_access_token({"sub": form_data.username})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/ingest")
def ingest(req: IngestRequest, username: str = Depends(get_current_user)):
    try:
        num_chunks = pipeline.ingest(req.file_paths)
        return {"chunks_added": num_chunks, "total_in_store": pipeline.vector_store.count()}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, username: str = Depends(get_current_user)):
    if pipeline.vector_store.count() == 0:
        raise HTTPException(status_code=400, detail="No documents ingested yet. Call /ingest first.")
    result = pipeline.answer_typed(req.query, top_k=req.top_k)
    sources = sorted({c.metadata.get("filename", c.chunk_id) for c in result.retrieved_chunks})
    return QueryResponse(query=result.query, answer=result.answer, sources=sources)
