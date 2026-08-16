# RAG Project

A modular Retrieval-Augmented Generation (RAG) project built in Python with FastAPI, ChromaDB, sentence-transformers, and OpenAI-compatible LLM access. The project is organized around a reusable pipeline that ingests PDFs, chunks them, embeds them, stores embeddings, retrieves relevant chunks, and answers questions with grounded context.

## Architecture overview

| Layer | Purpose | Key files |
| --- | --- | --- |
| API | Exposes FastAPI endpoints for auth, ingestion, and queries | [main.py](main.py) |
| Pipeline orchestrator | Coordinates ingestion, retrieval, prompt building, and answer generation | [rag_pipeline.py](rag_pipeline.py) |
| Ingestion | Loads source documents, mainly PDFs | [ingestion/loader.py](ingestion/loader.py) |
| Chunking | Splits documents into retrievable chunks | [chunking/chunker.py](chunking/chunker.py) |
| Embeddings | Converts text into vectors | [embeddings/embed.py](embeddings/embed.py) |
| Vector store | Stores and queries embeddings in ChromaDB | [vector_store/chromaDBstore.py](vector_store/chromaDBstore.py) |
| Retrieval | Retrieves the most relevant chunks | [retrieval/retriever.py](retrieval/retriever.py) |
| Prompting | Builds grounded prompts for the model | [prompt/prompt_temp.py](prompt/prompt_temp.py) |
| LLM abstraction | Wraps model providers such as OpenAI | [llm/model.py](llm/model.py) |
| Evaluation | Scores retrieval and answer quality | [evaluation/metrics.py](evaluation/metrics.py) |
| MLflow tracking | Logs experiments and metrics | [evaluation/mlflow_bridge.py](evaluation/mlflow_bridge.py) |
| LangSmith tracing | Optional hosted tracing and evaluation | [evaluation/langsmith_bridge.py](evaluation/langsmith_bridge.py) |

## Project flow

```text
PDFs / source docs
    -> Loader
    -> Chunker
    -> Embedder
    -> ChromaDB vector store
    -> Retriever
    -> Prompt builder
    -> LLM
    -> Final answer
```

The pipeline is intentionally modular so each layer can be swapped independently (for example, different embeddings, retrieval strategy, or model provider).

## Quickstart

### 1) Create and activate a Python environment

Recommended:

```bash
conda create -n ragproj311 python=3.11 -y
conda activate ragproj311
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Set up environment variables

Copy the example file:

```bash
copy .env.example .env
```

Then update `.env` with your real values.

### 4) Start the FastAPI app

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 5) Open Swagger UI

```text
http://127.0.0.1:8000/docs
```

### 6) Get a JWT token

Use the `/token` endpoint with:

- username: `admin`
- password: `admin123`

Then use the returned bearer token on protected routes.

### 7) Ingest documents

POST to `/ingest` with JSON like:

```json
{
  "file_paths": ["download/01_revenue_management_pricing.pdf"]
}
```

### 8) Query the pipeline

POST to `/query` with JSON like:

```json
{
  "query": "What is the difference between alpha and tau?",
  "top_k": 5
}
```

## Environment variables

This project expects values in a local `.env` file. Do not commit secrets to GitHub.

Example variables include:

- `OPENAI_API_KEY`
- `LANGCHAIN_API_KEY` (optional)
- `JWT_SECRET_KEY` (or a similar app secret for auth)

See [.env.example](.env.example) for the template.

## Evaluation

Run the benchmark suite:

```bash
python run_eval.py
```

This evaluates retrieval and answer quality using the dataset in [evaluation/eval_dataset.json](evaluation/eval_dataset.json) and logs metrics into MLflow.

### Evaluation metrics implemented

- Retrieval hit rate
- Mean reciprocal rank (MRR)
- Correctness (LLM-as-judge)
- Faithfulness (LLM-as-judge)

## MLflow

Start the UI:

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

## Project phases

- [x] Phase 1: Basic RAG app and modular architecture
- [x] Phase 2: Local evaluation harness and MLflow integration
- [x] Phase 3: OpenAI-compatible generation support
- [x] Phase 4: Retrieval improvements and experimentation
- [x] Phase 5: JWT-based API auth
- [x] Phase 6: Better observability and traceability
- [ ] Phase 7: Production-grade auth and user management
- [ ] Phase 8: Real-world corpus and benchmark tuning
- [ ] Phase 9: Frontend and deployment packaging

## Repository structure

```text
rag_project/
├── main.py
├── rag_pipeline.py
├── run_eval.py
├── requirements.txt
├── README.md
├── LICENSE
├── .env.example
├── download/
├── ingestion/
├── chunking/
├── embeddings/
├── evaluation/
├── llm/
├── prompt/
├── retrieval/
├── tests/
├── vector_store/
└── auth/
```

## Notes for portfolio / GitHub

This project is structured as a clean, modular RAG implementation with:

- FastAPI service endpoints
- vector-based document retrieval
- grounded LLM responses
- evaluation and metrics logging
- secure auth for protected ingestion endpoints
- MLflow experiment tracking

It is suitable as a portfolio project and a good base for extending into production RAG systems.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
