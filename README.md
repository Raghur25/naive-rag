# naive-rag

A minimal Retrieval-Augmented Generation (RAG) pipeline over a PDF.

```
PDF ──► split into chunks ──► embed ──► Qdrant          (index.py)
question ──► embed ──► top matches from Qdrant ──► LLM ──► answer + page numbers   (chat.py)
```

- **Loader / splitter:** LangChain `PyPDFLoader`, `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
- **Vector store:** [Qdrant](https://qdrant.tech) (`docker-compose.yml`), collection `learning_rag`
- **Models:** any OpenAI-compatible server. The code expects an embedding model served as `embedding` and a chat model served as `qwen3`. Change the names in `index.py`, `chat.py` if yours differ.

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync                              # install dependencies
cp .env.example .env                 # then fill in OPENAI_BASE_URL and OPENAI_API_KEY
docker compose up -d                 # start Qdrant on localhost:6333
uv run index.py                      # index the PDF into Qdrant (run once)
uv run chat.py                       # ask questions
```

No PDF is included in this repo. Put your own PDF in this folder and set `pdf_path` in `index.py` (line 11) to its file name before running `index.py`. Re-running it adds the chunks a second time, so delete the `learning_rag` collection first if you want a clean re-index.

## Gotcha when using a self-hosted, OpenAI-compatible server

`OpenAIEmbeddings` in LangChain (Python) defaults to `check_embedding_ctx_length=True`. That makes it split text into **OpenAI tokenizer token IDs** and send those numbers instead of the text. A non-OpenAI embedding model (for example bge-m3) reads those IDs against its own vocabulary and produces meaningless vectors. Nothing errors; search just returns poor matches.

This repo sets `check_embedding_ctx_length=False` so the raw text is sent. If you use LangChain with a self-hosted embedding server, do the same.

## Limits of this "naive" version

- It retrieves a fixed number of chunks (LangChain's default is 4) with no re-ranking.
- One question per run, no conversation memory.
- Answers depend on the retrieved chunks, so check the page numbers it cites.

## Related

[rag-queue](https://github.com/Raghur25/rag-queue) wraps the same retrieval in a FastAPI service with a background worker (RQ + Valkey). It expects the `learning_rag` collection created here.
