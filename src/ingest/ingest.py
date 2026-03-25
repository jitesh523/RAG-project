import argparse
import os
import uuid
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from src.index.faiss_index import build_faiss
from src.index.milvus_index import insert_rows
from src.config import Config
from prometheus_client import Counter, Histogram, pushadd_to_gateway, REGISTRY

EMBED_BATCHES_TOTAL = Counter(
    "embed_batches_total",
    "Total embedding batches processed",
)

EMBED_ITEMS_TOTAL = Counter(
    "embed_items_total",
    "Total items embedded",
)

EMBED_BATCH_DURATION = Histogram(
    "embed_batch_duration_seconds",
    "Duration of embedding batches",
)


def load_pdfs(input_dir: str):
    docs = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(root, f))
                docs.extend(loader.load())
    return docs


def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


# Prometheus counter for ingestion throughput (documents processed)
INGEST_DOCS_TOTAL = Counter("ingest_documents_total", "Total documents/chunks ingested")

INGEST_RETRIES = Counter(
    "ingest_retries_total",
    "Total retries performed during ingestion",
    labelnames=["stage"],
)


def to_milvus_rows(chunks, embeddings):
    rows = []
    texts = [c.page_content for c in chunks]
    embs = []
    bs = max(1, Config.EMBED_BATCH_SIZE)
    for i in range(0, len(texts), bs):
        t_batch = texts[i : i + bs]
        attempt = 0
        delay = max(0.001, Config.RETRY_BASE_DELAY_MS / 1000.0)
        start = time.time()
        while True:
            try:
                part = embeddings.embed_documents(t_batch)
                break
            except Exception:
                attempt += 1
                if attempt >= max(1, Config.RETRY_MAX_ATTEMPTS):
                    raise
                INGEST_RETRIES.labels("embed").inc()
                time.sleep(delay)
                delay *= 2
        EMBED_BATCH_DURATION.observe(time.time() - start)
        EMBED_BATCHES_TOTAL.inc()
        EMBED_ITEMS_TOTAL.inc(len(t_batch))
        embs.extend(part)
    for emb, doc in zip(embs, chunks):
        rid = str(uuid.uuid4())[:32]
        src = doc.metadata.get("source", "")
        page = int(doc.metadata.get("page", -1))
        rows.append((rid, emb, doc.page_content[:65000], src, page))
    return rows


def _parse_model_map(map_str: str):
    mp = {}
    try:
        for p in [x.strip() for x in (map_str or "").split(",") if x.strip()]:
            k, v = p.split(":", 1)
            mp[k.strip()] = v.strip().lower()
    except Exception:
        pass
    return mp


def _select_embed_model(doc_type: str) -> str:
    mp = _parse_model_map(Config.EMBED_MODEL_MAP)
    key = (doc_type or "").lower() or "default"
    which = mp.get(key) or mp.get("default") or "large"
    return Config.EMBED_MODEL_SMALL if which == "small" else Config.EMBED_MODEL_LARGE


def main(input_dir: str, batch_size: int):
    docs = load_pdfs(input_dir)
    chunks = chunk_docs(docs)
    # Build a hot FAISS index for API service warm start (optional persist w/ .save_local)
    faiss_index = build_faiss(chunks)
    faiss_index.save_local("./faiss_store")

    # Persist everything to Milvus
    # PDFs => doc_type 'pdf' for model selection
    embed_model = _select_embed_model("pdf")
    embeddings = OpenAIEmbeddings(model=embed_model, api_key=Config.OPENAI_API_KEY)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        rows = to_milvus_rows(batch, embeddings)
        ins_attempt = 0
        ins_delay = max(0.001, Config.RETRY_BASE_DELAY_MS / 1000.0)
        while True:
            try:
                part = (
                    Config.INGEST_TENANT
                    if Config.MILVUS_PARTITIONED and Config.INGEST_TENANT
                    else None
                )
                insert_rows(rows, partition=part)
                break
            except Exception:
                ins_attempt += 1
                if ins_attempt >= max(1, Config.RETRY_MAX_ATTEMPTS):
                    raise
                INGEST_RETRIES.labels("insert").inc()
                time.sleep(ins_delay)
                ins_delay *= 2
        # metrics: increment counter and optionally push to Pushgateway
        INGEST_DOCS_TOTAL.inc(len(batch))
        if Config.PUSHGATEWAY_URL:
            try:
                pushadd_to_gateway(
                    Config.PUSHGATEWAY_URL, job="ingest", registry=REGISTRY
                )
            except Exception:
                pass
        print(f"[ingest] inserted {i + len(batch)}/{len(chunks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with PDFs")
    parser.add_argument("--batch_size", type=int, default=200)
    args = parser.parse_args()
    main(args.input, args.batch_size)
