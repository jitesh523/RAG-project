import argparse
import os
import redis
from datetime import datetime
from pydantic import ValidationError
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import Config
from src.ingest.models import IngestData


def _load_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
        return loader.load()
    try:
        from unstructured.partition.auto import partition

        els = partition(filename=path)
        txt = "\n".join([str(e) for e in els])
        from types import SimpleNamespace

        return [
            SimpleNamespace(
                page_content=txt,
                metadata={
                    "source": os.path.basename(path),
                    "doc_type": ext[1:],
                    "date": datetime.utcfromtimestamp(os.path.getmtime(path))
                    .date()
                    .isoformat(),
                },
            )
        ]
    except Exception:
        return []


def load_docs(input_dir: str):
    out = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            path = os.path.join(root, f)
            docs = _load_file(path)
            for d in docs:
                if "source" not in d.metadata:
                    d.metadata["source"] = f
                out.append(d)
    return out


def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)




def enqueue(chunks):
    r = redis.Redis.from_url(Config.REDIS_URL, decode_responses=True)
    stream = Config.INGEST_STREAM
    for c in chunks:
        text = c.page_content
        meta = c.metadata or {}
        try:
            # Validate with shared model
            data = IngestData(
                text=text,
                source=str(meta.get("source", "")),
                page=int(meta.get("page", -1)),
                doc_type=str(meta.get("doc_type", "")),
                date=str(meta.get("date", "")),
            )
            r.xadd(stream, data.dict())
        except ValidationError as e:
            print(f"Validation failed for chunk: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    ds = load_docs(args.input)
    cs = chunk_docs(ds)
    enqueue(cs)
