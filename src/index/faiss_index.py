from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain.docstore.document import Document
from src.config import Config


def build_faiss(docs: List[Document]) -> FAISS:
    embeds = OpenAIEmbeddings(model=Config.EMBED_MODEL, api_key=Config.OPENAI_API_KEY)
    return FAISS.from_documents(docs, embeds)
