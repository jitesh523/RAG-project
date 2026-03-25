import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from src.config import Config


def main(questions_path: str):
    embeddings = OpenAIEmbeddings(
        model=Config.EMBED_MODEL, api_key=Config.OPENAI_API_KEY
    )
    vs = FAISS.load_local("./faiss_store", embeddings=embeddings)
    retriever = vs.as_retriever(k=5)
    qs = json.load(open(questions_path))
    for q in qs:
        docs = retriever.get_relevant_documents(q["question"])
        print("Q:", q["question"])
        print("Top source:", docs[0].metadata.get("source"))
        print("-" * 60)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)
    a = p.parse_args()
    main(a.questions)
