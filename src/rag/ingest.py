"""
Construction de la base vectorielle du RAG.

Lit les fiches de rag_corpus/, les decoupe en chunks, calcule les embeddings
(all-MiniLM-L6-v2, local) et les stocke dans un index Chroma persistant.

Usage : python -m src.rag.ingest
"""

import os
import glob

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src import config
from src.rag.embeddings import get_embeddings

CORPUS_DIR = os.path.join(config.RACINE, "rag_corpus")
INDEX_DIR = os.path.join(config.RACINE, "chroma_caries")


def charger_documents():
    docs = []
    for chemin in sorted(glob.glob(os.path.join(CORPUS_DIR, "*.md"))):
        docs.extend(TextLoader(chemin, encoding="utf-8").load())
    return docs


def main():
    documents = charger_documents()
    print("Fiches chargees :", len(documents))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    print("Chunks generes :", len(chunks))

    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=INDEX_DIR
    )
    print("Index Chroma cree ->", INDEX_DIR)


if __name__ == "__main__":
    main()
