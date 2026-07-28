"""
Fournisseur d'embeddings du RAG.

On utilise les embeddings Google Gemini (meme cle API que la generation, aucune
dependance lourde locale). Une alternative open-source locale (all-MiniLM-L6-v2
via sentence-transformers) est laissee en commentaire.
"""

from dotenv import load_dotenv

from src.rag._ssl import maybe_disable_ssl

load_dotenv()
maybe_disable_ssl()

MODELE_EMBED = "models/gemini-embedding-001"


def get_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=MODELE_EMBED, transport="rest")

    # Alternative locale (necessite sentence-transformers) :
    # from langchain_huggingface import HuggingFaceEmbeddings
    # return HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     model_kwargs={"device": "cpu"},
    # )
