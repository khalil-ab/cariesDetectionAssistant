"""
Assistant de traitement (RAG).

Recupere les passages pertinents dans l'index Chroma et genere une reponse
sourcee avec Gemini. La reponse est adaptee au resultat de la detection
(nombre de zones cariees, surface) fourni par le Bloc 1.

Necessite la variable d'environnement GOOGLE_API_KEY (voir .env).

Usage direct : python -m src.rag.assistant
"""

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma

from src.rag.ingest import INDEX_DIR
from src.rag.embeddings import get_embeddings

load_dotenv()

GABARIT = """Tu es un assistant pedagogique en sante dentaire. Reponds en francais,
de facon claire et prudente, en te basant UNIQUEMENT sur le contexte fourni.
Si le contexte ne suffit pas, dis-le. Rappelle que tu n'es pas un diagnostic
medical et qu'un dentiste doit confirmer.

Resultat de l'analyse de la radio (detection automatique) :
{detection}

Contexte documentaire :
{contexte}

Question : {question}

Reponse (cite les elements du contexte utilises) :"""


def _formater_detection(detection):
    if not detection:
        return "Aucun resultat de detection fourni."
    return (
        f"- Zones cariees detectees : {detection.get('nb_zones', 'n/a')}\n"
        f"- Surface cariee estimee : {detection.get('surface_pct', 'n/a')} % de l'image"
    )


class AssistantCaries:
    def __init__(self, k=4):
        embeddings = get_embeddings()
        self.vectordb = Chroma(
            persist_directory=INDEX_DIR, embedding_function=embeddings
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": k})
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", temperature=0.0, transport="rest"
            )
        return self._llm

    def rechercher(self, question):
        return self.retriever.invoke(question)

    def repondre(self, question, detection=None):
        docs = self.rechercher(question)
        contexte = "\n\n".join(d.page_content for d in docs)
        sources = sorted({os.path.basename(d.metadata.get("source", "")) for d in docs})

        prompt = GABARIT.format(
            detection=_formater_detection(detection),
            contexte=contexte,
            question=question,
        )
        reponse = self._get_llm().invoke(prompt).content
        return {"reponse": reponse, "sources": sources}


if __name__ == "__main__":
    assistant = AssistantCaries()
    detection = {"nb_zones": 3, "surface_pct": 1.2}
    res = assistant.repondre(
        "J'ai plusieurs caries detectees, quels traitements sont possibles ?",
        detection=detection,
    )
    print("REPONSE :\n", res["reponse"])
    print("\nSOURCES :", res["sources"])
