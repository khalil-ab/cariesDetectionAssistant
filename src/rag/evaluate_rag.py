"""
Evaluation du RAG (Bloc 4.3).

Sur un jeu de questions representatives :
  1. Pertinence du retrieval : la fiche source attendue est-elle bien recuperee
     dans le top-k ? (hit@k)
  2. Comparaison LLM seul vs LLM + RAG : on genere la reponse avec et sans
     contexte documentaire, pour montrer l'apport du RAG (reponses sourcees vs
     reponses non fondees).

Ecrit reports/rag_eval.json et reports/rag_eval_comparaison.md.

Usage : python -m src.rag.evaluate_rag
"""

import os
import json

from dotenv import load_dotenv

from src import config
from src.rag.assistant import AssistantCaries, GABARIT, PARAMS_LLM, _formater_detection

load_dotenv()

# Questions representatives + fiche source attendue
CAS = [
    ("Qu'est-ce qu'une carie et quels sont ses stades ?",
     "01_caries_definition_stades.md"),
    ("Comment traite-t-on une carie moderee ayant atteint la dentine ?",
     "02_traitement_par_stade.md"),
    ("Une carie limitee a l'email necessite-t-elle un fraisage ?",
     "02_traitement_par_stade.md"),
    ("Quand faut-il devitaliser une dent plutot que l'extraire ?",
     "03_devitalisation_vs_extraction.md"),
    ("Quels facteurs orientent le choix entre traitement de canal et extraction ?",
     "03_devitalisation_vs_extraction.md"),
    ("A quoi sert une radio panoramique pour depister les caries ?",
     "04_radio_panoramique_depistage.md"),
]


def _llm_seul(llm, question):
    """Reponse du LLM sans contexte documentaire (baseline sans RAG)."""
    prompt = (
        "Tu es un assistant en sante dentaire. Reponds en francais a la question "
        "suivante.\n\nQuestion : " + question + "\n\nReponse :"
    )
    return llm.invoke(prompt).content


def main():
    assistant = AssistantCaries(k=4)
    llm = assistant._get_llm()

    resultats = []
    hits = 0
    lignes_md = ["# Evaluation du RAG — LLM seul vs LLM + RAG\n"]

    for question, source_attendue in CAS:
        docs = assistant.rechercher(question)
        sources = [os.path.basename(d.metadata.get("source", "")) for d in docs]
        hit = source_attendue in sources
        hits += int(hit)

        rep_rag = assistant.repondre(question)
        rep_seul = _llm_seul(llm, question)

        resultats.append({
            "question": question,
            "source_attendue": source_attendue,
            "sources_recuperees": sources,
            "retrieval_hit": hit,
        })

        lignes_md += [
            f"\n## {question}",
            f"\n**Source attendue :** {source_attendue} — "
            f"**récupérée : {'oui' if hit else 'non'}** ({', '.join(sorted(set(sources)))})",
            "\n**LLM seul (sans RAG) :**\n", rep_seul,
            "\n**LLM + RAG (sourcé) :**\n", rep_rag["reponse"],
            "\n---",
        ]

    hit_rate = round(hits / len(CAS), 3)
    synthese = {
        "nb_questions": len(CAS),
        "retrieval_hit_at_4": hit_rate,
        "details": resultats,
    }

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    with open(os.path.join(config.REPORTS_DIR, "rag_eval.json"), "w",
              encoding="utf-8") as f:
        json.dump(synthese, f, indent=2, ensure_ascii=False)
    with open(os.path.join(config.REPORTS_DIR, "rag_eval_comparaison.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lignes_md))

    print(f"Retrieval hit@4 : {hit_rate} ({hits}/{len(CAS)})")
    print("Rapports -> reports/rag_eval.json et rag_eval_comparaison.md")


if __name__ == "__main__":
    main()
