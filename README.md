# Caries Detection Assistant

Détection des caries sur radios panoramiques dentaires, couplée à un assistant
qui explique le résultat et propose les options de traitement avec sources.

## Objectif métier

Sur une radio, une partie des caries (débutantes ou cachées entre les dents) sont
manquées ou interprétées différemment selon le praticien. L'outil sert de
**second lecteur** : il repère et délimite les caries, puis aide à la décision de
traitement (obturation, dévitalisation, extraction). Cible : dentistes et
étudiants en médecine dentaire.

## Deux briques intégrées

- **Détection (Deep Learning)** : segmentation des caries par U-Net (localisation
  et sévérité au pixel près).
- **Assistant (RAG)** : à partir des caries détectées, réponses sourcées sur le
  diagnostic et les traitements.

## Stack technique

| Domaine | Outils |
|---|---|
| Modèle | Python, PyTorch, OpenCV |
| RAG | embeddings + base vectorielle (Chroma/FAISS), LLM via API |
| Démo | Streamlit |
| MLOps | MLflow, DVC, FastAPI |

## Données

Dataset **DC1000** : ~1000 radios panoramiques + masques de caries annotés
(~1,1 Go). Source : [MLUA](https://github.com/Zzz512/MLUA) —
[téléchargement](https://drive.google.com/file/d/1Xn1oGHvhGF9GbkcLEtCOV5QvWWqt1y62/view).
Les données ne sont pas versionnées sur GitHub (voir `.gitignore`).

## Structure

```
├── data/        # radios + masques (hors Git)
├── notebooks/   # EDA, essais
├── src/
│   ├── data/    # préparation, EDA
│   ├── model/   # entraînement + évaluation U-Net
│   └── rag/     # assistant de traitement
├── app/         # démo Streamlit
└── models/      # modèles entraînés (hors Git)
```
