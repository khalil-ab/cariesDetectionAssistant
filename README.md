# Caries Detection Assistant

Détection des caries sur radios panoramiques dentaires, couplée à un assistant
qui explique le résultat et propose les options de traitement avec sources.

## Objectif métier

Sur une radio, une partie des caries (débutantes ou cachées entre les dents) sont
manquées ou interprétées différemment selon le praticien. L'outil sert de
**second lecteur** : il repère et délimite les caries, puis aide à la décision de
traitement (obturation, dévitalisation, extraction). Cible : dentistes et
étudiants en médecine dentaire. *Aide à la lecture, pas un diagnostic autonome.*

## Deux briques intégrées

- **Bloc 1 — Détection (Deep Learning)** : segmentation des caries par **U-Net**
  (localisation et sévérité au pixel près).
- **Bloc 2 — Assistant (RAG)** : le résultat de la détection (nombre de zones,
  surface) alimente un assistant qui répond avec des **sources** (embeddings +
  Chroma + Gemini).

Les deux servent le même cas : détecter, puis aider à décider.

## Stack technique

| Domaine | Outils |
|---|---|
| Modèle | Python, PyTorch, OpenCV |
| RAG | LangChain, embeddings + génération Gemini, base vectorielle Chroma |
| Démo | Streamlit |
| API | FastAPI |
| MLOps | MLflow (suivi), DVC, Docker |

## Données

Dataset **DC1000** : ~1000 radios panoramiques + masques de caries annotés
(~1,1 Go, images 384×384, caries ≈ 1 % des pixels → fort déséquilibre).
Source : [MLUA](https://github.com/Zzz512/MLUA) —
[téléchargement](https://drive.google.com/file/d/1Xn1oGHvhGF9GbkcLEtCOV5QvWWqt1y62/view).
Non versionné sur GitHub (voir `.gitignore`).

## Structure

```
├── data/                 # radios + masques (hors Git)
├── notebooks/
│   └── entrainement_colab.ipynb   # entrainement U-Net sur GPU en ligne
├── rag_corpus/           # fiches sources de l'assistant
├── src/
│   ├── config.py
│   ├── data/             # dataset, EDA
│   ├── model/            # U-Net, train, evaluate, predict
│   └── rag/              # ingest, assistant
├── app/
│   ├── app.py            # demo Streamlit
│   └── api.py            # API FastAPI
├── reports/              # sorties EDA + metriques
└── Dockerfile
```

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env      # renseigner GOOGLE_API_KEY
```

## Utilisation

**EDA**
```bash
python -m src.data.eda
```

**Entraînement (Bloc 1) — en ligne (GPU)**
Ouvrir `notebooks/entrainement_colab.ipynb` sur Google Colab (Runtime GPU),
exécuter les cellules, puis récupérer `models/unet_caries.pth` et le placer dans
`models/`.

**Évaluation + baseline**
```bash
python -m src.model.evaluate      # Dice/IoU U-Net vs baseline Otsu
```

**RAG (Bloc 2)**
```bash
python -m src.rag.ingest          # construit l'index Chroma
python -m src.rag.assistant       # test d'une question
```

**Démo**
```bash
streamlit run app/app.py
```

**API**
```bash
uvicorn app.api:app --reload      # http://localhost:8000/docs
```

## Notes

- **Entraînement** : machine locale sans GPU → l'entraînement se fait en ligne
  (notebook Colab). Le reste tourne en local.
- **Proxy SSL** : derrière un réseau qui intercepte le TLS, mettre
  `CARIES_INSECURE_SSL=1` dans `.env` (déjà géré). En ligne, laisser absent.
- **Baseline** : le seuillage d'Otsu obtient un Dice ≈ 0,004 sur le test —
  inexploitable, ce qui justifie l'approche Deep Learning.
