"""
API FastAPI : detection des caries et assistant de traitement.

Routes :
  - GET  /health    : etat de l'API
  - POST /predict   : image (base64) -> nb de zones + surface cariee
  - POST /assistant : question (+ detection) -> reponse sourcee

Lancement : uvicorn app.api:app --reload
"""

import os
import sys
import base64

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import predict

app = FastAPI(title="Assistant caries")

_modele = predict.charger_modele() if predict.modele_disponible() else None
_assistant = None


class RequetePredict(BaseModel):
    image_base64: str


class RequeteAssistant(BaseModel):
    question: str
    nb_zones: int | None = None
    surface_pct: float | None = None


@app.get("/health")
def health():
    return {"status": "ok", "modele_charge": _modele is not None}


@app.post("/predict")
def predict_route(req: RequetePredict):
    if _modele is None:
        return {"erreur": "modele non entraine (unet_caries.pth absent)"}
    donnees = base64.b64decode(req.image_base64)
    image = cv2.imdecode(np.frombuffer(donnees, np.uint8), cv2.IMREAD_GRAYSCALE)
    _, stats = predict.segmenter(image, _modele)
    return stats


@app.post("/assistant")
def assistant_route(req: RequeteAssistant):
    global _assistant
    if _assistant is None:
        from src.rag.assistant import AssistantCaries
        _assistant = AssistantCaries()
    detection = None
    if req.nb_zones is not None:
        detection = {"nb_zones": req.nb_zones, "surface_pct": req.surface_pct}
    return _assistant.repondre(req.question, detection=detection)
