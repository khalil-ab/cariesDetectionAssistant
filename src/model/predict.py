"""
Inference : segmentation des caries sur une radio + statistiques (nb de zones,
surface). Utilise par la demo Streamlit et l'API.
"""

import os

import cv2
import numpy as np
import torch

from src import config
from src.model.unet import UNet

CHEMIN_MODELE = os.path.join(config.MODELS_DIR, "unet_caries.pth")


def modele_disponible():
    return os.path.exists(CHEMIN_MODELE)


def charger_modele(device="cpu"):
    modele = UNet().to(device)
    modele.load_state_dict(torch.load(CHEMIN_MODELE, map_location=device))
    modele.eval()
    return modele


def segmenter(image_gris, modele, device="cpu", seuil=0.5):
    """
    image_gris : np.ndarray (H, W) en niveaux de gris.
    Retourne (masque_binaire redimensionne a l'image, stats).
    """
    h0, w0 = image_gris.shape
    x = cv2.resize(image_gris, (config.TAILLE_IMAGE, config.TAILLE_IMAGE))
    x = x.astype("float32") / 255.0
    tenseur = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        proba = torch.sigmoid(modele(tenseur))[0, 0].cpu().numpy()

    masque = (proba > seuil).astype("uint8")
    masque = cv2.resize(masque, (w0, h0), interpolation=cv2.INTER_NEAREST)

    nb_zones, _ = cv2.connectedComponents(masque)
    nb_zones = max(nb_zones - 1, 0)  # on retire le fond
    surface_pct = round(float(masque.mean()) * 100, 3)

    stats = {"nb_zones": nb_zones, "surface_pct": surface_pct}
    return masque, stats


def superposer(image_gris, masque):
    """Retourne une image RGB avec les caries surlignees en rouge."""
    rgb = cv2.cvtColor(image_gris, cv2.COLOR_GRAY2RGB)
    rgb[masque == 1] = [255, 0, 0]
    return rgb
