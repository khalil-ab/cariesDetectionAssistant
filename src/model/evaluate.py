"""
Evaluation du U-Net sur le jeu de test + comparaison a une baseline.

Baseline : seuillage d'Otsu (methode classique sans apprentissage) pour prouver
l'apport du modele. Ecrit reports/metrics_test.json.

Usage : python -m src.model.evaluate
"""

import os
import json

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import CariesDataset, lister_paires, split_dataset
from src.model.unet import UNet
from src.model.metrics import dice_coeff, iou_score


def dice_iou_numpy(pred, cible, eps=1e-6):
    pred = pred.astype("float32").ravel()
    cible = cible.astype("float32").ravel()
    inter = (pred * cible).sum()
    dice = (2 * inter + eps) / (pred.sum() + cible.sum() + eps)
    iou = (inter + eps) / (pred.sum() + cible.sum() - inter + eps)
    return dice, iou


def baseline_otsu(paires):
    """Seuillage d'Otsu : segmentation sans apprentissage (reference)."""
    dices, ious = [], []
    for chemin_img, chemin_msk in paires:
        img = cv2.imread(chemin_img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (config.TAILLE_IMAGE, config.TAILLE_IMAGE))
        _, pred = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        msk = cv2.imread(chemin_msk, cv2.IMREAD_GRAYSCALE)
        msk = cv2.resize(msk, (config.TAILLE_IMAGE, config.TAILLE_IMAGE),
                         interpolation=cv2.INTER_NEAREST)
        msk = (msk > config.SEUIL_MASQUE).astype("float32")
        d, i = dice_iou_numpy(pred, msk)
        dices.append(d)
        ious.append(i)
    return float(np.mean(dices)), float(np.mean(ious))


def evaluer_unet(paires, device):
    modele = UNet().to(device)
    chemin = os.path.join(config.MODELS_DIR, "unet_caries.pth")
    modele.load_state_dict(torch.load(chemin, map_location=device))
    modele.eval()

    loader = DataLoader(CariesDataset(paires), batch_size=config.BATCH_SIZE)
    dices, ious = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = modele(x)
            dices.append(dice_coeff(logits, y).item())
            ious.append(iou_score(logits, y).item())
    return float(np.mean(dices)), float(np.mean(ious))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Jeu de test de la MEME distribution que l'entrainement (imagettes train/),
    # jamais vu pendant l'entrainement grace au split deterministe.
    paires_all = lister_paires(config.TRAIN_IMAGES, config.TRAIN_LABELS)
    _, _, paires = split_dataset(paires_all)
    print("Images de test :", len(paires))

    dice_b, iou_b = baseline_otsu(paires)
    print(f"Baseline Otsu  : Dice {dice_b:.4f} | IoU {iou_b:.4f}")

    resultats = {"baseline_otsu": {"dice": dice_b, "iou": iou_b}}

    chemin_modele = os.path.join(config.MODELS_DIR, "unet_caries.pth")
    if os.path.exists(chemin_modele):
        dice_u, iou_u = evaluer_unet(paires, device)
        print(f"U-Net          : Dice {dice_u:.4f} | IoU {iou_u:.4f}")
        resultats["unet"] = {"dice": dice_u, "iou": iou_u}
    else:
        print("Modele U-Net absent (entrainement en ligne non encore fait).")

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    chemin = os.path.join(config.REPORTS_DIR, "metrics_test.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)
    print("Metriques ->", chemin)


if __name__ == "__main__":
    main()
