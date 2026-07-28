"""
Analyse d'erreurs du U-Net (Bloc 3.3).

Sur le jeu de test (meme distribution), calcule le Dice par image, identifie les
meilleurs et pires cas, et quantifie faux positifs / faux negatifs. Sauvegarde
un histogramme, des exemples visuels et un resume.

Usage : python -m src.model.error_analysis
"""

import os
import json

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.data.dataset import lister_paires, split_dataset
from src.model.predict import charger_modele, CHEMIN_MODELE

LIMITE = int(os.environ.get("EA_LIMITE", "0"))  # 0 = tout le test


def _charger(chemin_img, chemin_msk, taille):
    img = cv2.resize(cv2.imread(chemin_img, cv2.IMREAD_GRAYSCALE), (taille, taille))
    msk = cv2.resize(cv2.imread(chemin_msk, cv2.IMREAD_GRAYSCALE), (taille, taille),
                     interpolation=cv2.INTER_NEAREST)
    return img, (msk > config.SEUIL_MASQUE).astype("uint8")


def main():
    device = "cpu"
    modele = charger_modele(device)
    _, _, test = split_dataset(lister_paires(config.TRAIN_IMAGES, config.TRAIN_LABELS))
    if LIMITE:
        test = test[:LIMITE]
    print("Images analysees :", len(test))

    taille = config.TAILLE_IMAGE
    resultats = []
    fp_total = fn_total = tp_total = 0

    for chemin_img, chemin_msk in test:
        img, gt = _charger(chemin_img, chemin_msk, taille)
        x = torch.from_numpy(img.astype("float32") / 255.0)[None, None].to(device)
        with torch.no_grad():
            pred = (torch.sigmoid(modele(x))[0, 0].numpy() > 0.5).astype("uint8")

        inter = int((pred & gt).sum())
        dice = (2 * inter) / (int(pred.sum()) + int(gt.sum()) + 1e-6)
        fp = int((pred & (1 - gt)).sum())
        fn = int(((1 - pred) & gt).sum())
        tp_total += inter
        fp_total += fp
        fn_total += fn
        resultats.append({"img": os.path.basename(chemin_img), "dice": dice,
                          "fp": fp, "fn": fn, "img_path": chemin_img,
                          "msk_path": chemin_msk})

    dices = np.array([r["dice"] for r in resultats])
    resume = {
        "nb_images": len(resultats),
        "dice_moyen": round(float(dices.mean()), 4),
        "dice_median": round(float(np.median(dices)), 4),
        "images_echec_dice_inf_0.1_%": round(float((dices < 0.1).mean()) * 100, 1),
        "faux_positifs_pixels": fp_total,
        "faux_negatifs_pixels": fn_total,
        "rappel_pixel": round(tp_total / (tp_total + fn_total + 1e-6), 4),
        "precision_pixel": round(tp_total / (tp_total + fp_total + 1e-6), 4),
    }
    print(json.dumps(resume, indent=2, ensure_ascii=False))

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    with open(os.path.join(config.REPORTS_DIR, "error_analysis.json"), "w",
              encoding="utf-8") as f:
        json.dump(resume, f, indent=2, ensure_ascii=False)

    # Histogramme des Dice par image
    plt.figure(figsize=(7, 4))
    plt.hist(dices, bins=30, color="#3b7dd8")
    plt.xlabel("Dice par image (test)")
    plt.ylabel("Nombre d'images")
    plt.title("Distribution du Dice par image")
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "error_hist_dice.png"))

    # 3 meilleurs et 3 pires cas (vert = verite, rouge = prediction)
    tries = sorted(resultats, key=lambda r: r["dice"])
    selection = tries[:3] + tries[-3:]
    plt.figure(figsize=(12, 8))
    for i, r in enumerate(selection):
        img, gt = _charger(r["img_path"], r["msk_path"], taille)
        x = torch.from_numpy(img.astype("float32") / 255.0)[None, None]
        with torch.no_grad():
            pred = (torch.sigmoid(modele(x))[0, 0].numpy() > 0.5).astype("uint8")
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        rgb[gt == 1] = [0, 200, 0]      # verite terrain en vert
        rgb[pred == 1] = [255, 0, 0]    # prediction en rouge (par-dessus)
        plt.subplot(2, 3, i + 1)
        plt.imshow(rgb)
        plt.title(f"Dice {r['dice']:.2f}")
        plt.axis("off")
    plt.suptitle("Pires cas (haut) et meilleurs cas (bas) — vert=reel, rouge=predit")
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "error_exemples.png"))
    print("Rapports -> reports/error_analysis.json + figures")


if __name__ == "__main__":
    main()
