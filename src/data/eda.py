"""
Analyse exploratoire du dataset DC1000.

Calcule des statistiques (nombre d'images, part de pixels caries, images sans
carie...) et sauvegarde des figures dans reports/.

Usage : python -m src.data.eda
"""

import os
import json

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.data.dataset import lister_paires


def analyser(paires, nom):
    proportions = []      # part de pixels "carie" par image
    sans_carie = 0
    tailles = set()
    for chemin_img, chemin_msk in paires:
        msk = cv2.imread(chemin_msk, cv2.IMREAD_GRAYSCALE)
        tailles.add(cv2.imread(chemin_img, cv2.IMREAD_GRAYSCALE).shape)
        binaire = (msk > config.SEUIL_MASQUE)
        part = binaire.mean()
        proportions.append(part)
        if binaire.sum() == 0:
            sans_carie += 1
    proportions = np.array(proportions)
    return {
        "sous_ensemble": nom,
        "nb_images": len(paires),
        "images_sans_carie": int(sans_carie),
        "images_avec_carie": int(len(paires) - sans_carie),
        "part_pixels_carie_moyenne_%": round(float(proportions.mean()) * 100, 3),
        "part_pixels_carie_max_%": round(float(proportions.max()) * 100, 3),
        "tailles_images": [list(t) for t in tailles],
    }, proportions


def main():
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    train = lister_paires(config.TRAIN_IMAGES, config.TRAIN_LABELS)
    test = lister_paires(config.TEST_IMAGES, config.TEST_LABELS)

    stats_train, prop_train = analyser(train, "train")
    stats_test, prop_test = analyser(test, "test")

    stats = {"train": stats_train, "test": stats_test}
    with open(os.path.join(config.REPORTS_DIR, "eda_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # Figure 1 : histogramme de la part de pixels caries
    plt.figure(figsize=(7, 4))
    plt.hist(prop_train * 100, bins=40, color="#3b7dd8")
    plt.xlabel("Part de pixels 'carie' par image (%)")
    plt.ylabel("Nombre d'images")
    plt.title("Distribution de la surface cariee (train)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "eda_distribution_caries.png"))

    # Figure 2 : quelques exemples image + masque superpose
    plt.figure(figsize=(12, 6))
    for i, (chemin_img, chemin_msk) in enumerate(train[:3]):
        img = cv2.imread(chemin_img, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(chemin_msk, cv2.IMREAD_GRAYSCALE)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title("Radio")
        plt.axis("off")
        plt.subplot(2, 3, i + 4)
        plt.imshow(img, cmap="gray")
        plt.imshow(np.ma.masked_where(msk <= config.SEUIL_MASQUE, msk),
                   cmap="autumn", alpha=0.7)
        plt.title("Caries (masque)")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "eda_exemples.png"))
    print("Figures ->", config.REPORTS_DIR)


if __name__ == "__main__":
    main()
