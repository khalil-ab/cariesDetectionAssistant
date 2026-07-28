"""
Entrainement du U-Net de segmentation des caries.

Logue les hyperparametres et metriques dans MLflow, sauvegarde le meilleur
modele (meilleur Dice de validation) dans models/unet_caries.pth.

ATTENTION : entrainement lourd -> a lancer en ligne (GPU, ex. Colab/Kaggle),
pas sur une machine CPU. Voir notebooks/entrainement_colab.ipynb.

Usage : python -m src.model.train
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import mlflow

from src import config
from src.data.dataset import CariesDataset, lister_paires
from src.model.unet import UNet
from src.model.metrics import DiceBCELoss, dice_coeff, iou_score


def fixer_graine(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluer(modele, loader, device):
    modele.eval()
    dices, ious = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = modele(x)
            dices.append(dice_coeff(logits, y).item())
            ious.append(iou_score(logits, y).item())
    return float(np.mean(dices)), float(np.mean(ious))


def main():
    fixer_graine(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)

    paires = lister_paires(config.TRAIN_IMAGES, config.TRAIN_LABELS)
    print("Paires image/masque :", len(paires))
    train_p, val_p = train_test_split(
        paires, test_size=config.VAL_SPLIT, random_state=config.SEED
    )

    train_loader = DataLoader(
        CariesDataset(train_p), batch_size=config.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        CariesDataset(val_p), batch_size=config.BATCH_SIZE, shuffle=False
    )

    modele = UNet().to(device)
    optim = torch.optim.Adam(modele.parameters(), lr=config.LEARNING_RATE)
    perte_fn = DiceBCELoss()

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    chemin_modele = os.path.join(config.MODELS_DIR, "unet_caries.pth")

    mlflow.set_experiment("caries-segmentation")
    with mlflow.start_run():
        mlflow.log_params({
            "taille_image": config.TAILLE_IMAGE,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS,
            "lr": config.LEARNING_RATE,
            "modele": "unet",
        })

        meilleur_dice = 0.0
        for epoch in range(1, config.EPOCHS + 1):
            modele.train()
            pertes = []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                logits = modele(x)
                perte = perte_fn(logits, y)
                perte.backward()
                optim.step()
                pertes.append(perte.item())

            dice_val, iou_val = evaluer(modele, val_loader, device)
            perte_moy = float(np.mean(pertes))
            print(f"epoch {epoch}/{config.EPOCHS} | perte {perte_moy:.4f} "
                  f"| dice_val {dice_val:.4f} | iou_val {iou_val:.4f}")

            mlflow.log_metric("perte_train", perte_moy, step=epoch)
            mlflow.log_metric("dice_val", dice_val, step=epoch)
            mlflow.log_metric("iou_val", iou_val, step=epoch)

            if dice_val > meilleur_dice:
                meilleur_dice = dice_val
                torch.save(modele.state_dict(), chemin_modele)
                print("  -> meilleur modele sauvegarde")

        mlflow.log_metric("meilleur_dice_val", meilleur_dice)
        print("Termine. Meilleur Dice val :", round(meilleur_dice, 4))
        print("Modele ->", chemin_modele)


if __name__ == "__main__":
    main()
