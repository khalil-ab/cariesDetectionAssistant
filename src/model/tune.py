"""
Recherche d'hyperparametres du U-Net (Bloc 6).

Compare plusieurs configurations (taux d'apprentissage, nombre de filtres de
base, taille d'image) sur le meme decoupage train/val. Chaque configuration est
loguee dans MLflow. La meilleure (Dice de validation) est sauvegardee et
documentee dans reports/hyperparam_search.json.

A lancer en ligne (GPU). Usage : python -m src.model.tune
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

import mlflow

from src import config
from src.data.dataset import CariesDataset, lister_paires, split_dataset
from src.model.unet import UNet
from src.model.metrics import DiceBCELoss, dice_coeff, iou_score
from src.model.train import evaluer, fixer_graine

# Grille de configurations a comparer (recherche manuelle / grid leger)
GRILLE = [
    {"lr": 1e-3, "base": 32, "taille": 256},
    {"lr": 5e-4, "base": 32, "taille": 256},
    {"lr": 1e-3, "base": 16, "taille": 256},
    {"lr": 1e-3, "base": 32, "taille": 192},
]
EPOCHS_TUNE = int(os.environ.get("EPOCHS_TUNE", "12"))


def entrainer_config(cfg, train_p, val_p, device):
    train_loader = DataLoader(
        CariesDataset(train_p, taille=cfg["taille"]),
        batch_size=config.BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        CariesDataset(val_p, taille=cfg["taille"]),
        batch_size=config.BATCH_SIZE, shuffle=False,
    )
    modele = UNet(base=cfg["base"]).to(device)
    optim = torch.optim.Adam(modele.parameters(), lr=cfg["lr"])
    perte_fn = DiceBCELoss()

    meilleur_dice = 0.0
    for _ in range(EPOCHS_TUNE):
        modele.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            perte_fn(modele(x), y).backward()
            optim.step()
        dice_val, _ = evaluer(modele, val_loader, device)
        meilleur_dice = max(meilleur_dice, dice_val)
    return meilleur_dice, modele


def main():
    fixer_graine(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device, "| epochs par config :", EPOCHS_TUNE)

    train_p, val_p, _ = split_dataset(
        lister_paires(config.TRAIN_IMAGES, config.TRAIN_LABELS)
    )

    mlflow.set_experiment("caries-hyperparam")
    resultats = []
    meilleur = {"dice_val": -1}
    meilleur_modele = None

    for i, cfg in enumerate(GRILLE, 1):
        with mlflow.start_run(run_name=f"config-{i}"):
            mlflow.log_params(cfg)
            dice_val, modele = entrainer_config(cfg, train_p, val_p, device)
            mlflow.log_metric("dice_val", dice_val)
            print(f"config {i} {cfg} -> dice_val {dice_val:.4f}")
            resultats.append({**cfg, "dice_val": round(dice_val, 4)})
            if dice_val > meilleur["dice_val"]:
                meilleur = {**cfg, "dice_val": round(dice_val, 4)}
                meilleur_modele = modele

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    if meilleur_modele is not None:
        torch.save(meilleur_modele.state_dict(),
                   os.path.join(config.MODELS_DIR, "unet_caries.pth"))

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    with open(os.path.join(config.REPORTS_DIR, "hyperparam_search.json"), "w",
              encoding="utf-8") as f:
        json.dump({"configs": resultats, "meilleure": meilleur}, f,
                  indent=2, ensure_ascii=False)

    print("\nMeilleure configuration :", meilleur)
    print("Rapport -> reports/hyperparam_search.json")


if __name__ == "__main__":
    main()
