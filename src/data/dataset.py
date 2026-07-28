"""Dataset PyTorch pour la segmentation des caries (image -> masque binaire)."""

import os
import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src import config


def lister_paires(dossier_images, dossier_labels):
    """Retourne la liste des paires (chemin_image, chemin_masque) qui existent."""
    paires = []
    for chemin_img in sorted(glob.glob(os.path.join(dossier_images, "*.png"))):
        nom = os.path.basename(chemin_img)
        chemin_msk = os.path.join(dossier_labels, nom)
        if os.path.exists(chemin_msk):
            paires.append((chemin_img, chemin_msk))
    return paires


class CariesDataset(Dataset):
    """Charge une radio en niveaux de gris et son masque binaire de caries."""

    def __init__(self, paires, taille=config.TAILLE_IMAGE, seuil=config.SEUIL_MASQUE):
        self.paires = paires
        self.taille = taille
        self.seuil = seuil

    def __len__(self):
        return len(self.paires)

    def __getitem__(self, idx):
        chemin_img, chemin_msk = self.paires[idx]

        img = cv2.imread(chemin_img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.taille, self.taille))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, 0)  # (1, H, W)

        msk = cv2.imread(chemin_msk, cv2.IMREAD_GRAYSCALE)
        msk = cv2.resize(msk, (self.taille, self.taille), interpolation=cv2.INTER_NEAREST)
        msk = (msk > self.seuil).astype("float32")
        msk = np.expand_dims(msk, 0)  # (1, H, W)

        return torch.from_numpy(img), torch.from_numpy(msk)
