"""Fonctions de perte et metriques pour la segmentation binaire."""

import torch
import torch.nn as nn


def dice_coeff(logits, cibles, eps=1e-6):
    """Coefficient de Dice moyen sur le batch (apres sigmoid + seuil 0.5)."""
    probas = torch.sigmoid(logits)
    preds = (probas > 0.5).float()
    preds = preds.view(preds.size(0), -1)
    cibles = cibles.view(cibles.size(0), -1)
    inter = (preds * cibles).sum(1)
    union = preds.sum(1) + cibles.sum(1)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean()


def iou_score(logits, cibles, eps=1e-6):
    """Intersection over Union moyen sur le batch."""
    probas = torch.sigmoid(logits)
    preds = (probas > 0.5).float()
    preds = preds.view(preds.size(0), -1)
    cibles = cibles.view(cibles.size(0), -1)
    inter = (preds * cibles).sum(1)
    union = preds.sum(1) + cibles.sum(1) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


class DiceBCELoss(nn.Module):
    """Perte combinee BCE + Dice, robuste au fort desequilibre carie/fond."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, cibles, eps=1e-6):
        bce = self.bce(logits, cibles)
        probas = torch.sigmoid(logits)
        probas = probas.view(probas.size(0), -1)
        c = cibles.view(cibles.size(0), -1)
        inter = (probas * c).sum(1)
        dice = 1 - (2 * inter + eps) / (probas.sum(1) + c.sum(1) + eps)
        return bce + dice.mean()
