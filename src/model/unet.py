"""Architecture U-Net (segmentation binaire, entree 1 canal -> sortie 1 canal)."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv -> BatchNorm -> ReLU) x 2."""

    def __init__(self, entree, sortie):
        super().__init__()
        self.bloc = nn.Sequential(
            nn.Conv2d(entree, sortie, 3, padding=1),
            nn.BatchNorm2d(sortie),
            nn.ReLU(inplace=True),
            nn.Conv2d(sortie, sortie, 3, padding=1),
            nn.BatchNorm2d(sortie),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.bloc(x)


class UNet(nn.Module):
    """U-Net standard, profondeur 4, base 32 filtres (leger pour tourner en CPU/online)."""

    def __init__(self, canaux_entree=1, canaux_sortie=1, base=32):
        super().__init__()
        self.down1 = DoubleConv(canaux_entree, base)
        self.down2 = DoubleConv(base, base * 2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.down4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.conv4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)

        self.sortie = nn.Conv2d(base, canaux_sortie, 1)

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(self.pool(c1))
        c3 = self.down3(self.pool(c2))
        c4 = self.down4(self.pool(c3))

        b = self.bottleneck(self.pool(c4))

        u4 = self.conv4(torch.cat([self.up4(b), c4], dim=1))
        u3 = self.conv3(torch.cat([self.up3(u4), c3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), c1], dim=1))

        return self.sortie(u1)  # logits (pas de sigmoid ici)
