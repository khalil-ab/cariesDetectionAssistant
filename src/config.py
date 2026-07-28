"""Configuration centrale du projet (chemins et hyperparametres)."""

import os

# Racine du projet
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Racine du dataset DC1000. Par defaut on pointe vers l'emplacement telecharge ;
# surchargeable via la variable d'environnement CARIES_DATA_ROOT.
DATA_ROOT = os.environ.get(
    "CARIES_DATA_ROOT",
    r"D:\formationIA\filRouge\DC1000_dataset",
)

# Sous-ensembles du dataset
TRAIN_IMAGES = os.path.join(DATA_ROOT, "train", "images")
TRAIN_LABELS = os.path.join(DATA_ROOT, "train", "labels")
TEST_IMAGES = os.path.join(DATA_ROOT, "org_test_dataset", "images")
TEST_LABELS = os.path.join(DATA_ROOT, "org_test_dataset", "labels")

# Sorties
MODELS_DIR = os.path.join(RACINE, "models")
REPORTS_DIR = os.path.join(RACINE, "reports")

# Hyperparametres
TAILLE_IMAGE = 256      # les images sont redimensionnees en TAILLE x TAILLE
SEUIL_MASQUE = 127      # binarisation du masque (> seuil => carie)
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
