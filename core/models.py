"""
Inizializzazione dei modelli condivisi: YOLO, decoder Shurikode, DataAugmentor.
Importare da qui per avere una sola istanza in tutto il progetto.
"""

from shurikodecontext import Dec
from ultralytics import YOLO
from .DataAugmentor import DataAugmentor

YOLO_MODEL_PATH = "C:\\Users\\ricca\\Desktop\\shurikode_models\\ETRA_and_AI_and_clusters_and_crops.pt"

decoder_engine = Dec("L")
yolo_model     = YOLO(YOLO_MODEL_PATH)
augmentor      = DataAugmentor(clahe_clip_limit=1.0, clahe_tile_grid=(16, 16))
