"""
Inizializzazione dei modelli condivisi: YOLO, decoder Shurikode, DataAugmentor.
Importare da qui per avere una sola istanza in tutto il progetto.
"""

from shurikodecontext import Dec
from ultralytics import YOLO
from .DataAugmentor import DataAugmentor
from config import YOLO_MODEL_PATH

decoder_engine = Dec("L")
yolo_model     = YOLO(str(YOLO_MODEL_PATH))
augmentor      = DataAugmentor(clahe_clip_limit=1.0, clahe_tile_grid=(16, 16))
