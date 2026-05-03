from pathlib import Path

import os


def _load_simple_dotenv(dotenv_path: Path = Path(__file__).parent / ".env") -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_simple_dotenv()

# ── Paths ─────────────────────────────────────────────────────
IMAGES_DIR      = Path(os.environ["IMAGES_DIR"])
LABELS_EXT_DIR  = Path(os.environ["LABELS_EXT_DIR"])
WORK_DIR        = Path(os.getenv("WORK_DIR", "./output"))
YOLO_MODEL_PATH = Path(os.environ["YOLO_MODEL_PATH"])
VGGT_EDGE_ENABLED = os.getenv("VGGT_EDGE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
VGGT_EDGE_SEARCH_GLOB = os.getenv("VGGT_EDGE_SEARCH_GLOB", "vggt")
VGGT_GENERATE_IF_MISSING = os.getenv("VGGT_GENERATE_IF_MISSING", "1").strip().lower() not in {"0", "false", "no", "off"}
VGGT_EMBEDDED_PYTHON = Path(os.getenv("VGGT_EMBEDDED_PYTHON", str(Path("./python312/python.exe"))))
VGGT_MODEL_ID = os.getenv("VGGT_MODEL_ID", "facebook/VGGT-1B")
VGGT_PREPROCESS_MODE = os.getenv("VGGT_PREPROCESS_MODE", "crop").strip().lower()
VGGT_3D_ADJACENCY_FACTOR = float(os.getenv("VGGT_3D_ADJACENCY_FACTOR", "1.35"))
VGGT_3D_MUTUAL_TOP_K = int(os.getenv("VGGT_3D_MUTUAL_TOP_K", "3"))
VGGT_3D_MIN_POINTS_PER_OBSERVATION = int(os.getenv("VGGT_3D_MIN_POINTS_PER_OBSERVATION", "12"))
VGGT_3D_MAX_POINTS_PER_OBSERVATION = int(os.getenv("VGGT_3D_MAX_POINTS_PER_OBSERVATION", "256"))

# ── Soglie e parametri ────────────────────────────────────────
CONF_THRESHOLD  = 0.5
YOLO_CONF       = 0.05
YOLO_IOU        = 0.3
YOLO_IMGSZ      = 1280
SKIP_NODES      = {}
VGGT_SKIP_NODES = {"10", "12", "17_1"}   # nodi senza overlap tra viste
MIN_IMAGES      = 1
K_NEIGHBORS     = 3
DIST_MULTIPLIER = 2.5

# ── Dati dei componenti idraulici ─────────────────────────────
# Formato: code -> (valenza_fisica, nome)
COMPONENT_INFO = {
    37:  (1, "Idrante"),       119: (2, "Rubinetto"),
    11:  (2, "Giunto"),        15:  (2, "Saracinesca"),
    27:  (2, "Gibault FL"),    53:  (2, "Tubo"),
    54:  (2, "Curva 22"),      75:  (2, "Riduzione"),
    77:  (2, "Tazza"),         95:  (2, "Bout bic."),
    111: (2, "Curva 45"),     175:  (2, "Curva 90"),
    190: (2, "Curva 30"),     195:  (2, "Curva 45P"),
    202: (2, "Antisfilo"),    207:  (2, "Bout norm."),
    235: (2, "Curva 90P"),    237:  (2, "Manicotto"),
    245: (2, "Curva 11"),
    83:  (3, "Rampa ab."),     85:  (3, "Rampa ind."),
    116: (3, "Tee FL"),       159:  (3, "Collare PVC"),
    201: (3, "Tee BIC"),      232:  (3, "Collare Fe"),
    255: (3, "Tee rid."),
    61:  (4, "Rampa cond."),  192:  (4, "Croce"),
    114: (0, "D:50"),  125: (0, "D:65"),  228: (0, "D:80"),
    231: (0, "D:100"), 244: (0, "D:150"), 247: (0, "D:200"),
    250: (0, "All.1\""), 252: (0, "All.2\""),
}

DIAMETER_CODES = {114, 125, 228, 231, 244, 247, 250, 252}

VALENCE_COLORS = {
    0: "#555", 1: "#e74c3c", 2: "#3498db", 3: "#2ecc71", 4: "#f39c12"
}

# ── Co-occorrenze empiriche ───────────────────────────────────
# Conteggio di coppie di tipi connesse osservate nel dataset reale.
# Ricalcolate su tutti i file annotations/node*_gt.json disponibili.
COOC_COUNTS = {
    ( 15,  95): 27,  # Saracinesca ↔ Bout bic.
    ( 15, 116): 26,  # Saracinesca ↔ Tee FL
    ( 15, 255): 16,  # Saracinesca ↔ Tee rid.
    ( 53,  95):  6,  # Tubo ↔ Bout bic.
    ( 15, 192):  6,  # Saracinesca ↔ Croce
    ( 11,  53):  5,  # Giunto ↔ Tubo
    ( 15, 175):  3,  # Saracinesca ↔ Curva 90
    (175, 255):  3,  # Curva 90 ↔ Tee rid.
    ( 53, 202):  3,  # Tubo ↔ Antisfilo
    ( 54,  95):  3,  # Curva 22 ↔ Bout bic.
    ( 15,  75):  3,  # Saracinesca ↔ Riduzione
    ( 75,  95):  3,  # Riduzione ↔ Bout bic.
    (195, 202):  3,  # Curva 45P ↔ Antisfilo
    ( 15,  27):  2,  # Saracinesca ↔ Gibault FL
    ( 95, 237):  2,  # Bout bic. ↔ Manicotto
    (190, 202):  2,  # Curva 30 ↔ Antisfilo
    ( 95, 202):  2,  # Bout bic. ↔ Antisfilo
    ( 11, 207):  2,  # Giunto ↔ Bout norm.
    ( 15, 207):  2,  # Saracinesca ↔ Bout norm.
    (192, 255):  2,  # Croce ↔ Tee rid.
    (116, 116):  1,  # Tee FL ↔ Tee FL
    (119, 159):  1,  # Rubinetto ↔ Collare PVC
    ( 83, 119):  1,  # Rampa ab. ↔ Rubinetto
    ( 15,  53):  1,  # Saracinesca ↔ Tubo
    ( 15,  77):  1,  # Saracinesca ↔ Tazza
    ( 11, 195):  1,  # Giunto ↔ Curva 45P
    (255, 255):  1,  # Tee rid. ↔ Tee rid.
    ( 95, 255):  1,  # Bout bic. ↔ Tee rid.
    ( 11,  15):  1,  # Giunto ↔ Saracinesca
    ( 11, 255):  1,  # Giunto ↔ Tee rid.
}

# ── Triplette empiriche A–centro–C ────────────────────────────
# Strutture locali osservate nel dataset: il nodo centrale ha due vicini A e C.
# Chiave: (min(a,c), centro, max(a,c)) — la simmetria è normalizzata.
# Usate per valutare la compatibilità contestuale di un arco proposto.
# Ricalcolate su tutti i file annotations/node*_gt.json disponibili.
TRIPLET_COUNTS = {
    ( 15, 116,  15): 23,  # Saracinesca — Tee FL — Saracinesca
    ( 95,  15, 116): 14,  # Bout bic. — Saracinesca — Tee FL
    ( 15, 255,  15):  9,  # Saracinesca — Tee rid. — Saracinesca
    ( 95,  15, 255):  8,  # Bout bic. — Saracinesca — Tee rid.
    ( 15, 192,  15):  7,  # Saracinesca — Croce — Saracinesca
    ( 15,  95,  53):  6,  # Saracinesca — Bout bic. — Tubo
    ( 11,  53,  95):  5,  # Giunto — Tubo — Bout bic.
    ( 15, 175, 255):  4,  # Saracinesca — Curva 90 — Tee rid.
    ( 15, 255, 255):  4,  # Saracinesca — Tee rid. — Tee rid.
    ( 15, 255, 192):  4,  # Saracinesca — Tee rid. — Croce
    ( 15, 192, 255):  4,  # Saracinesca — Croce — Tee rid.
    ( 15, 116, 116):  3,  # Saracinesca — Tee FL — Tee FL
    ( 95,  15, 192):  3,  # Bout bic. — Saracinesca — Croce
    ( 15,  75,  95):  3,  # Saracinesca — Riduzione — Bout bic.
    ( 15,  95, 237):  2,  # Saracinesca — Bout bic. — Manicotto
    ( 15,  95,  54):  2,  # Saracinesca — Bout bic. — Curva 22
    ( 53, 202, 195):  2,  # Tubo — Antisfilo — Curva 45P
    ( 75,  15, 255):  2,  # Riduzione — Saracinesca — Tee rid.
    (255,  15, 255):  2,  # Tee rid. — Saracinesca — Tee rid.
    ( 15, 255,  95):  2,  # Saracinesca — Tee rid. — Bout bic.
    ( 11, 207,  15):  2,  # Giunto — Bout norm. — Saracinesca
    (116,  15, 207):  2,  # Tee FL — Saracinesca — Bout norm.
    ( 11, 255,  15):  2,  # Giunto — Tee rid. — Saracinesca
    (116,  15, 116):  1,  # Tee FL — Saracinesca — Tee FL
    ( 27,  15, 175):  1,  # Gibault FL — Saracinesca — Curva 90
    ( 15, 175,  15):  1,  # Saracinesca — Curva 90 — Saracinesca
    ( 53,  15, 175):  1,  # Tubo — Saracinesca — Curva 90
    (175, 255, 175):  1,  # Curva 90 — Tee rid. — Curva 90
    (255, 175, 255):  1,  # Tee rid. — Curva 90 — Tee rid.
    ( 15, 255, 175):  1,  # Saracinesca — Tee rid. — Curva 90
    ( 27,  15, 255):  1,  # Gibault FL — Saracinesca — Tee rid.
    ( 83, 119, 159):  1,  # Rampa ab. — Rubinetto — Collare PVC
    ( 53, 202, 190):  1,  # Tubo — Antisfilo — Curva 30
    (202, 190, 202):  1,  # Antisfilo — Curva 30 — Antisfilo
    ( 95, 202, 190):  1,  # Bout bic. — Antisfilo — Curva 30
    ( 15,  95, 202):  1,  # Saracinesca — Bout bic. — Antisfilo
    ( 77,  15, 116):  1,  # Tazza — Saracinesca — Tee FL
    ( 75,  15, 116):  1,  # Riduzione — Saracinesca — Tee FL
    ( 75,  95, 202):  1,  # Riduzione — Bout bic. — Antisfilo
    ( 95, 202, 195):  1,  # Bout bic. — Antisfilo — Curva 45P
    (202, 195, 202):  1,  # Antisfilo — Curva 45P — Antisfilo
    (202,  53, 202):  1,  # Antisfilo — Tubo — Antisfilo
    ( 11, 195, 202):  1,  # Giunto — Curva 45P — Antisfilo
    ( 54,  95,  75):  1,  # Curva 22 — Bout bic. — Riduzione
    ( 95,  15,  95):  1,  # Bout bic. — Saracinesca — Bout bic.
    (255, 192, 255):  1,  # Tee rid. — Croce — Tee rid.
    ( 11,  15, 192):  1,  # Giunto — Saracinesca — Croce
}

# ── Saturazione del grado per tipo ───────────────────────────
# Frazione delle istanze nel dataset che raggiungono la valenza completa.
# IMPORTANTE: saturazione bassa significa che un'estremità del componente
# è tipicamente fuori dalla scena inquadrata → NON forzare quel collegamento.
DEGREE_SATURATION = {
    116: 1.00,  # Tee FL       — sempre visibile con tutti e 3 i bracci
    192: 1.00,  # Croce        — sempre visibile con tutti e 4 i bracci
    255: 1.00,  # Tee rid.     — sempre visibile con tutti e 3 i bracci
    202: 1.00,  # Antisfilo
    190: 1.00,  # Curva 30
    195: 1.00,  # Curva 45P
     75: 1.00,  # Riduzione
    207: 1.00,  # Bout norm.
    119: 1.00,  # Rubinetto
     15: 0.91,  # Saracinesca  — quasi sempre visibile con entrambi i lati
     53: 0.75,  # Tubo
     95: 0.44,  # Bout bic.    — spesso terminale (un'estremità sotto terra)
     83: 0.00,  # Rampa ab.    — sempre terminale nella scena
    159: 0.00,  # Collare PVC
     77: 0.00,  # Tazza
     11: 0.00,  # Giunto       — sempre terminale: l'altro lato è fuori scena
     54: 0.00,  # Curva 22
}

# ── Coppie proibite (empiriche dal dataset) ───────────────────
# Nessuna di queste coppie è mai stata osservata connessa nel dataset.
FORBIDDEN_PAIRS = frozenset([
    # ── Same-type pairs (anatomically impossible in hydraulics) ───────────────
    ( 11,  11),  # Giunto ↔ Giunto
    ( 15,  15),  # Saracinesca ↔ Saracinesca
    ( 53,  53),  # Tubo ↔ Tubo
    ( 54,  54),  # Curva 22 ↔ Curva 22
    ( 75,  75),  # Riduzione ↔ Riduzione
    ( 77,  77),  # Tazza ↔ Tazza
    ( 83,  83),  # Rampa ab. ↔ Rampa ab.
    ( 95,  95),  # Bout bic. ↔ Bout bic.
    (116, 116),  # Tee FL ↔ Tee FL
    (119, 119),  # Rubinetto ↔ Rubinetto
    (159, 159),  # Collare PVC ↔ Collare PVC
    (190, 190),  # Curva 30 ↔ Curva 30
    (192, 192),  # Croce ↔ Croce
    (195, 195),  # Curva 45P ↔ Curva 45P
    (202, 202),  # Antisfilo ↔ Antisfilo
    (207, 207),  # Bout norm. ↔ Bout norm.
    # ── Cross-type forbidden pairs ────────────────────────────────────────────
    ( 11,  54),  # Giunto ↔ Curva 22
    ( 11,  75),  # Giunto ↔ Riduzione
    ( 11,  77),  # Giunto ↔ Tazza
    ( 11,  83),  # Giunto ↔ Rampa ab.
    ( 11,  95),  # Giunto ↔ Bout bic.
    ( 11, 116),  # Giunto ↔ Tee FL
    ( 11, 119),  # Giunto ↔ Rubinetto
    ( 11, 159),  # Giunto ↔ Collare PVC
    ( 11, 190),  # Giunto ↔ Curva 30
    ( 11, 192),  # Giunto ↔ Croce
    ( 11, 202),  # Giunto ↔ Antisfilo
    ( 15,  53),  # Saracinesca ↔ Tubo
    ( 15,  54),  # Saracinesca ↔ Curva 22
    ( 15,  83),  # Saracinesca ↔ Rampa ab.
    ( 15, 119),  # Saracinesca ↔ Rubinetto
    ( 15, 159),  # Saracinesca ↔ Collare PVC
    ( 15, 190),  # Saracinesca ↔ Curva 30
    ( 15, 195),  # Saracinesca ↔ Curva 45P
    ( 15, 202),  # Saracinesca ↔ Antisfilo
    ( 53,  54),  # Tubo ↔ Curva 22
    ( 53,  75),  # Tubo ↔ Riduzione
    ( 53,  77),  # Tubo ↔ Tazza
    ( 53,  83),  # Tubo ↔ Rampa ab.
    ( 53, 116),  # Tubo ↔ Tee FL
    ( 53, 119),  # Tubo ↔ Rubinetto
    ( 53, 159),  # Tubo ↔ Collare PVC
    ( 53, 190),  # Tubo ↔ Curva 30
    ( 53, 192),  # Tubo ↔ Croce
    ( 53, 195),  # Tubo ↔ Curva 45P
    ( 53, 207),  # Tubo ↔ Bout norm.
    ( 53, 255),  # Tubo ↔ Tee rid.
    ( 54,  75),  # Curva 22 ↔ Riduzione
    ( 54,  77),  # Curva 22 ↔ Tazza
    ( 54,  83),  # Curva 22 ↔ Rampa ab.
    ( 54, 116),  # Curva 22 ↔ Tee FL
    ( 54, 119),  # Curva 22 ↔ Rubinetto
    ( 54, 159),  # Curva 22 ↔ Collare PVC
    ( 54, 190),  # Curva 22 ↔ Curva 30
    ( 54, 192),  # Curva 22 ↔ Croce
    ( 54, 195),  # Curva 22 ↔ Curva 45P
    ( 54, 202),  # Curva 22 ↔ Antisfilo
    ( 54, 207),  # Curva 22 ↔ Bout norm.
    ( 54, 255),  # Curva 22 ↔ Tee rid.
    ( 75,  77),  # Riduzione ↔ Tazza
    ( 75,  83),  # Riduzione ↔ Rampa ab.
    ( 75, 116),  # Riduzione ↔ Tee FL
    ( 75, 119),  # Riduzione ↔ Rubinetto
    ( 75, 159),  # Riduzione ↔ Collare PVC
    ( 75, 190),  # Riduzione ↔ Curva 30
    ( 75, 192),  # Riduzione ↔ Croce
    ( 75, 195),  # Riduzione ↔ Curva 45P
    ( 75, 202),  # Riduzione ↔ Antisfilo
    ( 75, 207),  # Riduzione ↔ Bout norm.
    ( 75, 255),  # Riduzione ↔ Tee rid.
    ( 77,  83),  # Tazza ↔ Rampa ab.
    ( 77,  95),  # Tazza ↔ Bout bic.
    ( 77, 116),  # Tazza ↔ Tee FL
    ( 77, 119),  # Tazza ↔ Rubinetto
    ( 77, 159),  # Tazza ↔ Collare PVC
    ( 77, 190),  # Tazza ↔ Curva 30
    ( 77, 192),  # Tazza ↔ Croce
    ( 77, 195),  # Tazza ↔ Curva 45P
    ( 77, 202),  # Tazza ↔ Antisfilo
    ( 77, 207),  # Tazza ↔ Bout norm.
    ( 77, 255),  # Tazza ↔ Tee rid.
    ( 83,  95),  # Rampa ab. ↔ Bout bic.
    ( 83, 116),  # Rampa ab. ↔ Tee FL
    ( 83, 159),  # Rampa ab. ↔ Collare PVC
    ( 83, 190),  # Rampa ab. ↔ Curva 30
    ( 83, 192),  # Rampa ab. ↔ Croce
    ( 83, 195),  # Rampa ab. ↔ Curva 45P
    ( 83, 202),  # Rampa ab. ↔ Antisfilo
    ( 83, 207),  # Rampa ab. ↔ Bout norm.
    ( 83, 255),  # Rampa ab. ↔ Tee rid.
    ( 95, 116),  # Bout bic. ↔ Tee FL
    ( 95, 119),  # Bout bic. ↔ Rubinetto
    ( 95, 159),  # Bout bic. ↔ Collare PVC
    ( 95, 190),  # Bout bic. ↔ Curva 30
    ( 95, 192),  # Bout bic. ↔ Croce
    ( 95, 195),  # Bout bic. ↔ Curva 45P
    ( 95, 207),  # Bout bic. ↔ Bout norm.
    (116, 119),  # Tee FL ↔ Rubinetto
    (116, 159),  # Tee FL ↔ Collare PVC
    (116, 190),  # Tee FL ↔ Curva 30
    (116, 192),  # Tee FL ↔ Croce
    (116, 195),  # Tee FL ↔ Curva 45P
    (116, 202),  # Tee FL ↔ Antisfilo
    (116, 207),  # Tee FL ↔ Bout norm.
    (116, 255),  # Tee FL ↔ Tee rid.
    (119, 190),  # Rubinetto ↔ Curva 30
    (119, 192),  # Rubinetto ↔ Croce
    (119, 195),  # Rubinetto ↔ Curva 45P
    (119, 202),  # Rubinetto ↔ Antisfilo
    (119, 207),  # Rubinetto ↔ Bout norm.
    (119, 255),  # Rubinetto ↔ Tee rid.
    (159, 190),  # Collare PVC ↔ Curva 30
    (159, 192),  # Collare PVC ↔ Croce
    (159, 195),  # Collare PVC ↔ Curva 45P
    (159, 202),  # Collare PVC ↔ Antisfilo
    (159, 207),  # Collare PVC ↔ Bout norm.
    (159, 255),  # Collare PVC ↔ Tee rid.
    (190, 192),  # Curva 30 ↔ Croce
    (190, 195),  # Curva 30 ↔ Curva 45P
    (190, 207),  # Curva 30 ↔ Bout norm.
    (190, 255),  # Curva 30 ↔ Tee rid.
    (192, 195),  # Croce ↔ Curva 45P
    (192, 202),  # Croce ↔ Antisfilo
    (192, 207),  # Croce ↔ Bout norm.
    (195, 207),  # Curva 45P ↔ Bout norm.
    (195, 255),  # Curva 45P ↔ Tee rid.
    (202, 207),  # Antisfilo ↔ Bout norm.
    (202, 255),  # Antisfilo ↔ Tee rid.
    (207, 255),  # Bout norm. ↔ Tee rid.
])
