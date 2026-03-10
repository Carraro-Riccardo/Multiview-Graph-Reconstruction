from PIL import Image
from pathlib import Path
import cv2
import numpy as np
from typing import Union

class DataAugmentor:
    def __init__(
        self,
        clahe_clip_limit: float = 1.0,
        clahe_tile_grid: tuple = (16, 16)
    ):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid

    def augment(self, image_input: Union[Image.Image, str, Path, np.ndarray]) -> Image.Image:
        """
        Miglioramento standard (CLAHE) per la scena intera o crop generici.
        Accetta Path, Numpy o PIL e restituisce PIL Image RGB.
        """
        # 1. Gestione robusta input (Risolve il tuo errore originale)
        img = self._load_as_pil(image_input)

        # 2. Pipeline standard: RGB -> Gray -> CLAHE -> RGB
        img = img.convert("RGB")
        gray = self._to_gray_scale(img)
        gray = self._enhance_contrast(gray)
        img = gray.convert("RGB")

        return img

    def postprocess_for_decoding(self, patch_bgr: np.ndarray) -> Image.Image:
        """
        Prepara la patch specificamente per il decoder (Binarizzazione + Otsu).
        Accetta Numpy BGR (da OpenCV) -> Restituisce PIL RGB.
        """
        # 1. BGR -> Gray
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

        # 2. CLAHE (Opzionale: aiuta se l'illuminazione sulla patch è irregolare)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        # 3. Binarizzazione Otsu (Bianco/Nero netto)
        _, binary = cv2.threshold(
            gray_eq,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 4. Il decoder vuole RGB, quindi convertiamo il binario (L) in RGB
        # (Nota: sarà un RGB in scala di grigi, ma tecnicamente RGB)
        pil_binary = Image.fromarray(binary, mode="L")
        return pil_binary.convert("RGB")

    # --- Metodi Helper ---

    def _load_as_pil(self, image_input) -> Image.Image:
        if isinstance(image_input, (str, Path)):
            return Image.open(image_input).convert("RGB")
        if isinstance(image_input, np.ndarray):
            # Se arriva array (probabilmente BGR da OpenCV), lo convertiamo
            if image_input.ndim == 3:
                return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            return Image.fromarray(image_input)
        if isinstance(image_input, Image.Image):
            return image_input
        raise ValueError(f"Input type {type(image_input)} not supported")

    def _to_gray_scale(self, image: Image.Image) -> Image.Image:
        return image.convert("L")

    def _enhance_contrast(self, gray_image: Image.Image) -> Image.Image:
        gray = np.array(gray_image, dtype=np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid,
        )
        enhanced = clahe.apply(gray)
        return Image.fromarray(enhanced, mode="L")