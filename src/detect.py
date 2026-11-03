"""Deteccion YOLO de piezas de ajedrez."""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2

try:
    from .config import CONF_THRESHOLD, IOU_THRESHOLD, MODEL_BEST
except ImportError:  # Permite ejecutar como python src/detect.py
    from config import CONF_THRESHOLD, IOU_THRESHOLD, MODEL_BEST


def _cargar_modelo(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo entrenado: {model_path}. "
            "Completa el entrenamiento o indica --model."
        )
    from ultralytics import YOLO
    return YOLO(str(model_path))


def detect(
    image,
    model=None,
    conf: float = CONF_THRESHOLD,
    save_annotated: Optional[str] = None,
) -> List[Tuple[str, float, float, float]]:
    """Devuelve [(clase, x_centro, y_centro, confianza), ...]."""
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("La imagen proporcionada a detect() esta vacia.")
    if model is None:
        model = _cargar_modelo(MODEL_BEST)

    result = model.predict(
        source=image,
        conf=conf,
        iou=IOU_THRESHOLD,
        verbose=False,
    )[0]

    detecciones = []
    boxes = result.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        confianza = float(boxes.conf[i].item())
        nombre = result.names.get(cls_id, str(cls_id))
        detecciones.append((nombre, (x1 + x2) / 2.0, (y1 + y2) / 2.0, confianza))

    if save_annotated:
        anotada = result.plot()
        cv2.imwrite(str(save_annotated), anotada)
    return detecciones


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba una imagen con el modelo YOLO")
    parser.add_argument("image", help="Ruta de la imagen")
    parser.add_argument("--model", default=str(MODEL_BEST), help="Ruta de best.pt")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--save", help="Ruta opcional para guardar la imagen anotada")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {args.image}")
    model = _cargar_modelo(Path(args.model))
    detecciones = detect(image, model=model, conf=args.conf, save_annotated=args.save)
    print(detecciones)


if __name__ == "__main__":
    main()
