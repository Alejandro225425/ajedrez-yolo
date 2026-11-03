"""Evaluacion del modelo YOLO sobre val/test."""

import argparse
from pathlib import Path
import numpy as np

try:
    from .config import DATA_YAML, MODEL_BEST
except ImportError:  # Permite ejecutar como python src/evaluate.py
    from config import DATA_YAML, MODEL_BEST


def resolver_modelo(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo: {model_path}. Indica --model o ejecuta train.py primero."
        )
    from ultralytics import YOLO
    return YOLO(str(model_path))


def imprimir_confusion(metrics) -> None:
    cm = getattr(getattr(metrics, "confusion_matrix", None), "matrix", None)
    names = getattr(metrics, "names", None)
    if cm is None or names is None:
        print("[AVISO] Esta version de Ultralytics no expuso la matriz de confusion.")
        return

    cm = np.asarray(cm)
    filas = []
    for real in range(min(len(names), cm.shape[0])):
        for pred in range(min(len(names), cm.shape[1])):
            if real == pred:
                continue
            if cm[real, pred] > 0:
                filas.append((float(cm[real, pred]), names[real], names[pred]))
    filas.sort(reverse=True)
    print("\nClases mas confundidas (reales -> predichas):")
    if not filas:
        print("  No se detectaron confusiones relevantes.")
    else:
        for cantidad, real, pred in filas[:10]:
            print(f"  {real} -> {pred}: {cantidad:g}")


def evaluar(model_path: Path, split: str, batch: int, imgsz: int, data_yaml: Path = DATA_YAML) -> None:
    modelo = resolver_modelo(model_path)
    metrics = modelo.val(
        data=str(data_yaml),
        split=split,
        batch=batch,
        imgsz=imgsz,
        device="0" if _cuda_disponible() else "cpu",
        verbose=True,
    )
    print(f"\nResultados {split}:")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")

    print("\nMetricas por clase:")
    precisiones = getattr(metrics.box, "p", [])
    recalls = getattr(metrics.box, "r", [])
    mapas = getattr(metrics.box, "maps", [])
    for indice in range(len(metrics.names)):
        nombre = metrics.names.get(indice, str(indice))
        precision = float(precisiones[indice]) if indice < len(precisiones) else 0.0
        recall = float(recalls[indice]) if indice < len(recalls) else 0.0
        mapa = float(mapas[indice]) if indice < len(mapas) else 0.0
        print(
            f"  {indice:2d} {nombre}: precision={precision:.4f}, "
            f"recall={recall:.4f}, mAP50-95={mapa:.4f}"
        )
    imprimir_confusion(metrics)


def _cuda_disponible() -> bool:
    import torch
    return torch.cuda.is_available()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua el modelo de ajedrez")
    parser.add_argument("--model", default=str(MODEL_BEST), help="Ruta al modelo .pt")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--data", type=Path, default=DATA_YAML, help="Archivo data.yaml del dataset")
    args = parser.parse_args()
    if args.batch <= 0 or args.imgsz <= 0:
        parser.error("batch e imgsz deben ser positivos.")
    evaluar(
        Path(args.model),
        args.split,
        args.batch,
        args.imgsz,
        args.data.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
