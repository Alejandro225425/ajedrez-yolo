"""Entrenamiento YOLO con transferencia y ajuste fino."""

import argparse
import shutil
from pathlib import Path

try:
    from .config import (
        BASE_MODEL, BATCH_SIZE, DATASET_DIR, DATA_YAML, EPOCHS,
        IMGSZ, MODEL_DIR, PHASE_1_EPOCHS, PATIENCE,
    )
except ImportError:  # Permite ejecutar como python src/train.py
    from config import (
        BASE_MODEL, BATCH_SIZE, DATASET_DIR, DATA_YAML, EPOCHS,
        IMGSZ, MODEL_DIR, PHASE_1_EPOCHS, PATIENCE,
    )


def device_disponible() -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Falta instalar torch. Instala PyTorch CUDA/CPU antes de entrenar."
        ) from exc
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return "0"
        except Exception as exc:
            print(f"[AVISO] CUDA no funciono; se usara CPU: {exc}")
    print("[AVISO] No se detecto CUDA utilizable; se usara CPU.")
    return "cpu"


def dataset_disponible(data_yaml: Path, dataset_dir: Path) -> bool:
    if not data_yaml.exists():
        return False
    train_images = dataset_dir / "images" / "train"
    train_labels = dataset_dir / "labels" / "train"
    val_images = dataset_dir / "images" / "val"
    val_labels = dataset_dir / "labels" / "val"
    return (
        train_images.is_dir()
        and train_labels.is_dir()
        and val_images.is_dir()
        and val_labels.is_dir()
        and any(train_images.glob("*.*"))
        and any(val_images.glob("*.*"))
        and any(train_labels.glob("*.txt"))
        and any(val_labels.glob("*.txt"))
    )


def entrenar(
    epochs: int,
    batch: int,
    imgsz: int,
    data_yaml: Path = DATA_YAML,
    dataset_dir: Path = DATASET_DIR,
) -> None:
    if not dataset_disponible(data_yaml, dataset_dir):
        raise FileNotFoundError(
            f"Dataset incompleto o no encontrado en {dataset_dir} / data YAML: {data_yaml}. "
            "Coloca primero train/val en dataset/images y dataset/labels; "
            "no se iniciara ningun entrenamiento."
        )

    device = device_disponible()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Falta instalar ultralytics. Ejecuta: pip install -r requirements.txt"
        ) from exc
    model = YOLO(BASE_MODEL)
    fase1_epochs = min(PHASE_1_EPOCHS, epochs)

    print(f"[FASE 1] Transfer learning por {fase1_epochs} epocas, freeze=10")
    fase1 = model.train(
        data=str(data_yaml),
        epochs=fase1_epochs,
        imgsz=imgsz,
        batch=batch,
        freeze=10,
        fliplr=0.0,
        patience=PATIENCE,
        workers=2,
        device=device,
        project="runs/detect",
        name="chess_phase1",
    )

    fase2_epochs = max(epochs - fase1_epochs, 0)
    if fase2_epochs == 0:
        print("[FASE 2] No quedan epocas para el ajuste fino.")
        fase1_best = Path(fase1.save_dir) / "weights" / "best.pt"
        if not fase1_best.exists():
            raise FileNotFoundError(f"No se encontro el modelo de fase 1: {fase1_best}")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fase1_best, MODEL_DIR / "best.pt")
        print(f"[OK] Modelo de fase 1 guardado en {MODEL_DIR / 'best.pt'}")
        return

    fase1_best = Path(fase1.save_dir) / "weights" / "best.pt"
    if not fase1_best.exists():
        raise FileNotFoundError(f"No se encontro el modelo de fase 1: {fase1_best}")

    print(f"[FASE 2] Ajuste fino por {fase2_epochs} epocas con lr bajo")
    modelo_fino = YOLO(str(fase1_best))
    fase2 = modelo_fino.train(
        data=str(data_yaml),
        epochs=fase2_epochs,
        imgsz=imgsz,
        batch=batch,
        freeze=0,
        fliplr=0.0,
        patience=PATIENCE,
        workers=2,
        device=device,
        project="runs/detect",
        name="chess_phase2",
        lr0=0.0005,
        lrf=0.01,
    )
    fase2_best = Path(fase2.save_dir) / "weights" / "best.pt"
    if not fase2_best.exists():
        raise FileNotFoundError(
            f"No se genero el modelo final de la fase 2: {fase2_best}"
        )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fase2_best, MODEL_DIR / "best.pt")
    print(f"[OK] Modelo final guardado en {MODEL_DIR / 'best.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el detector YOLO de ajedrez")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Epocas totales")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Tamano de batch")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="Tamano de entrada")
    parser.add_argument("--data", type=Path, default=DATA_YAML, help="Archivo data.yaml del dataset")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR, help="Directorio raiz del dataset")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch <= 0 or args.imgsz <= 0:
        parser.error("epochs, batch e imgsz deben ser valores positivos.")
    entrenar(
        args.epochs,
        args.batch,
        args.imgsz,
        args.data.expanduser().resolve(),
        args.dataset.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
