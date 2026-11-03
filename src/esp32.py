"""Captura de frames desde una ESP32-CAM o fuentes locales."""

import argparse
import time

import cv2
import numpy as np
import requests

try:
    from .config import ESP32_CAPTURE_URL, ESP32_RETRIES, ESP32_TIMEOUT_SECONDS
except ImportError:  # Permite ejecutar como python src/esp32.py
    from config import ESP32_CAPTURE_URL, ESP32_RETRIES, ESP32_TIMEOUT_SECONDS


def capturar_esp32(
    url: str = ESP32_CAPTURE_URL,
    retries: int = ESP32_RETRIES,
    timeout: float = ESP32_TIMEOUT_SECONDS,
) -> np.ndarray:
    if retries < 1:
        raise ValueError("retries debe ser al menos 1.")
    ultimo_error = None
    for intento in range(1, retries + 1):
        try:
            respuesta = requests.get(url, timeout=timeout)
            respuesta.raise_for_status()
            frame = cv2.imdecode(
                np.frombuffer(respuesta.content, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is None:
                raise ValueError("La respuesta no contiene una imagen JPEG valida.")
            return frame
        except (requests.RequestException, ValueError) as exc:
            ultimo_error = exc
            print(f"[ESP32] Intento {intento}/{retries} fallido: {exc}")
            if intento < retries:
                time.sleep(0.5)
    raise ConnectionError(
        f"No se pudo capturar el frame desde {url} despues de {retries} intentos: {ultimo_error}"
    )


def crear_captura(source: str = "esp32", webcam_index: int = 0):
    if source == "esp32":
        return lambda: capturar_esp32()
    if source == "webcam":
        captura = cv2.VideoCapture(webcam_index)
        if not captura.isOpened():
            raise RuntimeError(f"No se pudo abrir la webcam con indice {webcam_index}.")
        return captura
    if source == "image":
        raise ValueError("Para imagen local use capturar_imagen().")
    raise ValueError("source debe ser 'esp32', 'webcam' o 'image'.")


def capturar_imagen(ruta: str) -> np.ndarray:
    frame = cv2.imread(ruta)
    if frame is None:
        raise FileNotFoundError(f"No se pudo leer la imagen local: {ruta}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba la captura de imagen")
    parser.add_argument("--source", choices=("esp32", "webcam", "image"), default="esp32")
    parser.add_argument("--url", default=ESP32_CAPTURE_URL)
    parser.add_argument("--image")
    parser.add_argument("--webcam", type=int, default=0)
    args = parser.parse_args()

    frame = capturar_imagen(args.image) if args.source == "image" else None
    if frame is None and args.source == "esp32":
        frame = capturar_esp32(args.url)
    if frame is None:
        captura = crear_captura(args.source, args.webcam)
        ok, frame = captura.read()
        if not ok:
            raise RuntimeError("No se pudo leer un frame de la webcam.")

    cv2.imshow("Prueba de captura", frame)
    print("Presiona cualquier tecla para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
