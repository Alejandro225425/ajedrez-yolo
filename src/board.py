"""Homografia, asignacion de casillas y construccion de FEN."""

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import chess

try:
    from .config import CORNER_JSON, WARP_SIZE
except ImportError:  # Permite ejecutar como python src/main.py
    from config import CORNER_JSON, WARP_SIZE


CLASE_A_PIEZA = {
    "white_pawn": chess.PAWN, "white_knight": chess.KNIGHT,
    "white_bishop": chess.BISHOP, "white_rook": chess.ROOK,
    "white_queen": chess.QUEEN, "white_king": chess.KING,
    "black_pawn": chess.PAWN, "black_knight": chess.KNIGHT,
    "black_bishop": chess.BISHOP, "black_rook": chess.ROOK,
    "black_queen": chess.QUEEN, "black_king": chess.KING,
}

Detection = Tuple[str, float, float, float]


def detectar_esquinas_tablero(image) -> np.ndarray | None:
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(gris, 50, 150)
    contornos, _ = cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for contorno in sorted(contornos, key=cv2.contourArea, reverse=True)[:20]:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        aproximado = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
        if len(aproximado) == 4 and area > image.shape[0] * image.shape[1] * 0.20:
            candidatos.append(aproximado.reshape(4, 2).astype("float32"))
    return ordenar_esquinas(candidatos[0]) if candidatos else None


def ordenar_esquinas(esquinas: np.ndarray) -> np.ndarray:
    puntos = np.asarray(esquinas, dtype="float32").reshape(4, 2)
    suma = puntos.sum(axis=1)
    diferencia = np.diff(puntos, axis=1).reshape(-1)
    return np.array([
        puntos[np.argmin(suma)],
        puntos[np.argmin(diferencia)],
        puntos[np.argmax(suma)],
        puntos[np.argmax(diferencia)],
    ], dtype="float32")


def seleccionar_esquinas(image, json_path: Path = CORNER_JSON) -> np.ndarray:
    if json_path.exists():
        datos = json.loads(json_path.read_text(encoding="utf-8"))
        return ordenar_esquinas(np.array(datos["corners"], dtype="float32"))

    puntos: List[Tuple[int, int]] = []
    imagen_original = image.copy()
    ventana = "Selecciona: TL, TR, BR, BL (r=reiniciar, enter=guardar)"

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 4:
            puntos.append((x, y))
            cv2.circle(image, (x, y), 7, (0, 0, 255), -1)
            cv2.imshow(ventana, image)

    cv2.namedWindow(ventana)
    cv2.setMouseCallback(ventana, click)
    while True:
        cv2.imshow(ventana, image)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla == ord("r"):
            puntos.clear()
            image[:] = imagen_original
        elif tecla == 13 and len(puntos) == 4:
            break
        elif tecla == ord("q"):
            cv2.destroyWindow(ventana)
            raise RuntimeError("Seleccion de esquinas cancelada.")
    cv2.destroyWindow(ventana)

    esquinas = ordenar_esquinas(np.array(puntos, dtype="float32"))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"corners": esquinas.tolist()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return esquinas


def obtener_esquinas(image, permitir_clic: bool = True) -> np.ndarray:
    if CORNER_JSON.exists():
        datos = json.loads(CORNER_JSON.read_text(encoding="utf-8"))
        return ordenar_esquinas(np.array(datos["corners"], dtype="float32"))
    detectadas = detectar_esquinas_tablero(image)
    if detectadas is not None:
        return detectadas
    if not permitir_clic:
        raise RuntimeError(
            "No se pudieron detectar las esquinas. Habilita la seleccion manual."
        )
    return seleccionar_esquinas(image)


def rectificar_tablero(image, esquinas: np.ndarray):
    destino = np.array([
        [0, 0], [WARP_SIZE - 1, 0],
        [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1],
    ], dtype="float32")
    matriz = cv2.getPerspectiveTransform(esquinas, destino)
    warpeada = cv2.warpPerspective(image, matriz, (WARP_SIZE, WARP_SIZE))
    return warpeada, matriz


def deteccion_a_casilla(
    deteccion: Sequence,
    matriz_homografia: np.ndarray,
    usar_punto_inferior: bool = True,
) -> str | None:
    if len(deteccion) < 4:
        return None
    _, cx, cy, _ = deteccion[:4]
    if len(deteccion) >= 5:
        cy = float(deteccion[4])  # Punto inferior-central opcional: (clase,cx,cy,conf,cy_inferior)
    punto = np.array([[[float(cx), float(cy)]]], dtype="float32")
    # Las detecciones estan en la imagen original; la homografia directa
    # convierte sus centros a coordenadas de la vista cenital 800x800.
    transformado = cv2.perspectiveTransform(punto, matriz_homografia)[0][0]
    columna = int(transformado[0] // (WARP_SIZE / 8))
    fila_visual = int(transformado[1] // (WARP_SIZE / 8))
    if not (0 <= columna < 8 and 0 <= fila_visual < 8):
        return None
    fila = 7 - fila_visual
    return chess.square_name(chess.square(columna, fila))


def construir_fen(
    detecciones: Iterable[Detection],
    matriz_homografia: np.ndarray,
    turno: chess.Color = chess.WHITE,
) -> Tuple[str, dict]:
    ocupacion = {}
    tablero = chess.Board(None)

    for deteccion in detecciones:
        clase, cx, cy, confianza = deteccion[:4]
        if clase not in CLASE_A_PIEZA:
            continue
        casilla = deteccion_a_casilla(deteccion, matriz_homografia)
        if casilla is None:
            continue
        actual = ocupacion.get(casilla)
        if actual is None or confianza > actual["confianza"]:
            ocupacion[casilla] = {
                "clase": clase,
                "pieza": CLASE_A_PIEZA[clase],
                "color": chess.WHITE if clase.startswith("white_") else chess.BLACK,
                "confianza": confianza,
            }

    for casilla, dato in ocupacion.items():
        tablero.set_piece_at(chess.parse_square(casilla), chess.Piece(dato["pieza"], dato["color"]))

    tablero.turn = turno
    return tablero.fen(), ocupacion


def dibujar_fen(warped_image, ocupacion: dict):
    imagen = warped_image.copy()
    for casilla, dato in ocupacion.items():
        square = chess.parse_square(casilla)
        columna = chess.square_file(square)
        fila = 7 - chess.square_rank(square)
        x = int((columna + 0.5) * WARP_SIZE / 8)
        y = int((fila + 0.5) * WARP_SIZE / 8)
        cv2.putText(
            imagen, dato["clase"].replace("_", " "),
            (x - 45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
            cv2.LINE_AA,
        )
    return imagen
