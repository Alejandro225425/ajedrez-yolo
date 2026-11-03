"""Pipeline completo de captura, deteccion, jugada y Stockfish."""

import argparse
from pathlib import Path

import cv2
import chess

try:
    from .config import MODEL_BEST, STOCKFISH_TIME_SECONDS, CORNER_JSON
    from .board import obtener_esquinas, rectificar_tablero, construir_fen, dibujar_fen
    from .detect import _cargar_modelo, detect
    from .engine import evaluar_fen
    from .esp32 import capturar_esp32, capturar_imagen, crear_captura
    from .moves import deducir_jugada
except ImportError:  # Permite ejecutar como python src/main.py
    from config import MODEL_BEST, STOCKFISH_TIME_SECONDS, CORNER_JSON
    from board import obtener_esquinas, rectificar_tablero, construir_fen, dibujar_fen
    from detect import _cargar_modelo, detect
    from engine import evaluar_fen
    from esp32 import capturar_esp32, capturar_imagen, crear_captura
    from moves import deducir_jugada


def procesar_frame(frame, modelo, esquinas, turno=chess.WHITE):
    warped, homografia = rectificar_tablero(frame, esquinas)
    detecciones = detect(frame, model=modelo)
    fen, ocupacion = construir_fen(detecciones, homografia, turno=turno)
    return fen, dibujar_fen(warped, ocupacion), detecciones


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de deteccion y jugadas")
    parser.add_argument("--source", choices=("esp32", "webcam", "image"), default="esp32")
    parser.add_argument("--image", help="Ruta de imagen local")
    parser.add_argument("--webcam", type=int, default=0)
    parser.add_argument("--model", default=str(MODEL_BEST))
    parser.add_argument("--initial-fen", default=chess.STARTING_FEN)
    parser.add_argument("--corners", action="store_true", help="Forzar seleccion manual de esquinas")
    args = parser.parse_args()

    modelo = _cargar_modelo(Path(args.model))
    frame = None
    captura = None

    if args.source == "image":
        if not args.image:
            parser.error("--image es obligatorio con --source image.")
        frame = capturar_imagen(args.image)
    elif args.source == "esp32":
        frame = capturar_esp32()
    else:
        captura = crear_captura("webcam", args.webcam)

    if frame is None:
        ok, frame = captura.read()
        if not ok:
            raise RuntimeError("No se pudo obtener el primer frame.")

    esquinas = obtener_esquinas(frame, permitir_clic=args.corners)
    fen_confirmado = args.initial_fen
    print(f"FEN inicial: {fen_confirmado}")
    print("Espacio: confirmar la deteccion. q: salir.")

    try:
        while True:
            if args.source != "image":
                if args.source == "esp32":
                    frame = capturar_esp32()
                else:
                    ok, frame = captura.read()
                    if not ok:
                        print("[AVISO] No se pudo leer un nuevo frame.")
                        continue

            try:
                turno_actual = chess.Board(fen_confirmado).turn
                fen_actual, tablero_anotado, detecciones = procesar_frame(
                    frame, modelo, esquinas, turno=turno_actual
                )
            except Exception as exc:
                print(f"[ERROR] No se pudo procesar el frame: {exc}")
                cv2.imshow("Tablero detectado", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            cv2.putText(tablero_anotado, f"FEN: {fen_actual[:30]}...",
                        (10, 790), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.imshow("Tablero detectado", tablero_anotado)
            tecla = cv2.waitKey(100 if args.source != "image" else 1) & 0xFF

            if tecla == ord("q"):
                break
            if tecla == 32:
                movimiento = deducir_jugada(fen_confirmado, fen_actual)
                if movimiento[0] is not None:
                    move, uci, san = movimiento
                    print(f"Jugada detectada: {uci} / {san}")
                    fen_confirmado = fen_actual
                else:
                    print(f"[AVISO] {movimiento[2]}")

                tablero_siguiente = chess.Board(fen_confirmado)
                resultado = evaluar_fen(tablero_siguiente.fen(), STOCKFISH_TIME_SECONDS)
                if resultado["move"] is None:
                    print("Stockfish no encontro una jugada.")
                else:
                    mejor = resultado["move"]
                    san_mejor = tablero_siguiente.san(mejor)
                    print(f"Mejor jugada Stockfish: {mejor.uci()} / {san_mejor}")
                    print(f"Evaluacion en centipeones: {resultado['centipawns']}")

            if args.source == "image" and tecla != 32:
                # Mantener la imagen abierta hasta q o espacio
                continue

    finally:
        if captura is not None:
            captura.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
