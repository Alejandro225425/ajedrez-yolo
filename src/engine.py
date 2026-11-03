"""Interfaz con Stockfish mediante python-chess."""

from pathlib import Path

import chess
import chess.engine

try:
    from .config import STOCKFISH_PATH, STOCKFISH_TIME_SECONDS
except ImportError:  # Permite ejecutar como python src/engine.py
    from config import STOCKFISH_PATH, STOCKFISH_TIME_SECONDS


def evaluar_fen(fen: str, time: float = STOCKFISH_TIME_SECONDS, engine_path: Path = STOCKFISH_PATH):
    if not Path(engine_path).exists():
        raise FileNotFoundError(
            f"No se encontro Stockfish en {engine_path}. "
            "Descarga Stockfish y configura STOCKFISH_PATH o src/config.py."
        )
    if time <= 0:
        raise ValueError("El tiempo de analisis debe ser mayor que cero.")

    motor = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    try:
        tablero = chess.Board(fen)
        resultado = motor.play(tablero, chess.engine.Limit(time=time))
        informacion = motor.analyse(tablero, chess.engine.Limit(time=time))
        puntuacion = informacion["score"].pov(tablero.turn)
        return {
            "move": resultado.move,
            "uci": resultado.move.uci() if resultado.move else None,
            "score": puntuacion,
            "centipawns": puntuacion.score(mate_score=10000) if puntuacion else None,
        }
    finally:
        motor.quit()


def best_move(fen: str, time: float = STOCKFISH_TIME_SECONDS):
    return evaluar_fen(fen, time)["move"]
