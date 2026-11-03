"""Deduce una jugada legal comparando dos posiciones FEN."""

import chess


def deducir_jugada(fen_anterior: str, fen_actual: str):
    """Devuelve (move, uci, san) o (None, None, mensaje) ante un fallo."""
    try:
        anterior = chess.Board(fen_anterior)
        actual = chess.Board(fen_actual)
    except ValueError as exc:
        return None, None, f"FEN invalido: {exc}"

    pieza_anterior = anterior.piece_map()
    pieza_actual = actual.piece_map()
    candidatas = []

    for movimiento in anterior.legal_moves:
        prueba = anterior.copy(stack=False)
        san = prueba.san(movimiento)
        prueba.push(movimiento)
        if prueba.piece_map() == pieza_actual:
            candidatas.append((movimiento, movimiento.uci(), san))

    # Algunos FEN capturados pueden omitir turnos/enroque; comparar solo ocupacion y piezas
    if len(candidatas) == 1:
        return candidatas[0]

    if len(candidatas) > 1:
        return None, None, (
            "Mas de una jugada legal produce la misma posicion: "
            + ", ".join(c[1] for c in candidatas)
        )

    diferencias = []
    for square in chess.SQUARES:
        antes = pieza_anterior.get(square)
        despues = pieza_actual.get(square)
        if antes != despues:
            diferencias.append(f"{chess.square_name(square)}: {antes or '-'} -> {despues or '-'}")
    detalle = "; ".join(diferencias[:12]) if diferencias else "sin diferencias de piezas"
    return None, None, (
        "No coincide ninguna jugada legal entre los dos FEN. "
        f"Diferencias: {detalle}"
    )
