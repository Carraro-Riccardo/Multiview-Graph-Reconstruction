"""Ruoli strutturali minimi dei componenti usati dalla ReID."""

from __future__ import annotations

from graph_reconstruction.config import COMPONENT_INFO


ANCHOR_CODES: frozenset[int] = frozenset(
    code for code, (valence, _) in COMPONENT_INFO.items() if valence >= 3
)
TERMINAL_CODES: frozenset[int] = frozenset(
    code for code, (valence, _) in COMPONENT_INFO.items() if valence == 1
)
RARE_REDUCER_CODES: frozenset[int] = frozenset({75, 255})


def is_anchor(code: int) -> bool:
    """Indica un componente che può stabilizzare la struttura di un ramo."""
    return code in ANCHOR_CODES or code in RARE_REDUCER_CODES


def is_terminal(code: int) -> bool:
    """Indica un componente con valenza fisica unitaria."""
    return code in TERMINAL_CODES
