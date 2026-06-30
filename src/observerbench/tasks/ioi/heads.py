"""Canonical IOI head groups used by the ObserverBench paper tasks."""

from __future__ import annotations

from typing import Iterable

Head = tuple[int, int]

NAME_MOVERS: tuple[Head, ...] = ((9, 9), (9, 6), (10, 0))
BACKUP_NAME_MOVERS: tuple[Head, ...] = (
    (9, 0),
    (9, 7),
    (10, 1),
    (10, 2),
    (10, 6),
    (10, 10),
    (11, 2),
    (11, 9),
)
NEGATIVE_NAME_MOVERS: tuple[Head, ...] = ((10, 7), (11, 10))

GROUPS: dict[str, tuple[Head, ...]] = {
    "P": NAME_MOVERS,
    "B": BACKUP_NAME_MOVERS,
    "E": NEGATIVE_NAME_MOVERS,
}


def format_head(head: Head) -> str:
    return f"{head[0]}.{head[1]}"


def format_heads(heads: Iterable[Head]) -> str:
    return ",".join(format_head(head) for head in heads)


def head_records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    idx = 0
    for group, heads in GROUPS.items():
        for layer, head in heads:
            rows.append(
                {
                    "head_idx": idx,
                    "group": group,
                    "layer": layer,
                    "head": head,
                    "label": f"{group}:{layer}.{head}",
                }
            )
            idx += 1
    return rows
