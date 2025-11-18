"""Common utility helpers used across the project."""

from __future__ import annotations

from collections.abc import Sized
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = ["track"]


def track[TrackItem](
    iterable: Iterable[TrackItem],
    *,
    description: str,
    context: str,
) -> Iterator[TrackItem]:
    """
    Yield the items of ``iterable`` while displaying a Rich progress bar.

    Parameters
    ----------
    iterable:
        Collection of items to iterate through.
    description:
        Human-readable description that is shown above the progress bar.
    context:
        Name of the subsystem running the task (e.g., ``"Preprocessor"``).

    Yields
    ------
    ReturnType
        The next item from ``iterable``.
    """
    console = Console()
    console.print(
        f"[bold yellow]--{context}--[/bold yellow]\n"
        f"[bold blue]{description}[/bold blue]",
    )

    total: int | None = len(iterable) if isinstance(iterable, Sized) else None

    with Progress(
        TextColumn(text_format=""),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task(description=description, total=total)
        for item in iterable:
            yield item
            progress.advance(task_id=task_id)
