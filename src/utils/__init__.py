"""Common utility helpers used across the project."""

from __future__ import annotations

import signal
from collections.abc import Iterable, Iterator, Sized
from contextlib import contextmanager
from typing import TypeVar

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

__all__ = ["message", "prompt_confirm", "spinner", "track"]

TrackItem = TypeVar("TrackItem")


def message(description: str, context: str) -> None:
    console = Console()
    console.print(
        f"[bold yellow]--{context}--[/bold yellow]\n"
        f"[bold blue]{description}[/bold blue]",
    )

    print("")


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

    print("")


@contextmanager
def spinner(
    *,
    description: str,
    context: str,
    spinner_name: str = "dots",
) -> Iterator[None]:
    """Display a simple spinner while a block of work executes."""
    console = Console()
    console.print(
        f"[bold yellow]--{context}--[/bold yellow]\n"
        f"[bold blue]{description}[/bold blue]",
    )
    with console.status(description, spinner=spinner_name) as status:
        start = console.get_time()
        yield
        end = console.get_time()
        elapsed = end - start
        if status is not None:
            status.update(
                f"[green]{elapsed:.2f}[/green]",
            )
    print("")


class _InputTimeout(Exception):
    pass


def _handle_timeout(signum, frame):
    raise _InputTimeout


def prompt_confirm(
    prompt: str, *, default: bool = False, timeout_seconds: float | None = None
) -> bool:
    console = Console()
    suffix = "[Y/n]" if default else "[y/N]"

    value = default

    while True:
        # Set up alarm if timeout is requested
        if timeout_seconds is not None:
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(int(timeout_seconds))

        try:
            response = console.input(f"[magenta]{prompt}[/magenta] {suffix} ")
            # Cancel alarm once input succeeded
            if timeout_seconds is not None:
                signal.alarm(0)
        except _InputTimeout:
            console.print(
                "[magenta]Timed out waiting for input,"
                f"defaulting to {default}.[/magenta]"
            )
            break

        response = response.strip()
        if not response:
            break
        if response[0].lower() == "y":
            value = True
            break
        if response[0].lower() == "n":
            value = False
            break
        console.print("[red]Please respond with 'y' or 'n'.[/red]")

    print("")

    return value
