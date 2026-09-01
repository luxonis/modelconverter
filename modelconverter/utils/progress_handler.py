import time
from collections.abc import Callable
from dataclasses import dataclass

from rich.progress import Progress, TaskID, TextColumn


@dataclass
class _PBState:
    use_time: bool
    total: int
    start_time: float
    reps_done: int
    task_id: TaskID


def _format_time(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"


def create_progress_handler(
    benchmark_time: int, repetitions: int
) -> tuple[Progress, Callable[[], None], Callable[[], bool]]:
    """
    Returns:
      progress: Rich Progress instance (context-manage it with `with progress:`)
      on_tick(): call once per iteration to update the bar
      should_continue(): loop guard for time/rep modes
    """
    use_time = benchmark_time > 0
    total = int(benchmark_time) if use_time else int(repetitions)

    if use_time:
        progress = Progress(TextColumn("{task.description}"))
        task_id = progress.add_task(
            f"[magenta]Time Elapsed (mm:ss) [cyan]00:00 / {_format_time(total)}",
            total=total,
        )
        state = _PBState(True, total, time.time(), 0, task_id)
    else:
        progress = Progress()
        task_id = progress.add_task("[magenta]Repetition", total=total)
        state = _PBState(False, total, time.time(), 0, task_id)

    def should_continue() -> bool:
        if state.use_time:
            return (time.time() - state.start_time) < state.total
        return state.reps_done < state.total

    def on_tick() -> None:
        if state.use_time:
            elapsed = min(time.time() - state.start_time, state.total)
            progress.update(
                state.task_id,
                completed=int(elapsed),
                description=(
                    f"[magenta]Time Elapsed (mm:ss) "
                    f"[cyan]{_format_time(elapsed)} / {_format_time(state.total)}"
                ),
            )
        else:
            state.reps_done += 1
            progress.update(state.task_id, advance=1)

    return progress, on_tick, should_continue
