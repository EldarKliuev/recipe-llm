from collections import deque
from statistics import median


class Metrics:
    """In-process counters for GET /metrics-lite.

    Deliberately not Prometheus: the service runs with a single worker, and core-api
    already persists per-run metadata in generation_runs.
    """

    def __init__(self, window: int = 200) -> None:
        self.queue_depth = 0
        self.in_flight = 0
        self.runs_total = 0
        self.runs_failed = 0
        self.runs_with_repair = 0
        self.repair_attempts_total = 0
        self._latencies_ms: deque[float] = deque(maxlen=window)

    def enter_queue(self) -> None:
        self.queue_depth += 1

    def leave_queue(self) -> None:
        self.queue_depth = max(0, self.queue_depth - 1)

    def start_run(self) -> None:
        self.in_flight += 1

    def finish_run(self, *, latency_ms: float, retries: int, failed: bool) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        self.runs_total += 1
        if failed:
            self.runs_failed += 1
        if retries:
            self.runs_with_repair += 1
            self.repair_attempts_total += retries
        self._latencies_ms.append(latency_ms)

    def snapshot(self) -> dict[str, object]:
        latencies = sorted(self._latencies_ms)
        return {
            "queue_depth": self.queue_depth,
            "in_flight": self.in_flight,
            "runs_total": self.runs_total,
            "runs_failed": self.runs_failed,
            "runs_with_repair": self.runs_with_repair,
            "repair_attempts_total": self.repair_attempts_total,
            "repair_rate": round(self.runs_with_repair / self.runs_total, 4)
            if self.runs_total
            else 0.0,
            "latency_ms_p50": round(median(latencies), 1) if latencies else None,
            "latency_ms_p95": round(percentile(latencies, 0.95), 1) if latencies else None,
            "latency_window": len(latencies),
        }


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight
