import json
from pathlib import Path


class JsonlMetricsLogger:
    """
    A simple JSONL (JSON Lines) metrics logger that appends metrics as JSON objects to a file.
    Each line in the file corresponds to a single metrics entry.
    Args:
        path (Path): The file path where metrics will be logged.
    """
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, metrics: dict):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
