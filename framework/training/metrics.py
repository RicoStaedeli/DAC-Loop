from pathlib import Path
import json

class JsonlMetricsLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, metrics: dict):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")