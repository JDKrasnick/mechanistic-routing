"""Intervention dataset: append-only JSONL of state-candidate outcome records."""
import json
from pathlib import Path
from typing import Dict, Iterator, List

import pandas as pd


class InterventionDataset:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def extend(self, records: List[Dict]):
        for r in records:
            self.append(r)

    def __iter__(self) -> Iterator[Dict]:
        if not self.path.exists():
            return iter([])
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self))

    def n_states(self) -> int:
        return len({r["state_id"] for r in self})
