"""Debug-warning collector.

Core principle from the spec: if a feature cannot be detected reliably, surface
an explicit warning instead of pretending it worked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Warn:
    stage: str
    code: str
    message: str
    severity: str = "warn"  # info | warn | error


@dataclass
class WarningCollector:
    items: List[Warn] = field(default_factory=list)

    def info(self, stage: str, code: str, message: str) -> None:
        self.items.append(Warn(stage, code, message, "info"))

    def warn(self, stage: str, code: str, message: str) -> None:
        self.items.append(Warn(stage, code, message, "warn"))

    def error(self, stage: str, code: str, message: str) -> None:
        self.items.append(Warn(stage, code, message, "error"))

    def has_errors(self) -> bool:
        return any(w.severity == "error" for w in self.items)

    def as_list(self) -> List[dict]:
        return [
            {"stage": w.stage, "code": w.code, "message": w.message, "severity": w.severity}
            for w in self.items
        ]
