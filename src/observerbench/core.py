from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class AccessRegime:
    name: str
    gradients: bool = False
    hvp: bool = False
    forward_only: bool = True
    differentiable_readout: bool = False
    notes: str = ""


@dataclass
class ObserverResult:
    task: str
    observer: str
    access_regime: str
    observer_family: str
    metrics: Dict[str, float]
    known_failure_modes: List[str] = field(default_factory=list)
    recommendation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObserverCard:
    observer: str
    task: str
    access_regime: str
    estimand: str
    measurement_design: str
    validation_target: str
    metrics: Dict[str, float]
    known_failure_modes: List[str]
    recommendation: str
    notes: str = ""

    def to_markdown(self) -> str:
        lines = [
            f"# ObserverCard: `{self.observer}` on `{self.task}`",
            "",
            f"**Access regime.** {self.access_regime}",
            f"**Estimand.** {self.estimand}",
            f"**Measurement design.** {self.measurement_design}",
            f"**Validation target.** {self.validation_target}",
            "",
            "## Metrics",
        ]
        for k, v in sorted(self.metrics.items()):
            if isinstance(v, float):
                lines.append(f"- `{k}`: {v:.6g}")
            else:
                lines.append(f"- `{k}`: {v}")
        lines += ["", "## Known failure modes"]
        if self.known_failure_modes:
            lines += [f"- {m}" for m in self.known_failure_modes]
        else:
            lines.append("- None observed under this benchmark configuration.")
        lines += ["", "## Recommendation", self.recommendation]
        if self.notes:
            lines += ["", "## Notes", self.notes]
        return "\n".join(lines) + "\n"

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_markdown(), encoding="utf-8")


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
