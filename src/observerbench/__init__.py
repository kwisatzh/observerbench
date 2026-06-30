"""ObserverBench reproduction CLI.

v0 is a paper reproduction workbench, not a general benchmark platform.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["__version__"]

__version__ = "0.0.0"


def _ensure_matplotlib_cache() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "observerbench-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


_ensure_matplotlib_cache()
