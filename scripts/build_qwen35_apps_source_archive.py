#!/usr/bin/env python3
"""Build the source-sealed Qwen3.5 APPS registration archive.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
import tarfile
from typing import Sequence

from observerbench.provenance import file_sha256
from observerbench.tasks.qwen35_apps.registration import (
    load_qwen35_apps_config,
    require_resolved_revisions,
    verify_qwen35_apps_source_manifest,
)


ARCHIVE_SCHEMA = "observerbench.controlarena_apps_qwen35.source_archive.v0"
DEFAULT_CONFIG = Path(
    "configs/revision/ai_control/controlarena_apps_qwen3_5_9b_v0.json"
)


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive path: {value}")
    return path


def build_archive(
    repo: Path,
    output: Path,
    config_path: Path | None = None,
) -> Path:
    """Verify the seal, then write a deterministic archive and sidecar manifest."""

    repo = repo.resolve()
    config_path = (
        (repo / DEFAULT_CONFIG).resolve()
        if config_path is None
        else config_path.resolve()
    )
    config = load_qwen35_apps_config(config_path)
    require_resolved_revisions(config)
    source_manifest_path = (repo / str(config["seal"]["manifest"])).resolve()
    source_manifest = verify_qwen35_apps_source_manifest(
        config_path, source_manifest_path
    )
    source_hashes = source_manifest.get("artifact_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Qwen3.5 APPS source manifest has no artifact hashes")

    names = tuple(sorted({*map(str, source_hashes), str(config["seal"]["manifest"])}))
    files = {name: repo / _safe_relative_path(name) for name in names}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"archive allowlist file is missing: {name}")
        expected = source_hashes.get(name)
        if expected is not None and file_sha256(path) != str(expected):
            raise ValueError(f"source-seal mismatch: {name}")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as bundle:
                for name, path in files.items():
                    data = path.read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o755 if data.startswith(b"#!") else 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    bundle.addfile(info, io.BytesIO(data))
    temporary.replace(output)

    payload = {
        "schema": ARCHIVE_SCHEMA,
        "task_id": config["task_id"],
        "archive": output.name,
        "archive_sha256": file_sha256(output),
        "source_bundle_sha256": source_manifest["source_bundle_sha256"],
        "config_sha256": source_manifest["config_sha256"],
        "model_revision": source_manifest["model_revision"],
        "sae_revision": source_manifest["sae_revision"],
        "files": {name: file_sha256(path) for name, path in files.items()},
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_archive(args.repo, args.output, args.config)
    print(manifest.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
