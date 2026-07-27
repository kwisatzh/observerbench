#!/usr/bin/env python3
"""Build the allowlisted, source-sealed Qwen Copy-v2 Colab archive.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive path: {value}")
    return path


def build_archive(repo: Path, output: Path) -> Path:
    source_manifest_path = (
        repo
        / "configs"
        / "revision"
        / "phase10"
        / "qwen_copy_v2_source_manifest.json"
    )
    config_path = (
        repo
        / "configs"
        / "revision"
        / "phase10"
        / "qwen2_5_7b_induction_copy_v2.json"
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("schema")
        != "observerbench.qwen_induction_phase10.source_manifest.v1"
        or source_manifest.get("status") != "frozen_before_copy_v2_outcomes"
        or config.get("schema") != "observerbench.qwen_induction_phase10.v1"
        or config.get("status") != "frozen_before_copy_v2_outcomes"
        or config.get("data_version") != "copy-v2"
        or json_sha256(config) != source_manifest.get("config_sha256")
    ):
        raise ValueError("source manifest and frozen config differ")
    source_hashes = source_manifest.get("artifact_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("frozen source manifest has no artifact hashes")
    normalized_source_hashes = {
        str(label): str(digest)
        for label, digest in sorted(source_hashes.items())
    }
    if source_manifest.get("source_bundle_sha256") != json_sha256(
        normalized_source_hashes
    ):
        raise ValueError("frozen source manifest has a mismatched bundle digest")
    for label, expected in normalized_source_hashes.items():
        path = repo / safe_relative_path(str(label))
        if not path.is_file() or file_sha256(path) != str(expected):
            raise ValueError(f"source-seal mismatch: {label}")

    exclusion = config.get("token_pool", {}).get(
        "exclude_copy_v1_allocated_tokens", {}
    )
    source_artifact = str(exclusion.get("source_artifact", ""))
    copy_v1_path = repo / safe_relative_path(source_artifact)
    if (
        not copy_v1_path.is_file()
        or file_sha256(copy_v1_path)
        != str(exclusion.get("source_artifact_sha256", ""))
    ):
        raise ValueError("Copy-v1 exclusion artifact differs from the frozen config")
    copy_v1_payload = json.loads(copy_v1_path.read_text(encoding="utf-8"))
    if (
        int(copy_v1_payload.get("token_pool_size", -1))
        != int(exclusion.get("source_token_pool_size", -2))
        or str(copy_v1_payload.get("token_pool_sha256", ""))
        != str(exclusion.get("source_token_pool_sha256", ""))
    ):
        raise ValueError("Copy-v1 exclusion token-pool identity changed")

    extras = {
        "README.md",
        "configs/revision/phase10/qwen2_5_7b_induction_copy_v2.json",
        "configs/revision/phase10/qwen_copy_v2_source_manifest.json",
        source_artifact,
    }
    names = tuple(sorted(set(normalized_source_hashes) | extras))
    files = {name: repo / safe_relative_path(name) for name in names}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"archive allowlist file is missing: {name}")

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
        "schema": "observerbench.qwen_induction_phase10.source_archive.v1",
        "archive": output.name,
        "archive_sha256": file_sha256(output),
        "source_bundle_sha256": source_manifest["source_bundle_sha256"],
        "config_sha256": source_manifest["config_sha256"],
        "files": {name: file_sha256(path) for name, path in files.items()},
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_archive(args.repo.resolve(), args.output.resolve())
    print(manifest.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
