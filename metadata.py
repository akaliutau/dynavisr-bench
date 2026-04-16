import hashlib
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional, Set, List, Dict, Sequence

GENERATOR_VERSION = "1.0.0"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_SHA256_FILENAME = "manifest.sha256"
DATASET_SHA256_FILENAME = "dataset.sha256"

def stable_json_dumps(data: Any, *, indent: Optional[int] = None) -> str:
    if indent is None:
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=indent)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    path = Path(path)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(package_name: str) -> Optional[str]:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _payload_manifest_entries(root: Path, exclude_names: Set[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in exclude_names):
        entries.append(
            {
                "relative_path": _relative_posix(path, root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def aggregate_payload_sha256(entries: Sequence[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["relative_path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_dataset_manifest(
    out_dir: str | Path,
    *,
    seed: int,
    num_examples: int,
    snapshot_after_bounce: int,
    require_overlap_at_snapshot: Optional[bool],
    source_paths: Optional[Sequence[str | Path]] = None,
) -> Path:
    out_dir = Path(out_dir)
    exclude_names = {MANIFEST_FILENAME, MANIFEST_SHA256_FILENAME, DATASET_SHA256_FILENAME}
    payload_entries = _payload_manifest_entries(out_dir, exclude_names=exclude_names)
    payload_sha256 = aggregate_payload_sha256(payload_entries)

    source_entries: List[Dict[str, Any]] = []
    for source_path in source_paths or ():
        source_path = Path(source_path)
        if source_path.exists():
            source_entries.append(
                {
                    "path": source_path.name,
                    "sha256": sha256_file(source_path),
                }
            )

    manifest = {
        "dataset_name": out_dir.name,
        "generator_version": GENERATOR_VERSION,
        "reproducibility": {
            "seed": seed,
            "num_examples": num_examples,
            "snapshot_after_bounce": snapshot_after_bounce,
            "require_overlap_at_snapshot": require_overlap_at_snapshot,
            "deterministic_json_serialization": True,
            "sorted_file_manifest": True,
            "requires_clean_output_dir": True,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
            "pillow_version": package_version("Pillow"),
        },
        "source_files": source_entries,
        "payload_sha256": payload_sha256,
        "payload_files": payload_entries,
    }

    manifest_path = out_dir / MANIFEST_FILENAME
    manifest_path.write_text(stable_json_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    manifest_sha256 = sha256_file(manifest_path)
    (out_dir / MANIFEST_SHA256_FILENAME).write_text(
        f"{manifest_sha256}  {MANIFEST_FILENAME}\n",
        encoding="utf-8",
    )
    (out_dir / DATASET_SHA256_FILENAME).write_text(
        f"{payload_sha256}  {out_dir.name}\n",
        encoding="utf-8",
    )

    return manifest_path