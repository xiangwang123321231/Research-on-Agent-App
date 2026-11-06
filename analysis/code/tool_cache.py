"""Content/version keyed artifact manifests with explicit tool outcomes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

SCHEMA = "tool-artifacts-v1"


def fingerprint(apk_sha: str, stage: str, launcher: str, options: tuple) -> str:
    path = Path(launcher)
    artifacts = [path]
    if stage == "jadx":
        root = path.parent.parent if path.parent.name == "bin" else path
        artifacts.extend(sorted((root / "lib").glob("jadx*.jar")))
    identities = []
    for item in artifacts:
        try:
            stat = item.stat()
            identities.append((str(item.resolve()), stat.st_size, stat.st_mtime_ns))
        except OSError:
            identities.append((str(item), None, None))
    payload = json.dumps([SCHEMA, apk_sha, stage, identities, options], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_manifest(root: Path, key: str) -> dict | None:
    try:
        data = json.loads((root / "_tool_outcome.json").read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema") == SCHEMA and data.get("cache_key") == key:
            return data
    except (OSError, ValueError, TypeError):
        pass
    return None


def write_manifest(root: Path, key: str, outcome: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    target = root / "_tool_outcome.json"
    temp = root / "_tool_outcome.json.tmp"
    temp.write_text(json.dumps({**outcome, "schema": SCHEMA, "cache_key": key}, ensure_ascii=False), encoding="utf-8")
    temp.replace(target)
