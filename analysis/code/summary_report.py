from __future__ import annotations

from pathlib import Path
import json
from collections import Counter


class SummaryReport:
    """Build summary statistics from a batch of signal-extraction results."""

    def load_jsonl(self, path: str | Path) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        with open(Path(path), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def build_summary(self, records: list[dict[str, object]]) -> dict[str, object]:
        signal_type_counts: Counter[str] = Counter()
        layer_counts: Counter[str] = Counter()
        error_count = 0
        total_signals = 0
        total_raw_signals = 0
        cross_layer_counts: Counter[int] = Counter()

        for r in records:
            if r.get("error"):
                error_count += 1
                continue
            total_raw_signals += int(r.get("raw_signal_count", 0))
            for sig in r.get("signals", []):
                total_signals += 1
                signal_type_counts[str(sig.get("signal_type", "unknown"))] += 1
                layer_count = int(sig.get("layer_count", 1))
                cross_layer_counts[layer_count] += 1
                for ly in sig.get("layers", []):
                    layer_counts[str(ly)] += 1
        return {
            "total_apks": len(records),
            "error_count": error_count,
            "total_signals_merged": total_signals,
            "total_signals_raw": total_raw_signals,
            "avg_signals_per_apk": (
                total_signals / max(len(records) - error_count, 1)
            ),
            "signal_type_counts": dict(signal_type_counts),
            "layer_counts": dict(layer_counts),
            "cross_layer_agreement": {
                str(k): v for k, v in sorted(cross_layer_counts.items())
            },
        }

    def write_json(self, path: str | Path, summary: dict[str, object]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    def write_markdown(self, path: str | Path, summary: dict[str, object]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Batch Summary",
            "",
            f"- Total APKs: {summary.get('total_apks', 0)}",
            f"- Errors: {summary.get('error_count', 0)}",
            f"- Total signals (merged): {summary.get('total_signals_merged', 0)}",
            f"- Total signals (raw): {summary.get('total_signals_raw', 0)}",
            f"- Avg signals per APK: {summary.get('avg_signals_per_apk', 0):.1f}",
            "",
            "## Signal Type Counts",
        ]
        for stype, count in sorted(
            summary.get("signal_type_counts", {}).items(),
            key=lambda x: -x[1],
        ):
            lines.append(f"- {stype}: {count}")

        lines.extend(["", "## Layer Distribution"])
        for layer, count in sorted(
            summary.get("layer_counts", {}).items(),
            key=lambda x: -x[1],
        ):
            lines.append(f"- {layer}: {count}")

        lines.extend(["", "## Cross-Layer Agreement"])
        for n, count in sorted(
            summary.get("cross_layer_agreement", {}).items(),
            key=lambda x: int(x[0]),
        ):
            lines.append(f"- Seen in {n} layers: {count}")

        path.write_text("\n".join(lines), encoding="utf-8")
