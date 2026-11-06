from __future__ import annotations

from pathlib import Path
import argparse
import json

from apk_llm_analysis import APKAnalyzer, SignalAggregator, EXTRACTOR_VERSION
from feature_vectorizer import FEATURE_SCHEMA_VERSION
from registry import EVIDENCE_REGISTRY_VERSION, SIGNAL_SCHEMA_VERSION
from reporting import ReportWriter, split_code_scope_details
from research_evidence import ResearchEvidenceCollector, SCHEMA_VERSION
from summary_report import SummaryReport


def _research_record_schema_current(record: dict[str, object]) -> bool:
    evidence = record.get("research_evidence")
    return (
        isinstance(evidence, dict)
        and evidence.get("schema_version") == SCHEMA_VERSION
        and evidence.get("registry_version") == EVIDENCE_REGISTRY_VERSION
    )


def _research_record_complete(record: dict[str, object]) -> bool:
    evidence = record.get("research_evidence")
    return (
        _research_record_schema_current(record)
        and isinstance(evidence, dict)
        and evidence.get("status") in {"success", "partial"}
    )


def _signal_record_complete(record: dict[str, object]) -> bool:
    metadata = record.get("metadata")
    return (
        isinstance(metadata, dict)
        and metadata.get("extractor_version") == EXTRACTOR_VERSION
        and metadata.get("signal_schema_version") == SIGNAL_SCHEMA_VERSION
        and metadata.get("signal_registry_version") == EVIDENCE_REGISTRY_VERSION
        and metadata.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
        and not record.get("error")
    )


def _signal_record_present(record: dict[str, object]) -> bool:
    """Return whether a successful signal record can be reused by path.

    Batch append/resume is intentionally keyed by the APK path already present
    in ``apks.jsonl``.  Version validation remains available through
    ``_signal_record_complete`` for explicit full-batch migrations, but adding
    new APKs must not force re-analysis of existing records.
    """
    return bool(record.get("apk_path")) and not record.get("error")


def _research_record_present(record: dict[str, object]) -> bool:
    evidence = record.get("research_evidence")
    return (
        isinstance(evidence, dict)
        and evidence.get("status") in {"success", "partial"}
    )


def _research_record(
    apk_path: str,
    sha256: object,
    evidence: dict[str, object],
) -> dict[str, object]:
    return {
        "apk_path": apk_path,
        "sha256": sha256,
        "research_evidence": evidence,
    }


def main() -> int:
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Extract neutral LLM signals from APKs for downstream ML training")
    parser.add_argument("path", nargs="?", default=str(root_dir / "apks"),
                        help="APK path or directory")
    parser.add_argument("--jadx-bin", default=str(root_dir / "tool" / "jadx" / "bin" / "jadx"),
                        help="Path to jadx executable")
    parser.add_argument("--apktool-bin", default=str(root_dir / "tool" / "apktool_3.0.2.jar"),
                        help="Path to apktool jar")
    parser.add_argument("--work-dir", default=str(root_dir / "_analysis_work"),
                        help="Working directory for extracted artifacts")
    parser.add_argument("--output-dir", default=str(root_dir / "output"),
                        help="Directory for signals, code-scope details and research evidence")
    parser.add_argument(
        "--force", action="store_true",
        help="Rewrite this batch's JSONL and summary instead of resuming",
    )
    args = parser.parse_args()

    analyzer = APKAnalyzer(
        jadx_bin=args.jadx_bin,
        apktool_bin=args.apktool_bin,
        work_dir=Path(args.work_dir),
    )
    aggregator = SignalAggregator()
    writer = ReportWriter()
    summary_tool = SummaryReport()
    path = Path(args.path)
    output_dir = Path(args.output_dir)

    if path.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
        output_jsonl = output_dir / f"{path.name}.jsonl"
        output_scope_jsonl = output_dir / f"{path.name}.code_scope.jsonl"
        output_research_jsonl = output_dir / f"{path.name}.research_evidence.jsonl"
        output_summary = output_dir / f"{path.name}.summary.json"
        apk_paths = sorted(path.rglob("*.apk"))

        # Signals and post-classification research evidence have independent
        # resume state.  An existing signal record is never rewritten merely
        # because its research record is absent.
        signal_done: set[str] = set()
        signal_records: dict[str, dict[str, object]] = {}
        records: list[dict[str, object]] = []
        if output_jsonl.exists() and not args.force:
            with open(output_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ap = rec.get("apk_path")
                    if ap and _signal_record_present(rec):
                        apk_key = str(ap)
                        signal_done.add(apk_key)
                        signal_records[apk_key] = rec
                    records.append(rec)

        research_done: set[str] = set()
        if output_research_jsonl.exists() and not args.force:
            with open(output_research_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ap = rec.get("apk_path")
                    if ap and _research_record_present(rec):
                        research_done.add(str(ap))

        total = len(apk_paths)
        remaining = sum(
            1 for p in apk_paths
            if str(p) not in signal_done or str(p) not in research_done
        )
        if remaining == 0:
            print(f"[{total}/{total}] all done — nothing to do")
            summary = summary_tool.build_summary(records)
            summary_tool.write_json(output_summary, summary)
            return 0
        print(f"[{total - remaining}/{total}] fully cached, {remaining} remaining")

        write_mode = "w" if args.force else "a"
        with (
            open(output_jsonl, write_mode, encoding="utf-8") as jsonl_f,
            open(output_scope_jsonl, write_mode, encoding="utf-8") as scope_jsonl_f,
            open(output_research_jsonl, write_mode, encoding="utf-8") as research_jsonl_f,
        ):
            for idx, apk_path in enumerate(apk_paths, start=1):
                apk_key = str(apk_path)
                needs_signals = apk_key not in signal_done
                needs_research = apk_key not in research_done
                if not needs_signals and not needs_research:
                    continue

                try:
                    if needs_signals:
                        print(f"[{idx}/{total}] analyzing {apk_path}")
                        result = analyzer.analyze(apk_path)
                        merged = aggregator.merge(result.signals)
                        payload: dict[str, object] = {
                            "apk_path": result.apk_path,
                            "sha256": result.sha256,
                            "metadata": result.metadata,
                            "signals": merged,
                            "signal_count": len(merged),
                            "raw_signal_count": len(result.signals),
                        }
                        payload, scope_payload = split_code_scope_details(
                            payload, output_scope_jsonl)
                        # Persist the diagnostic lists before the main record
                        # that points to them. Resume still keys on signals.
                        if scope_payload is not None:
                            scope_jsonl_f.write(
                                json.dumps(scope_payload, ensure_ascii=False) + "\n")
                            scope_jsonl_f.flush()
                        jsonl_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        jsonl_f.flush()
                        records.append(payload)
                        signal_records[apk_key] = payload
                        signal_done.add(apk_key)

                        research_payload = _research_record(
                            result.apk_path,
                            result.sha256,
                            result.research_evidence,
                        )
                        research_jsonl_f.write(
                            json.dumps(research_payload, ensure_ascii=False) + "\n")
                        research_jsonl_f.flush()
                        if _research_record_complete(research_payload):
                            research_done.add(apk_key)
                        print(
                            f"[{idx}/{total}] done {apk_path}  "
                            f"({len(merged)} signals, "
                            f"zip={result.metadata.get('zip_analysis', {}).get('status', 'unknown')}, "
                            f"research={result.research_evidence.get('status')})"
                        )
                    else:
                        # Backfill only the sidecar.  Existing signals and
                        # metadata remain exactly as they were serialized.
                        print(f"[{idx}/{total}] collecting research evidence {apk_path}")
                        signal_record = signal_records[apk_key]
                        embedded = signal_record.get("research_evidence")
                        if isinstance(embedded, dict) and _research_record_complete(
                            {"research_evidence": embedded}
                        ):
                            evidence = embedded
                        else:
                            apktool_dir = analyzer.work_dir / "apktool" / apk_path.stem
                            jadx_dir = analyzer.jadx_output_dir or (
                                analyzer.work_dir / "jadx" / apk_path.stem
                            )
                            evidence = ResearchEvidenceCollector().collect(
                                apk_path,
                                signal_record.get("signals", []),
                                apktool_dir=apktool_dir if apktool_dir.exists() else None,
                                jadx_dir=jadx_dir if jadx_dir.exists() else None,
                            )
                        research_payload = _research_record(
                            apk_key,
                            signal_record.get("sha256"),
                            evidence,
                        )
                        research_jsonl_f.write(
                            json.dumps(research_payload, ensure_ascii=False) + "\n")
                        research_jsonl_f.flush()
                        if _research_record_complete(research_payload):
                            research_done.add(apk_key)
                        print(
                            f"[{idx}/{total}] research done {apk_path}  "
                            f"(status={evidence.get('status')})"
                        )
                except Exception as exc:
                    if needs_signals:
                        error_record = {"apk_path": apk_key, "error": str(exc)}
                        records.append(error_record)
                        jsonl_f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                        jsonl_f.flush()
                    else:
                        research_error = _research_record(
                            apk_key,
                            signal_records[apk_key].get("sha256"),
                            {
                                "schema_version": SCHEMA_VERSION,
                                "registry_version": EVIDENCE_REGISTRY_VERSION,
                                "status": "failed",
                                "error": type(exc).__name__,
                            },
                        )
                        research_jsonl_f.write(
                            json.dumps(research_error, ensure_ascii=False) + "\n")
                        research_jsonl_f.flush()
                    print(f"[{idx}/{total}] failed {apk_path}: {exc}")

        summary = summary_tool.build_summary(records)
        summary_tool.write_json(output_summary, summary)
        print(json.dumps({
            "status": "ok", "records": len(records),
            "signals_output": str(output_jsonl),
            "code_scope_output": str(output_scope_jsonl),
            "research_output": str(output_research_jsonl),
            "summary": str(output_summary),
        }, ensure_ascii=False, indent=2))
        return 0

    # Single APK mode
    result = analyzer.analyze(path)
    merged = aggregator.merge(result.signals)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "apk_path": result.apk_path,
        "sha256": result.sha256,
        "metadata": result.metadata,
        "signals": merged,
        "signal_count": len(merged),
        "raw_signal_count": len(result.signals),
    }
    research_payload = _research_record(
        result.apk_path,
        result.sha256,
        result.research_evidence,
    )
    signals_path = output_dir / f"{path.stem}.signals.json"
    scope_path = output_dir / f"{path.stem}.code_scope.json"
    research_path = output_dir / f"{path.stem}.research_evidence.json"
    payload, scope_payload = split_code_scope_details(payload, scope_path)
    if scope_payload is not None:
        writer.write_json(scope_path, scope_payload)
    writer.write_json(signals_path, payload)
    writer.write_json(research_path, research_payload)
    print(json.dumps({
        "status": "ok",
        "apk_path": result.apk_path,
        "signal_count": len(merged),
        "signals_output": str(signals_path),
        "code_scope_output": str(scope_path) if scope_payload is not None else None,
        "research_output": str(research_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
