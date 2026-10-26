"""Command-line / batch entrypoint - the "8x8 at scale" path.

  python -m classqa.cli assess recording.mp4 [--vtt transcript.vtt]
  python -m classqa.cli batch /path/to/folder   --out results/  [--workers 2]
  python -m classqa.cli batch links.txt         --out results/        # one Drive/URL link per line

Batch runs reuse the same worker, checkpoints and guards as the app, so an interrupted batch can simply be
re-run: finished stages (and finished classes) are not paid for twice."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import media
from .worker import JobRunner


def _sources(target: str) -> list[dict]:
    p = Path(target)
    if p.is_dir():
        out = []
        for f in sorted(x for x in p.rglob("*") if x.suffix.lower() in media.SUPPORTED_EXT):
            vtts = sorted(f.parent.glob("*.vtt"))
            key = f.stem.split()[0].lower()
            vtt = next((v for v in vtts if v.name.lower().startswith(key)), None)
            out.append({"type": "path", "ref": str(f), "vtt": str(vtt) if vtt else None})
        return out
    if p.is_file() and p.suffix == ".txt":
        return [{"type": "link", "ref": ln.strip(), "vtt": None}
                for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if target.startswith("http"):
        return [{"type": "link", "ref": target, "vtt": None}]
    return [{"type": "path", "ref": target, "vtt": None}]


def _submit(runner: JobRunner, s: dict, opts: dict):
    vtt = Path(s["vtt"]).read_text(encoding="utf-8", errors="ignore") if s.get("vtt") else None
    if s["type"] == "link":
        return runner.submit(source_type="link", source_ref=s["ref"], vtt_text=vtt, options=opts)
    return runner.submit(source_type="path", source_ref=Path(s["ref"]).name, media_path=Path(s["ref"]),
                         vtt_text=vtt, options=opts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="classqa")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assess")
    a.add_argument("source")
    a.add_argument("--vtt")
    a.add_argument("--self-consistency", type=int, default=1)
    b = sub.add_parser("batch")
    b.add_argument("target", help="folder of recordings, a .txt file of links, or a single link")
    b.add_argument("--out", default="batch_results")
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--self-consistency", type=int, default=1)
    args = ap.parse_args(argv)
    opts = {"self_consistency": args.self_consistency}

    if args.cmd == "assess":
        runner = JobRunner(workers=1)
        s = _sources(args.source)[0]
        if args.vtt:
            s["vtt"] = args.vtt
        job = runner.wait(_submit(runner, s, opts).id)
        print(json.dumps({"job": job.id, "status": job.status.value, "error": job.error_code,
                          "message": job.error_message, "rating": (job.result or {}).get("rating"),
                          "review_reasons": job.review_reasons, "usage": job.usage}, indent=1))
        return 0 if job.status.value != "failed" else 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = JobRunner(workers=args.workers)
    jobs = [_submit(runner, s, opts) for s in _sources(args.target)]
    print(f"submitted {len(jobs)} jobs", file=sys.stderr)
    rows = []
    with open(out / "reports.jsonl", "w") as fj:
        for j in jobs:
            j = runner.wait(j.id, timeout=6 * 3600)
            r = j.result or {}
            rows.append({"job_id": j.id, "source": j.source_ref, "status": j.status.value, "rating": r.get("rating"),
                         "score": r.get("weighted_score"), "confidence": r.get("confidence"),
                         "needs_review": r.get("needs_review"), "review_reasons": " | ".join(j.review_reasons or []),
                         "error_code": j.error_code, "error": j.error_message,
                         "est_cost_usd": (j.usage or {}).get("est_cost_usd"), "seconds": r.get("job", {}).get("seconds")})
            if r:
                fj.write(json.dumps(r) + "\n")
            print(json.dumps(rows[-1]), file=sys.stderr)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["job_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'summary.csv'} and {out / 'reports.jsonl'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
