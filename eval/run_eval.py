"""Evaluate the pipeline on the labelled 8x8 sample classes.

Usage:
  python eval/run_eval.py --data "/path/to/class recordings" --tag v3 [--mode vtt|full] [--only EX-1,PO-2]

mode=vtt  : provided Zoom transcript + frames + audio excerpts from the mp4 (saves quota)
mode=full : ignore the .vtt; Gemini transcribes the audio (true end-to-end path)

Raw outputs go to eval/runs/ (gitignored). Only an anonymised summary table
(sample id, true label, prediction, scores) is printed / written to eval/results_<tag>.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classqa import llm, pipeline, rubric  # noqa: E402

LABELS = {"excellent": ("EX", "Excellent"), "good": ("GO", "Good"), "average": ("AV", "Average"),
          "weak": ("PO", "Poor"), "quality issue": ("QI", "Poor"), "demo": ("DE", None)}


def samples(root: Path):
    for folder, (code, label) in LABELS.items():
        d = root / folder
        if not d.exists():
            continue
        mp4s, vtts = sorted(d.glob("*.mp4")), sorted(d.glob("*.vtt"))
        for i, mp4 in enumerate(mp4s, 1):
            # pair transcript by batch code / language token
            key = mp4.stem.split()[0].lower() if not mp4.stem.lower().startswith("demo") else mp4.stem.split()[-1].lower()
            vtt = next((v for v in vtts if key in v.name.lower()), None)
            yield f"{code}-{i}", label, mp4, vtt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", default=pipeline.PROMPT_VERSION)
    ap.add_argument("--mode", choices=["vtt", "full"], default="vtt")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    out_dir = Path(__file__).parent / "runs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid, label, mp4, vtt in samples(Path(args.data)):
        if only and sid not in only:
            continue
        t0 = time.time()
        vtt_text = vtt.read_text(encoding="utf-8", errors="ignore") if (vtt and args.mode == "vtt") else None
        try:
            rep = pipeline.assess(mp4, vtt_text, progress=lambda f, m: None)
        except Exception as e:  # noqa: BLE001
            print(f"{sid}: ERROR {e}", flush=True)
            continue
        (out_dir / f"{sid}_{args.mode}.json").write_text(json.dumps(rep, indent=1))
        s = rep["dimension_scores"]
        row = {"id": sid, "mode": args.mode, "actual": label or "(unlabelled)", "pred": rep["rating"],
               "holistic": rep["holistic_rating"], "score": rep["weighted_score"], "conf": rep["confidence"],
               **{k: s[k] for k in rubric.WEIGHTS}, "secs": round(time.time() - t0)}
        rows.append(row)
        print(json.dumps(row), flush=True)

    lab = [r for r in rows if r["actual"] in rubric.RATINGS]
    idx = rubric.RATINGS.index
    exact = sum(r["pred"] == r["actual"] for r in lab)
    within1 = sum(abs(idx(r["pred"]) - idx(r["actual"])) <= 1 for r in lab)
    hol = sum(r["holistic"] == r["actual"] for r in lab)
    hdr = ["id", "actual", "pred", "holistic", "score", "conf", *rubric.WEIGHTS]
    md = [f"### Run `{args.tag}` (mode={args.mode})", "",
          f"Exact: **{exact}/{len(lab)}**, within one level: **{within1}/{len(lab)}**, "
          f"holistic-only exact: {hol}/{len(lab)}", "",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in rows]
    md += ["", f"Gemini calls: {llm.STATS['calls']} (failed {llm.STATS['failed_calls']}), cache hits "
               f"{llm.STATS['cache_hits']}, tokens in/out: {llm.STATS['prompt_tokens']}/{llm.STATS['output_tokens']}"]
    text = "\n".join(md)
    (Path(__file__).parent / f"results_{args.tag}_{args.mode}.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
