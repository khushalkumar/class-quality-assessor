"""Stability: re-run only the scoring stage N times per class and measure rating variance / flip rate.

Each repeat uses a fresh checkpoint store, so the scoring call is not served from cache; the transcript is the
provided Zoom .vtt (no Gemini transcription calls) and the ffmpeg stages are free. Cost: N scoring calls per class.
Raw reports go to eval/runs/stability/ (gitignored); only an anonymised summary is written to
eval/frozen/stability.json and printed.

Usage: python eval/stability.py --data "/path/to/class recordings" --only AV-1,PO-2,EX-1 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import samples  # noqa: E402

from classqa import llm as llm_mod  # noqa: E402
from classqa import rubric, worker  # noqa: E402
from classqa.llm import LLMClient  # noqa: E402
from classqa.store import JobStatus, SQLiteStore  # noqa: E402

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--only", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--vary-seed", action="store_true",
                    help="use a different sampling seed per repeat (default: the production fixed seed)")
    ap.add_argument("--out", default="stability.json")
    ap.add_argument("--quota-db", default=str(Path(tempfile.gettempdir()) / "classqa_quota.sqlite"))
    args = ap.parse_args()
    only = args.only.split(",")
    llm = LLMClient(quota=SQLiteStore(args.quota_db))   # shared daily request guard across repeats
    raw = HERE / "runs" / "stability"
    raw.mkdir(parents=True, exist_ok=True)
    out = {"repeats": args.repeats, "seed": "varied" if args.vary_seed else f"fixed {llm_mod.SEED}",
           "classes": []}
    base_seed = llm_mod.SEED
    for sid, label, mp4, vtt in samples(Path(args.data)):
        if sid not in only:
            continue
        vtt_text = vtt.read_text(encoding="utf-8", errors="ignore")
        reps = []
        for k in range(args.repeats):
            if args.vary_seed:
                llm_mod.SEED = base_seed + 101 * (k + 1)
            with tempfile.TemporaryDirectory() as td:
                runner = worker.JobRunner(store=SQLiteStore(Path(td) / "ck.sqlite"), llm=llm, start=False)
                job = worker.run_sync(mp4, vtt_text, runner=runner)
            if job.status == JobStatus.FAILED:
                print(f"{sid} rep {k}: FAILED {job.error_code}", flush=True)
                reps.append({"failed": job.error_code})
                continue
            r = job.result
            (raw / f"{sid}_{Path(args.out).stem}_rep{k}.json").write_text(json.dumps(r, indent=1))
            reps.append({"rating": r["rating"], "weighted_score": r["weighted_score"],
                         "holistic": r["holistic_rating"], "capped": r["capped_by_red_flag"],
                         "needs_review": r["needs_review"], "model": r["model"],
                         "dimension_scores": r["dimension_scores"],
                         "evidence_check": {k2: r["evidence_check"][k2] for k2 in
                                            ("exact", "approximate", "unverified", "nonverbal")}})
            print(sid, k, reps[-1]["rating"], reps[-1]["weighted_score"], reps[-1]["model"], flush=True)
        out["classes"].append({"id": sid, "label": label, "runs": reps})
    (HERE / "frozen" / args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(summarise(out), indent=1))


def summarise(out: dict) -> dict:
    idx = rubric.RATINGS.index
    per, flips, pairs = [], 0, 0
    dim_sd = {k: [] for k in rubric.WEIGHTS}
    for c in out["classes"]:
        ok = [r for r in c["runs"] if "rating" in r]
        rs = [r["rating"] for r in ok]
        ws = [r["weighted_score"] for r in ok]
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                pairs += 1
                flips += rs[i] != rs[j]
        for k in dim_sd:
            v = [r["dimension_scores"][k] for r in ok]
            if len(v) > 1:
                dim_sd[k].append(statistics.pstdev(v))
        per.append({"id": c["id"], "label": c["label"], "ratings": rs,
                    "score_mean": round(statistics.mean(ws), 3) if ws else None,
                    "score_sd": round(statistics.pstdev(ws), 3) if len(ws) > 1 else None,
                    "rating_index_var": round(statistics.pvariance([idx(r) for r in rs]), 3) if len(rs) > 1 else None,
                    "modal_agreement": (max(rs.count(x) for x in set(rs)) / len(rs)) if rs else None,
                    "any_flip": len(set(rs)) > 1, "failed": len(c["runs"]) - len(ok)})
    return {"per_class": per, "pairwise_flip_rate": round(flips / pairs, 3) if pairs else None,
            "classes_with_any_flip": sum(p["any_flip"] for p in per), "n_classes": len(per),
            "mean_dimension_sd": {k: round(statistics.mean(v), 3) for k, v in dim_sd.items() if v}}


if __name__ == "__main__":
    main()
