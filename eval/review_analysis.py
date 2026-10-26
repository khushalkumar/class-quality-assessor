"""Offline (no API calls): apply the trust layer - quote verification and review routing - to saved
eval outputs, using the provided .vtt transcripts. Prints only anonymised aggregates.
Usage: python eval/review_analysis.py --data "/path/to/class recordings" --tag v4-l35"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import samples  # noqa: E402

from classqa import quality, rubric  # noqa: E402
from classqa.transcript import from_vtt_text, render  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--tag", default="v4-l35")
args = ap.parse_args()
rows, tot = [], {"exact": 0, "approximate": 0, "unverified": 0, "nonverbal": 0}
for sid, label, _mp4, vtt in samples(Path(args.data)):
    f = Path(__file__).parent / "runs" / args.tag / f"{sid}_vtt.json"
    if not f.exists() or not vtt:
        continue
    rep = json.loads(f.read_text())
    tr = from_vtt_text(vtt.read_text(encoding="utf-8", errors="ignore"))
    a = rep["assessment"]
    st = quality.verify_evidence(a, render(tr, max_chars=10 ** 9))
    for k in tot:
        tot[k] += st[k]
    ws = rubric.weighted_score(rep["dimension_scores"])
    pred = rubric.to_rating(ws)
    capped = bool(rep.get("capped_by_red_flag"))
    if capped:
        pred = rubric.MAJOR_FLAG_CAP
    reasons = quality.review_reasons(confidence=rep["confidence"], rating=pred, holistic=rep["holistic_rating"],
                                     weighted=ws, run_ratings=[pred], quote_stats=st, missing_signals=[],
                                     major_flags=sum(1 for x in a["red_flags"] if x["severity"] == "major"),
                                     capped=capped)
    rows.append((sid, label or "demo", pred, "yes" if reasons else "no", st["unverified"], "; ".join(reasons)))
print("| id | actual | pred | needs review | unverified quotes | reasons |\n|---|---|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(map(str, r)) + " |")
lab = [r for r in rows if r[1] in rubric.RATINGS]
wrong = [r for r in lab if r[1] != r[2]]
print(f"\nquotes: {tot}")
print(f"routed to review: {sum(r[3] == 'yes' for r in lab)}/{len(lab)}; misclassified routed: "
      f"{sum(r[3] == 'yes' for r in wrong)}/{len(wrong)}; auto-accepted correct: "
      f"{sum(r[3] == 'no' and r[1] == r[2] for r in lab)}/{sum(r[3] == 'no' for r in lab)}")
