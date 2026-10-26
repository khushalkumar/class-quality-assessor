"""Re-apply the current aggregation (weights, thresholds, red-flag cap) to saved
run outputs without any API calls. Usage: python eval/rescore.py v4-l35 [vtt|full]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classqa import rubric  # noqa: E402

LAB = {"EX": "Excellent", "GO": "Good", "AV": "Average", "PO": "Poor", "QI": "Poor"}
tag, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "vtt")
rows = []
for f in sorted((Path(__file__).parent / "runs" / tag).glob(f"*_{mode}.json")):
    r = json.loads(f.read_text())
    ws = rubric.weighted_score(r["dimension_scores"])
    pred = rubric.to_rating(ws)
    if r.get("capped_by_red_flag") and rubric.RATINGS.index(pred) > rubric.RATINGS.index(rubric.MAJOR_FLAG_CAP):
        pred = rubric.MAJOR_FLAG_CAP
    sid = f.name[:4]
    rows.append((sid, LAB.get(sid[:2], "(unlabelled)"), pred, r["holistic_rating"], ws))
lab = [x for x in rows if x[1] in rubric.RATINGS]
i = rubric.RATINGS.index
print(f"exact {sum(p == a for _, a, p, *_ in lab)}/{len(lab)}, "
      f"within-one {sum(abs(i(p) - i(a)) <= 1 for _, a, p, *_ in lab)}/{len(lab)}")
print("| id | actual | pred | holistic | score |\n|---|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(map(str, r)) + " |")
