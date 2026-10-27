"""Advanced offline evaluation (no API calls).

  build   : read the saved model outputs (eval/runs/<tag>, gitignored) and the provided .vtt transcripts, and write
            an ANONYMISED frozen summary to eval/frozen/<tag>.json (ids, labels, dimension scores, cap flag,
            holistic rating, quote-verification counts - no text, no names). Needs the confidential data locally.
  report  : compute agreement metrics from frozen files only (this is what the CI regression gate re-uses):
            quadratic weighted kappa + confusion matrix (in-sample and leave-one-out thresholds), per-dimension
            analysis, cross-model dimension consistency, stability summary.

Usage:
  python eval/advanced_eval.py build --data "/path/to/class recordings" --tag v4-l35 [--tag v4lite]
  python eval/advanced_eval.py report [--md eval/results_advanced.md]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classqa import evalmetrics as em  # noqa: E402
from classqa import quality, rubric  # noqa: E402

HERE = Path(__file__).parent
FROZEN = HERE / "frozen"
DIMS = list(rubric.WEIGHTS)
SHORT = {"rapport_opening": "rapport", "individual_engagement": "engagement", "teaching_clarity": "clarity",
         "energy_warmth": "energy", "structure_closure": "closure", "professional_setup": "setup"}


def build(data: Path, tag: str) -> None:
    from run_eval import samples

    from classqa.transcript import from_vtt_text, render
    rows = []
    for sid, label, _mp4, vtt in samples(data):
        f = HERE / "runs" / tag / f"{sid}_vtt.json"
        if not f.exists():
            continue
        rep = json.loads(f.read_text())
        a = rep["assessment"]
        qs = {"exact": 0, "approximate": 0, "unverified": 0, "nonverbal": 0}
        if vtt:
            tr = from_vtt_text(vtt.read_text(encoding="utf-8", errors="ignore"))
            st = quality.verify_evidence(json.loads(json.dumps(a)), render(tr, max_chars=10 ** 9))
            qs = {k: st[k] for k in qs}
        rows.append({"id": sid, "actual": label, "dimension_scores": rep["dimension_scores"],
                     "capped": bool(rep.get("capped_by_red_flag")), "holistic": rep["holistic_rating"],
                     "confidence": rep["confidence"],
                     "major_flags": sum(1 for x in a.get("red_flags", []) if x.get("severity") == "major"),
                     "quotes": qs, "model": rep.get("model"), "prompt_version": rep.get("prompt_version")})
    FROZEN.mkdir(exist_ok=True)
    (FROZEN / f"{tag}.json").write_text(json.dumps({"tag": tag, "mode": "vtt", "classes": rows}, indent=1) + "\n")
    print(f"wrote {FROZEN / f'{tag}.json'} ({len(rows)} classes)")


def load(tag: str) -> list[dict]:
    return json.loads((FROZEN / f"{tag}.json").read_text())["classes"]


def labelled_rows(classes: list[dict]) -> list[dict]:
    """Rows for the label metrics, scored with the CURRENT rubric weights/thresholds/cap (so a rubric change that
    hurts agreement is caught by the regression gate)."""
    return [{"id": c["id"], "actual": c["actual"], "capped": c["capped"],
             "score": rubric.weighted_score(c["dimension_scores"]), "dims": c["dimension_scores"],
             "holistic": c["holistic"]} for c in classes if c["actual"] in rubric.RATINGS]


def quote_rate(classes: list[dict]) -> dict:
    t = {k: sum(c["quotes"][k] for c in classes) for k in ("exact", "approximate", "unverified", "nonverbal")}
    verbal = t["exact"] + t["approximate"] + t["unverified"]
    t["verified_rate"] = round((t["exact"] + t["approximate"]) / verbal, 3) if verbal else None
    return t


def per_dimension(rows: list[dict]) -> dict:
    lab = [rubric.RATINGS.index(r["actual"]) for r in rows]
    hol = [rubric.RATINGS.index(r["holistic"]) for r in rows]
    out = {}
    for d in DIMS:
        x = [r["dims"][d] for r in rows]
        by = {lv: round(statistics.mean([r["dims"][d] for r in rows if r["actual"] == lv]), 2)
              for lv in rubric.RATINGS if any(r["actual"] == lv for r in rows)}
        out[d] = {"spearman_vs_label": round(em.spearman(x, lab), 2),
                  "spearman_vs_holistic": round(em.spearman(x, hol), 2),
                  "mean_by_label": by, "range": [min(x), max(x)]}
    return out


def cross_model(a: list[dict], b: list[dict]) -> dict:
    """Same prompt, different model: per-dimension mean absolute difference and exact agreement."""
    bm = {c["id"]: c for c in b}
    shared = [c for c in a if c["id"] in bm]
    out = {"n": len(shared)}
    for d in DIMS:
        diffs = [abs(c["dimension_scores"][d] - bm[c["id"]]["dimension_scores"][d]) for c in shared]
        out[d] = {"mad": round(statistics.mean(diffs), 2), "exact": sum(x == 0 for x in diffs),
                  "within_one": sum(x <= 1 for x in diffs)}
    return out


def compute(tag: str = "v4-l35", other: str | None = "v4lite") -> dict:
    classes = load(tag)
    rows = labelled_rows(classes)
    res = {"tag": tag, "labels": [r["actual"] for r in rows], "ids": [r["id"] for r in rows],
           **em.summary(rows), "quotes": quote_rate(classes), "per_dimension": per_dimension(rows)}
    hol = [r["holistic"] for r in rows]
    act = [r["actual"] for r in rows]
    res["holistic_only"] = {"exact": em.exact(act, hol), "within_one": em.within_one(act, hol),
                            "qwk": round(em.quadratic_kappa(act, hol), 3)}
    if other and (FROZEN / f"{other}.json").exists():
        res["cross_model"] = {"other": other, **cross_model(classes, load(other))}
    st = FROZEN / "stability.json"
    if st.exists():
        sys.path.insert(0, str(HERE))
        from stability import summarise
        res["stability"] = summarise(json.loads(st.read_text()))
    return res


def to_md(res: dict) -> str:
    L = []
    lab = rubric.RATINGS
    for key, name in (("in_sample", "In-sample (frozen thresholds)"), ("leave_one_out", "Leave-one-out thresholds")):
        s = res[key]
        L += [f"**{name}:** exact {s['exact']}/{res['n']}, within one {s['within_one']}/{res['n']}, "
              f"quadratic weighted kappa **{s['qwk']}**, mean abs error {s['mae_levels']} levels", "",
              "| actual \\ predicted | " + " | ".join(lab) + " |", "|---|" + "---|" * len(lab)]
        L += [f"| {a} | " + " | ".join(str(v) for v in row) + " |" for a, row in zip(lab, s["confusion"], strict=True)]
        L.append("")
    h = res["holistic_only"]
    L += [f"Model's holistic label alone: exact {h['exact']}/{res['n']}, within one {h['within_one']}/{res['n']}, "
          f"kappa {h['qwk']}.", ""]
    q = res["quotes"]
    L += [f"Quotes: {q['exact']} exact, {q['approximate']} approximate, {q['unverified']} unverified, "
          f"{q['nonverbal']} non-verbal; verified rate {q['verified_rate']}.", ""]
    L += ["| dimension | Spearman vs label | Spearman vs holistic | mean (Poor/Avg/Good/Exc) |", "|---|---|---|---|"]
    for d, v in res["per_dimension"].items():
        m = "/".join(str(v["mean_by_label"].get(x, "-")) for x in lab)
        L.append(f"| {SHORT[d]} | {v['spearman_vs_label']} | {v['spearman_vs_holistic']} | {m} |")
    if "cross_model" in res:
        c = res["cross_model"]
        L += ["", f"Cross-model dimension consistency ({res['tag']} vs {c['other']}, same prompt, n={c['n']}):", "",
              "| dimension | mean abs diff | exact | within one |", "|---|---|---|---|"]
        L += [f"| {SHORT[d]} | {c[d]['mad']} | {c[d]['exact']}/{c['n']} | {c[d]['within_one']}/{c['n']} |"
              for d in DIMS]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", required=True)
    b.add_argument("--tag", action="append", required=True)
    r = sub.add_parser("report")
    r.add_argument("--md")
    args = ap.parse_args()
    if args.cmd == "build":
        for t in args.tag:
            build(Path(args.data), t)
    else:
        res = compute()
        print(json.dumps({k: v for k, v in res.items() if k != "per_dimension"}, indent=1))
        md = to_md(res)
        print(md)
        if args.md:
            Path(args.md).write_text(md + "\n")


if __name__ == "__main__":
    main()
