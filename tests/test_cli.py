"""Batch entrypoint with a fake LLM (no API calls)."""
import csv

from conftest import FakeBackend

from classqa import cli, worker
from classqa.llm import LLMClient


def test_batch_folder(tmp_path, media_dir, monkeypatch):
    monkeypatch.setattr(worker, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv("CLASSQA_DB", str(tmp_path / "db.sqlite"))
    folder = tmp_path / "in"
    folder.mkdir()
    for n in ("good.mp4", "corrupt.mp4"):
        (folder / n).write_bytes((media_dir / n).read_bytes())
    orig = worker.JobRunner.__init__

    def init(self, store=None, llm=None, **kw):
        orig(self, store=store, llm=LLMClient(FakeBackend(), sleep=lambda s: None), **kw)
    monkeypatch.setattr(worker.JobRunner, "__init__", init)
    monkeypatch.setattr(cli.JobRunner, "__init__", init)
    assert cli.main(["batch", str(folder), "--out", str(tmp_path / "out")]) == 0
    rows = list(csv.DictReader(open(tmp_path / "out" / "summary.csv")))
    by = {r["source"]: r for r in rows}
    assert by["corrupt.mp4"]["status"] == "failed" and by["corrupt.mp4"]["error_code"] == "corrupt_media"
    assert by["good.mp4"]["status"] in ("done", "needs_review") and by["good.mp4"]["rating"]
