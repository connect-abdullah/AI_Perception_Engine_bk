"""CLI output files and the backend upload order/payloads (with a fake sender, no network)."""

import json

import pytest

from activity_events.post import post_results, run_payloads
from activity_events.run import process
from activity_events.runfolder import load_run

from .conftest import FakePose


def walk(n, start=0):
    return [(start + i, (i * 10, 100, 50, 150)) for i in range(n)]


@pytest.fixture
def processed(make_run, tmp_path):
    root = make_run({1: walk(60), 2: [(i, (300, 100, 50, 150)) for i in range(40)]})
    zones = tmp_path / "zones.json"
    zones.write_text(json.dumps({"zones": [{"name": "right", "polygon": [[320, 0], [640, 0], [640, 480], [320, 480]]}]}))
    summary = process(root, zones_path=zones, estimator=FakePose())
    return root, summary


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_process_writes_result_files(processed):
    root, summary = processed
    out = root / "activity_events"
    acts, evs = read_jsonl(out / "activities.jsonl"), read_jsonl(out / "events.jsonl")
    assert acts and evs
    assert {a["local_track_id"] for a in acts} == {"1", "2"}
    assert json.loads((out / "summary.json").read_text())["activity"]["intervals"] == len(acts)
    assert summary["events"]["by_type"].get("entered_zone") == 1


def test_post_sends_run_first_then_activities_then_events(processed):
    root, _ = processed
    sent = []
    counts = post_results(root, lambda path, body: sent.append((path, body)) or {}, chunk=50)
    paths = [p for p, _ in sent]
    n_run = paths.count("/api/v1/ingest/run")
    assert n_run == 2 and counts["run_chunks"] == 2  # 100 observations in chunks of 50
    assert paths[:n_run] == ["/api/v1/ingest/run"] * n_run
    first_event = paths.index("/api/v1/ingest/event")
    assert all(p == "/api/v1/ingest/activity" for p in paths[n_run:first_event])
    assert counts["activities"] + counts["events"] + n_run == len(sent)


def test_run_payload_matches_backend_schema(make_run):
    from app.entities.ingest.schema import RunIngestRequest

    run = load_run(make_run({1: walk(5)}))
    (body,) = list(run_payloads(run))
    req = RunIngestRequest(**body)
    assert len(req.observations) == 5 and req.camera_code == "CAM-A"


def test_post_refuses_without_start_time(make_run):
    root = make_run({1: walk(30)}, start_time=None)
    process(root, estimator=None)
    with pytest.raises(ValueError):
        post_results(root, lambda p, b: {})
