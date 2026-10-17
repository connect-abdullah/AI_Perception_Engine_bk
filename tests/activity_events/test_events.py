"""Local events: view entry/exit at the frame border, zone entry/exit with debounce, dwell."""

import json

import pytest

from activity_events.events import EventSettings, run_events, track_events
from activity_events.runfolder import load_run
from activity_events.zones import Zone, load_zones

from .conftest import W

# A zone covering the right half of the frame, in image pixels.
RIGHT = Zone("right", ((320, 0), (640, 0), (640, 480), (320, 480)), dwell_seconds=2.0)


def walk(x0, x1, n, y=100, w=50, h=150, start=0):
    step = (x1 - x0) / (n - 1)
    return [(start + i, (int(x0 + i * step), y, w, h)) for i in range(n)]


def types(events):
    return [e.event_type for e in events]


def test_zone_polygon_contains():
    assert RIGHT.contains((400, 200)) and not RIGHT.contains((100, 200))


def test_walk_through_frame_gives_view_and_zone_events(make_run):
    # foot x goes from 25 to 615: enters at the left border, crosses into RIGHT, leaves at the right border
    run = load_run(make_run({1: walk(0, W - 50, 60)}))
    ev = track_events(run, run.tracks["1"], [RIGHT])
    assert types(ev) == ["entered_view", "entered_zone", "dwell", "left_view"]
    zone_ev = ev[1]
    assert zone_ev.event_meta["zone"] == "right" and 2.0 < zone_ev.timestamp_seconds < 3.5


def test_start_or_end_away_from_border_is_not_a_view_event(make_run):
    run = load_run(make_run({1: walk(150, 250, 30)}))
    assert track_events(run, run.tracks["1"], []) == []


def test_brief_zone_blip_is_ignored(make_run):
    # stands left of the line, steps over it for 0.3 s, comes back
    obs = [(i, (200, 100, 50, 150)) for i in range(20)]
    obs += [(20 + i, (300, 100, 50, 150)) for i in range(3)]  # foot x = 325: inside RIGHT
    obs += [(23 + i, (200, 100, 50, 150)) for i in range(20)]
    run = load_run(make_run({1: obs}))
    assert track_events(run, run.tracks["1"], [RIGHT]) == []


def test_exit_zone(make_run):
    run = load_run(make_run({1: walk(450, 150, 40)}))  # starts inside RIGHT, walks out
    no_dwell = Zone("right", RIGHT.polygon)
    ev = track_events(run, run.tracks["1"], [no_dwell])
    assert types(ev) == ["exited_zone"]  # started inside, so no entered_zone


def test_dwell_reports_duration(make_run):
    obs = walk(100, 400, 20) + [(20 + i, (400, 100, 50, 150)) for i in range(50)]
    run = load_run(make_run({1: obs}))
    ev = [e for e in track_events(run, run.tracks["1"], [RIGHT]) if e.event_type == "dwell"]
    assert len(ev) == 1
    meta = ev[0].event_meta
    assert meta["zone"] == "right" and meta["duration_seconds"] >= 4.0 and meta["ongoing_at_track_end"]


def test_short_fragments_produce_no_events(make_run):
    run = load_run(make_run({1: walk(0, 100, 5)}))
    assert run_events(run, [RIGHT])[0] == []


def test_no_frame_size_means_no_view_events(make_run):
    run = load_run(make_run({1: walk(0, W - 50, 60)}, size=None))
    assert "entered_view" not in types(track_events(run, run.tracks["1"], [RIGHT]))


def test_payload_matches_backend_event_schema(make_run):
    from app.entities.ingest.schema import EventIngestRequest

    run = load_run(make_run({1: walk(0, W - 50, 60)}))
    ev = track_events(run, run.tracks["1"], [RIGHT])[1]
    req = EventIngestRequest(**ev.ingest_payload())
    assert req.event_type == "entered_zone"
    assert req.event_meta["zone"] == "right" and req.event_meta["camera_code"] == "CAM-A"


def test_zone_file_is_rescaled_to_run_frame_size(tmp_path):
    p = tmp_path / "z.json"
    p.write_text(json.dumps({"camera_code": "CAM-A", "frame_size": [320, 240],
                             "zones": [{"name": "a", "polygon": [[0, 0], [160, 0], [160, 240]]}]}))
    (zone,) = load_zones(p, frame_size=(640, 480))
    assert zone.polygon[1] == (320.0, 0.0)


def test_duplicate_zone_names_rejected(tmp_path):
    p = tmp_path / "z.json"
    tri = [[0, 0], [1, 0], [0, 1]]
    p.write_text(json.dumps({"zones": [{"name": "a", "polygon": tri}, {"name": "a", "polygon": tri}]}))
    with pytest.raises(ValueError):
        load_zones(p)


def test_settings_are_respected(make_run):
    run = load_run(make_run({1: walk(0, W - 50, 60)}))
    ev = track_events(run, run.tracks["1"], [RIGHT], EventSettings(edge_margin_px=0))
    assert "entered_view" in types(ev)  # x == 0 still counts as touching the border
