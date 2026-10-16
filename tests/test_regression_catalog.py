"""Regression catalog tests – no database, no video, no tracker required.

These tests verify that the two MEVA tracking-regression files vendored at
docs/regression/ are present, well-formed, and contain the named failure and
stable-control cases described in HANDOFF_tracking_owner.md.

They do NOT score the tracker against live video.  They are a static guard that
ensures the regression set stays intact and is not accidentally modified.  When
Yahya provides the source videos and the original eval run, the separate
tracker-evaluation tests will use these cases as the expected outcome.

Run with:
    poetry run pytest tests/test_regression_catalog.py -v
"""

from __future__ import annotations

import csv
import pathlib
import re

import pytest

REGRESSION_DIR = pathlib.Path(__file__).parent.parent / "docs" / "regression"
HANDOFF_MD = REGRESSION_DIR / "HANDOFF_tracking_owner.md"
STABILITY_CSV = REGRESSION_DIR / "track_stability_observations.csv"

# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def _load_csv() -> list[dict]:
    with STABILITY_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _track_key(row: dict) -> tuple[str, str]:
    """(camera, track-number-without-T) → (camera, track-id-string)."""
    return row["camera"].strip(), row["track"].strip()


def _index_by_camera_track(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    index: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = _track_key(row)
        index.setdefault(key, []).append(row)
    return index


# ---------------------------------------------------------------------------
# File presence
# ---------------------------------------------------------------------------

class TestFilesPresent:
    def test_handoff_md_exists(self):
        assert HANDOFF_MD.exists(), (
            f"{HANDOFF_MD} not found. "
            "Copy HANDOFF_tracking_owner.md into docs/regression/."
        )

    def test_stability_csv_exists(self):
        assert STABILITY_CSV.exists(), (
            f"{STABILITY_CSV} not found. "
            "Copy track_stability_observations.csv into docs/regression/."
        )

    def test_stability_csv_is_non_empty(self):
        rows = _load_csv()
        assert len(rows) > 0, "track_stability_observations.csv contains no data rows"

    def test_stability_csv_has_required_columns(self):
        rows = _load_csv()
        required = {"pair", "camera", "track", "obs", "frames", "visual_finding",
                    "in_review_window", "class"}
        actual = set(rows[0].keys())
        missing = required - actual
        assert not missing, f"CSV is missing columns: {missing}"

    def test_stability_csv_has_expected_row_count(self):
        """Handoff document states 59 tracks from 30 geometry-sampled pairs."""
        rows = _load_csv()
        assert len(rows) == 60, (
            f"Expected 60 rows (30 pairs × 2 cameras) in track_stability_observations.csv, "
            f"got {len(rows)}. Do not modify the regression set."
        )

    def test_handoff_md_references_meva_clip(self):
        content = HANDOFF_MD.read_text(encoding="utf-8")
        assert "2018-03-09.10-15-00.10-20-00.school" in content
        assert "G330" in content
        assert "G299" in content


# ---------------------------------------------------------------------------
# Crossing-switch cases
# ---------------------------------------------------------------------------

CROSSING_SWITCHES = [
    ("G330", "864"),
    ("G299", "808"),
    ("G299", "580"),
    ("G330", "307"),
    ("G299", "293"),
]


class TestCrossingSwitches:
    @pytest.fixture(scope="class")
    def index(self):
        return _index_by_camera_track(_load_csv())

    @pytest.mark.parametrize("camera,track", CROSSING_SWITCHES)
    def test_track_present(self, index, camera, track):
        assert (camera, track) in index, (
            f"Crossing-switch track {camera} T{track} not found in CSV"
        )

    @pytest.mark.parametrize("camera,track", CROSSING_SWITCHES)
    def test_track_classified_as_switch(self, index, camera, track):
        for row in index[(camera, track)]:
            cls = row["class"].lower()
            assert "switch" in cls, (
                f"{camera} T{track}: expected class containing 'switch', got '{row['class']}'"
            )


# ---------------------------------------------------------------------------
# Chain-track cases
# ---------------------------------------------------------------------------

CHAIN_TRACKS = [
    ("G330", "518"),
    ("G330", "539"),
    ("G330", "769"),
    ("G330", "375"),
    ("G330", "118"),
    ("G299", "686"),
]


class TestChainTracks:
    @pytest.fixture(scope="class")
    def index(self):
        return _index_by_camera_track(_load_csv())

    @pytest.mark.parametrize("camera,track", CHAIN_TRACKS)
    def test_chain_track_present(self, index, camera, track):
        assert (camera, track) in index, (
            f"Chain-track {camera} T{track} not found in CSV"
        )

    @pytest.mark.parametrize("camera,track", CHAIN_TRACKS)
    def test_chain_track_classified_as_switch(self, index, camera, track):
        for row in index[(camera, track)]:
            cls = row["class"].lower()
            assert "switch" in cls, (
                f"{camera} T{track}: expected 'switch' in class, got '{row['class']}'"
            )


# ---------------------------------------------------------------------------
# Wrong-first-association cases
# ---------------------------------------------------------------------------

WRONG_FIRST = [
    ("G330", "111"),
    ("G299", "748"),
]


class TestWrongFirstAssociation:
    @pytest.fixture(scope="class")
    def index(self):
        return _index_by_camera_track(_load_csv())

    @pytest.mark.parametrize("camera,track", WRONG_FIRST)
    def test_wrong_first_track_present(self, index, camera, track):
        assert (camera, track) in index

    @pytest.mark.parametrize("camera,track", WRONG_FIRST)
    def test_wrong_first_classified(self, index, camera, track):
        for row in index[(camera, track)]:
            cls = row["class"].lower()
            assert "wrong" in cls or "switch" in cls, (
                f"{camera} T{track}: expected 'wrong' or 'switch' in class, got '{row['class']}'"
            )


# ---------------------------------------------------------------------------
# Fragmentation: blue-tracksuit in G299 (T360, T412)
# ---------------------------------------------------------------------------

FRAGMENTATION_STABLE = [
    ("G299", "360"),
    ("G299", "412"),
]


class TestFragmentationBluetTracksuit:
    """G299 T360 and T412 each show only the blue-tracksuit person throughout
    their window – they are the cleanest fragmentation example because each
    fragment is a pure track.  Both are classified 'stable' in the CSV because
    each individual track contains only one person; the fragmentation shows up
    as two separate stable tracks for the same person.
    """

    @pytest.fixture(scope="class")
    def index(self):
        return _index_by_camera_track(_load_csv())

    @pytest.mark.parametrize("camera,track", FRAGMENTATION_STABLE)
    def test_fragment_track_present(self, index, camera, track):
        assert (camera, track) in index, (
            f"Fragmentation track {camera} T{track} not found in CSV"
        )

    @pytest.mark.parametrize("camera,track", FRAGMENTATION_STABLE)
    def test_fragment_tracks_are_individually_stable(self, index, camera, track):
        """Each fragment contains only one person (stable), but they are
        two separate IDs for the same person – the fragmentation at the track
        level is shown by their disjoint frame windows."""
        for row in index[(camera, track)]:
            cls = row["class"].lower()
            assert "stable" in cls, (
                f"{camera} T{track}: expected 'stable' class, got '{row['class']}'"
            )

    def test_fragment_frame_windows_do_not_overlap(self):
        rows = _load_csv()
        index = _index_by_camera_track(rows)

        def frames(camera, track):
            for row in index[(camera, track)]:
                m = re.search(r"(\d+)-(\d+)", row["frames"])
                if m:
                    return int(m.group(1)), int(m.group(2))
            return None

        w360 = frames("G299", "360")
        w412 = frames("G299", "412")
        assert w360 is not None and w412 is not None
        # T360 ends before T412 starts (the gap is the fragmentation)
        assert w360[1] < w412[0], (
            f"Expected T360 ({w360}) to end before T412 ({w412}) begins"
        )


# ---------------------------------------------------------------------------
# Stable-control tracks (must not be broken by tracker improvements)
# ---------------------------------------------------------------------------

STABLE_CONTROLS = [
    ("G330", "73"),
    ("G299", "119"),
    ("G299", "156"),
    ("G299", "412"),
    ("G299", "644"),
]


class TestStableControls:
    """These tracks must remain stable after any tracker change.
    Regressions here (a stable track becoming a switch) indicate a new bug.
    """

    @pytest.fixture(scope="class")
    def index(self):
        return _index_by_camera_track(_load_csv())

    @pytest.mark.parametrize("camera,track", STABLE_CONTROLS)
    def test_stable_control_present(self, index, camera, track):
        assert (camera, track) in index, (
            f"Stable-control track {camera} T{track} not found in CSV"
        )

    @pytest.mark.parametrize("camera,track", STABLE_CONTROLS)
    def test_stable_control_classified_stable(self, index, camera, track):
        for row in index[(camera, track)]:
            cls = row["class"].lower()
            assert "stable" in cls, (
                f"Stable-control {camera} T{track}: "
                f"expected 'stable' in class, got '{row['class']}'. "
                "This track must not be reclassified unless the regression set is formally updated."
            )


# ---------------------------------------------------------------------------
# Short / single-detection tracks
# ---------------------------------------------------------------------------

SINGLE_DETECTIONS = [
    ("G330", "195"),
    ("G330", "757"),
    ("G330", "42"),
    ("G330", "873"),
    ("G299", "192"),
]

SHORT_AMBIGUOUS = [
    ("G330", "824"),
    ("G299", "728"),
    ("G299", "816"),
]


class TestShortAndSingleTracks:
    @pytest.fixture(scope="class")
    def index(self):
        return _index_by_camera_track(_load_csv())

    @pytest.mark.parametrize("camera,track", SINGLE_DETECTIONS)
    def test_single_detection_present(self, index, camera, track):
        assert (camera, track) in index

    @pytest.mark.parametrize("camera,track", SINGLE_DETECTIONS)
    def test_single_detection_classified(self, index, camera, track):
        for row in index[(camera, track)]:
            cls = row["class"].lower()
            assert "single" in cls or "ambiguous" in cls or "short" in cls, (
                f"{camera} T{track}: expected single/ambiguous/short class, got '{row['class']}'"
            )

    @pytest.mark.parametrize("camera,track", SHORT_AMBIGUOUS)
    def test_short_ambiguous_present(self, index, camera, track):
        assert (camera, track) in index

    @pytest.mark.parametrize("camera,track", SHORT_AMBIGUOUS)
    def test_short_ambiguous_classified(self, index, camera, track):
        for row in index[(camera, track)]:
            cls = row["class"].lower()
            assert "ambiguous" in cls or "short" in cls or "single" in cls, (
                f"{camera} T{track}: expected ambiguous/short/single class, got '{row['class']}'"
            )


# ---------------------------------------------------------------------------
# Completeness: every row that the HANDOFF calls out in a specific window
# has `in_review_window = yes` or `no` (not an unexpected value)
# ---------------------------------------------------------------------------

class TestCsvIntegrity:
    def test_all_class_values_are_known(self):
        known_stems = {
            "stable",
            "switch",
            "wrong",
            "single",
            "short",
            "ambiguous",
        }
        for row in _load_csv():
            cls = row["class"].lower()
            matched = any(stem in cls for stem in known_stems)
            assert matched, (
                f"Unexpected class value '{row['class']}' for "
                f"{row['camera']} T{row['track']}. "
                "Add it to known_stems or correct the CSV."
            )

    def test_in_review_window_values(self):
        # Valid stems: "yes ...", "no ...", "n/a", "stable ...", or empty.
        valid_starts = ("yes", "no", "n/a", "stable", "")
        for row in _load_csv():
            val = row["in_review_window"].lower().strip()
            assert any(val.startswith(s) for s in valid_starts), (
                f"Unexpected in_review_window value '{row['in_review_window']}' "
                f"for {row['camera']} T{row['track']}. "
                f"Valid starts: {valid_starts}"
            )

    def test_obs_counts_are_positive_integers(self):
        for row in _load_csv():
            try:
                n = int(row["obs"])
                assert n >= 1, f"obs must be >= 1, got {n} for {row['camera']} T{row['track']}"
            except ValueError:
                pytest.fail(
                    f"obs column is not an integer for {row['camera']} T{row['track']}: "
                    f"'{row['obs']}'"
                )
