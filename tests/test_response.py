"""The API envelope must encode datetimes/UUIDs (activity/event/history responses contain them)."""

import json
import uuid
from datetime import datetime, timezone

from app.core.response import fail, ok


def test_ok_encodes_datetime_and_uuid():
    body = json.loads(ok({"t": datetime(2026, 10, 17, 10, tzinfo=timezone.utc), "id": uuid.UUID(int=1)}).body)
    assert body["data"]["t"].startswith("2026-10-17T10:00:00") and body["data"]["id"].endswith("0001")


def test_fail_encodes_datetime():
    body = json.loads(fail("bad", data={"t": datetime(2026, 1, 1)}).body)
    assert body["ok"] is False and body["data"]["t"] == "2026-01-01T00:00:00"
