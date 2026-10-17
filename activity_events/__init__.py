"""Activity/state detection and local events on top of perception run folders.

Input is any run directory that follows docs/reid_handoff_contract.md
(manifest.json + observations.jsonl + crops/). Output is two JSON-lines files whose
records match the backend's /api/v1/ingest/activity and /api/v1/ingest/event bodies.
See docs/activity_event_contract.md.

Independent of reid/ and app/: this package only reads run folders and writes files
(and, optionally, POSTs them to the backend over HTTP).
"""
