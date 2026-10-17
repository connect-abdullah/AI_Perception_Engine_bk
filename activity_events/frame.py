"""Save one video frame with a labelled pixel grid, to read off zone polygon coordinates.

    python -m activity_events.frame VIDEO --at 12.5 --out frame_grid.png [--grid 50]

Open the PNG, note the corner points of each zone (x, y in pixels), and put them in
zones/<camera_code>.json (see zones/EXAMPLE.json).
"""

from __future__ import annotations

import argparse

import cv2


def save_grid_frame(video, at_seconds: float, out, grid: int = 50) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, at_seconds * 1000.0)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"no frame at {at_seconds}s")
    h, w = img.shape[:2]
    for x in range(0, w, grid):
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(img, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    for y in range(0, h, grid):
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(img, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.imwrite(str(out), img)
    return w, h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--at", type=float, default=0.0, help="seconds into the video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=50)
    args = ap.parse_args(argv)
    w, h = save_grid_frame(args.video, args.at, args.out, args.grid)
    print(f"saved {args.out} ({w}x{h}); use frame_size [{w}, {h}] in the zones file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
