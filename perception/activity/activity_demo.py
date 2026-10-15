"""Live activity demo: person detection + tracking + pose -> standing / sitting / walking.

Run:   python activity_demo.py                 (webcam)
       python activity_demo.py --source video.mp4
Keys:  q = quit

What happens each frame
  1. YOLO-pose finds every person: a box + 17 body keypoints (shoulders, hips, knees...).
  2. The tracker (ByteTrack) gives each person a stable ID across frames (#1, #2...).
  3. Simple rules turn keypoints + movement into a state:
       - sitting vs standing: how far the knees are below the hips, compared with
         the torso length (when you sit, your thighs go flat, so knees ~ hip height)
       - moving vs still: how fast the box moves, measured in "body heights per second"
         so it works whether you are near or far from the camera
  4. The last few labels are majority-voted so the label doesn't flicker.
  5. Every time someone's state changes it is written to activity_log.csv
     (this is a first version of the "state intervals" the project needs).
"""

import argparse
import csv
import time
from collections import Counter, defaultdict, deque
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# COCO keypoint indices
L_SHO, R_SHO, L_HIP, R_HIP, L_KNEE, R_KNEE = 5, 6, 11, 12, 13, 14

KP_CONF = 0.5          # ignore keypoints the model is unsure about
SIT_RATIO = 0.55       # knee drop / torso length below this -> sitting
MOVE_SPEED = 0.35      # body-heights per second above this -> moving
SMOOTH_FRAMES = 12     # majority vote window

COLORS = {
    "standing": (80, 200, 120),
    "sitting": (255, 170, 60),
    "walking": (60, 120, 255),
    "moving": (60, 120, 255),
    "still": (180, 180, 180),
}


def visible(kps, conf, i):
    return conf[i] >= KP_CONF


def posture(kps, conf):
    """Return 'sitting', 'standing' or None (legs not visible)."""
    ratios = []
    for sho, hip, knee in ((L_SHO, L_HIP, L_KNEE), (R_SHO, R_HIP, R_KNEE)):
        if visible(kps, conf, sho) and visible(kps, conf, hip) and visible(kps, conf, knee):
            torso = np.linalg.norm(kps[hip] - kps[sho])
            if torso < 1:
                continue
            knee_drop = kps[knee][1] - kps[hip][1]  # image y grows downwards
            ratios.append(knee_drop / torso)
    if not ratios:
        return None
    return "sitting" if np.mean(ratios) < SIT_RATIO else "standing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index (0) or a video file path")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    args = ap.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)  # downloads the weights the first time
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {args.source!r}")

    history = defaultdict(lambda: deque(maxlen=15))           # id -> (t, cx, cy, h)
    votes = defaultdict(lambda: deque(maxlen=SMOOTH_FRAMES))  # id -> recent raw labels
    current = {}                                               # id -> shown label

    log = open("activity_log.csv", "a", newline="")
    writer = csv.writer(log)
    if log.tell() == 0:
        writer.writerow(["timestamp", "person_id", "from_state", "to_state"])

    last = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()

        res = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        out = res.plot(labels=False, boxes=True)  # draws boxes + skeletons

        if res.boxes is not None and res.boxes.id is not None and res.keypoints is not None:
            ids = res.boxes.id.int().tolist()
            boxes = res.boxes.xyxy.cpu().numpy()
            kxy = res.keypoints.xy.cpu().numpy()
            kconf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kxy.shape[:2])

            for pid, (x1, y1, x2, y2), kps, conf in zip(ids, boxes, kxy, kconf):
                h = max(y2 - y1, 1.0)
                history[pid].append((now, (x1 + x2) / 2, y2, h))

                speed = 0.0
                hist = history[pid]
                if len(hist) >= 2 and hist[-1][0] - hist[0][0] > 0.2:
                    dt = hist[-1][0] - hist[0][0]
                    dist = np.hypot(hist[-1][1] - hist[0][1], hist[-1][2] - hist[0][2])
                    speed = dist / np.mean([e[3] for e in hist]) / dt
                moving = speed > MOVE_SPEED

                p = posture(kps, conf)
                if p == "sitting":
                    raw = "sitting"
                elif p == "standing":
                    raw = "walking" if moving else "standing"
                else:
                    raw = "moving" if moving else "still"
                votes[pid].append(raw)
                label = Counter(votes[pid]).most_common(1)[0][0]

                if current.get(pid) != label:
                    writer.writerow([datetime.now().isoformat(timespec="seconds"), pid,
                                     current.get(pid, ""), label])
                    log.flush()
                    current[pid] = label

                color = COLORS.get(label, (255, 255, 255))
                text = f"#{pid} {label}"
                if p is None:
                    text += " (legs hidden)"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                tx, ty = int(x1), max(int(y1) - 10, th + 10)
                cv2.rectangle(out, (tx, ty - th - 8), (tx + tw + 8, ty + 4), color, -1)
                cv2.putText(out, text, (tx + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-3))
        last = now
        cv2.putText(out, f"{fps:4.1f} FPS  |  step back so your knees are visible  |  q = quit",
                    (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow("Activity demo", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    log.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
