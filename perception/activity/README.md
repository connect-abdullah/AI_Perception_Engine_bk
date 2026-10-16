# Activity detection – prototype v0

Person detection + tracking + pose -> per-person state: **standing / sitting / walking**
(or **moving / still** when the legs are not visible).

- Model: YOLO11n-pose (Ultralytics), tracker: ByteTrack
- sitting vs standing: knee drop below the hips relative to torso length
- moving vs still: box speed in body-heights per second (scale-independent)
- labels are majority-voted over ~12 frames to avoid flicker
- every state change is appended to `activity_log.csv` (timestamp, person_id, from_state, to_state)

## Run
```
pip install -r requirements.txt
python activity_demo.py                      # webcam
python activity_demo.py --source video.mp4   # video file
```
Press `q` to quit. Tuning constants are at the top of `activity_demo.py`.

## Limitations (v0)
- Rule-based thresholds tuned by eye; only tested on webcam footage so far
- Sitting/standing needs the knees to be visible
- Person IDs are local to one video; global IDs (P001...) come from the identity layer
