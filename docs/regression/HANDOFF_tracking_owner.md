# MEVA 10:15 tracker-instability regression set (for the tracking owner)

**Clip:** MEVA `2018-03-09.10-15-00.10-20-00.school`, cameras G330 and G299, 30 fps. Frames are each camera's own
frame index; times = frame / 30.
**Run:** `eval_runs\meva_gym_2018-03-09_10-15\{G330,G299}\observations.jsonl` (stride 10, so switch frames are
accurate only to about 10 frames).
**Evidence:** `reports\meva_track_stability_1015\strips\<CAM>_T<id>.jpg` (crops along each track) and
`track_stability_observations.csv`.

**Scope:** 59 tracks from 30 geometry-sampled pairs in one crowded clip. This is a biased sample, **not a tracker
error rate**. Of the 59: 28 (+1 probable) contain more than one person, 14 are stable, 4 are stable but
low-resolution, and 11 are single detections or too short to judge. Findings are visual and provisional, with
MEVA boxes used only as corroboration.

**Ask:** use these as a fixed regression set: re-run the tracker and check these tracks/frames. A manual review of
all 30 pairs is not needed.

## Crossing switches (ID passes to another person when people cross)

| Camera | Track | Frames (approx. time) | What happens |
|---|---|---|---|
| G330 | T864 | 8860-8870 (295.5 s) | beige T-shirt -> blue hoodie |
| G299 | T808 | 8840-8870 (295 s) | beige T-shirt -> grey jacket, in the same crossing as G330 T864 |
| G299 | T580 / T578 | 7010-7050 (234 s) | the two tracks exchange the white-jacket and navy-hoodie people |
| G330 | T307 | 3890-3930 (130 s) | denim jacket -> long-haired person |
| G299 | T293 | 3870-3880 (129 s) | red jacket -> denim jacket |

## Chain tracks (one ID passes through 3-6 people in a cluster)

| Camera | Track | Frames (approx. time) | What happens |
|---|---|---|---|
| G330 | T518 | 5800-6430 (193-214 s) | at least 6 people |
| G330 | T539 | 6060-6560 (202-219 s) | switches ~6240, 6330, 6360, 6390 |
| G330 | T769 | 7920-9000 (264-300 s) | blue tracksuit -> navy hoodie/cap (~8140) -> red top/camo (~8250-8300) -> other (~8870) |
| G330 | T375 | 4490-5270 (150-176 s) | at least 5 people |
| G330 | T118 | 1880-2580 (63-86 s) | 4 people (~1920, ~2220, ~2330) |
| G299 | T686 | 7710-8880 (257-296 s) | 5 people (~7770, 7890, 7950, 8410) |
| G299 | T35 | 360-1030 (12-34 s) | grey hoodie -> grey sweatsuit -> black -> grey hoodie |

## Wrong first association (the first detection is another person)

| Camera | Track | Frame | What happens |
|---|---|---|---|
| G330 | T111 | 1780 | first detection is a striped/plaid person, then brown coat 1810-2410; the track switches again ~2450 |
| G299 | T748 | 8190 | first detection is a blue hoodie, then the beige T-shirt person 8220-8550 |

## Fragmentation (one person split across several tracks)

| Person | Tracks | Notes |
|---|---|---|
| Blue tracksuit, G299 | T360 (4730-4790), T412 (5250-5660) | both tracks contain only this person (cleanest case) |
| Red top / camouflage trousers, G330 | T199 (2810-2890), T383 (4580-4670) | pure tracks; also absorbed by the chains T539, T625, T769 |
| Red top / camouflage trousers, G299 | T214, T345, T375, T686, T391 | apparently the same person, not verified |
| Navy hoodie / white cap | G330 T539, T625, T769; G299 T578, T580, T686 | |

Label-free context: ~41-42 track starts per person on screen; median track length 2.7 s.

## Single / too-short tracks (continuity not judgeable)

- **Single detection:** G330 T195 (2740), T757 (7790), T42 (610), T873 (8900); G299 T192 (2390).
- **0.7-1 s:** G330 T824 (8410-8430, occluded), T749 (7720-7750, also switches); G299 T150, T215, T728, T816.

## Stable comparison examples (should stay stable)

| Camera | Track | Frames | Person |
|---|---|---|---|
| G330 | T73 | 1180-2290 (37 s) | seated, white cap |
| G330 | T122 | 2020-~3500 | yellow-cap person (switches only ~3580) |
| G299 | T156 | 2050-2680 | yellow-cap person |
| G299 | T119 | 1570-2290 | brown coat |
| G299 | T412 | 5250-5660 | blue tracksuit, through a brief occluder at 5340 |
| G299 | T644 | 7320-7870 | striped sweater |
| G299 | T375 | 4860-6240 | stable ~46 s before switching ~6300 |

**Suggested regression check:** for each listed track, a crop strip (or MEVA-box match, if KPF ids are treated
with care) before and after a tracker change. Pass means the switch/chain is split at the listed frames, the
stable tracks stay whole, and fragments merge only when they show the same person.
