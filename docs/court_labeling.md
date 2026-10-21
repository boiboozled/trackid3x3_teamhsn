# Court landmark labelling (allowed by the rules: "manual selection of fixed court landmarks")

Images: `/data/trackid3x3/plates/IMG_0114.png`, `/data/trackid3x3/plates/IMG_0115.png`
(median background plates, players removed; camera is static to < 0.5 px).

CVAT: one task with both plates, labels from `docs/cvat_court_labels.json`
(Project/Task -> Labels -> Raw -> paste). Export as **CVAT for images 1.1** to
`/data/trackid3x3/plates/court_landmarks.xml`.

## What to click (both plates)

Points, one per label, at the **centre of the white line(s)** where they meet; zoom in
(the lines are ~2 px wide). Skip a point if it is hidden; never guess. Each landmark is
its own points label (not a skeleton), and only the first click of a points shape is
read.

| # | label | where |
|---|---|---|
| 1 | c_0_0 | baseline / right sideline corner (official RKU0 is ~48 px off here) |
| 2 | c_0_1105 | near corner (closest to camera) |
| 3 | c_1505_1105 | left corner (behind the railing) |
| 4 | c_1505_0 | far corner |
| 5 | base_arc_x95 | where the right straight part of the 2-pt line meets the baseline |
| 6 | base_arc_x1410 | where the left straight part of the 2-pt line meets the baseline |
| 7 | base_key_x510 | right side of the key meets the baseline |
| 8 | base_key_x995 | left side of the key meets the baseline |
| 9 | ft_key_x510 | free-throw line / right key corner (near the "RKU" text) |
| 10 | ft_key_x995 | free-throw line / left key corner |

Polylines (8-15 clicks each, along the line centre, end to end of the visible curve):
`arc` = the curved part of the 2-pt line only (not the straight parts),
`no_charge` = the small semicircle under the basket.

See `docs/IMG_0114_template_rough.png` for where the numbers are (template projected
with a rough homography — the true positions are a few px away; click the image,
not the overlay) and `docs/court_template.png` for the top-down layout. Both are drawn as
seen from behind the top line, where the cameras stand: **x = 0 (`c_0_0`) is the
right-hand baseline corner**, next to the scoreboard. This orientation comes from the
official registration files.

Polylines are drawn in CVAT but not used yet; `scripts/register_court.py` checks the
10 points only.

## How they are used
Held-out accuracy check for the automatic (classic CV) line registration, and the
fallback homography if it fails. Not used for anything player-related.
