"""Checks a submission file against the format the organisers specified."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from bb3x3.data import COURT_SIZE_CM, Game, load_positions


def validate_submission(path: Path, game: Game, n_frames: int, max_off_court_cm: float = 600.0) -> list[str]:
    """Return a list of problems; empty means the file is valid."""
    problems = []
    text = path.read_text()
    first = text.split("\n", 1)[0].split(",")
    if not first[0].strip().isdigit():
        problems.append(f"first line looks like a header: {first}")
    bad_cols = [i + 1 for i, line in enumerate(text.splitlines()) if line.strip() and len(line.split(",")) != 5]
    if bad_cols:
        problems.append(f"{len(bad_cols)} lines without exactly 5 columns, e.g. line {bad_cols[0]}")

    pos = load_positions(path)
    if len(pos) == 0:
        return problems + ["no rows"]
    if pos["frame"].min() < 1 or pos["frame"].max() > n_frames:
        problems.append(f"frames outside 1..{n_frames}: {pos['frame'].min()}..{pos['frame'].max()}")

    valid = set(game.identities)
    invalid = Counter((t, j) for t, j in zip(pos["team"], pos["jersey"]) if (t, j) not in valid)
    if invalid:
        problems.append(f"identities not on the roster: {dict(invalid)}")

    per_frame = Counter(pos["frame"].tolist())
    if max(per_frame.values()) > len(valid):
        problems.append(f"frames with more than {len(valid)} rows: {sum(c > len(valid) for c in per_frame.values())}")
    dup = Counter(zip(pos["frame"].tolist(), pos["team"].tolist(), pos["jersey"].tolist()))
    n_dup = sum(c > 1 for c in dup.values())
    if n_dup:
        problems.append(f"{n_dup} (frame, identity) pairs appear more than once")

    if not (np.isfinite(pos["x"]).all() and np.isfinite(pos["y"]).all()):
        problems.append("non-finite coordinates")
    w, h = COURT_SIZE_CM
    far = (pos["x"] < -max_off_court_cm) | (pos["x"] > w + max_off_court_cm)
    far |= (pos["y"] < -max_off_court_cm) | (pos["y"] > h + max_off_court_cm)
    if far.any():
        problems.append(f"{int(far.sum())} rows more than {max_off_court_cm:.0f} cm off court (metres instead of cm?)")
    return problems
