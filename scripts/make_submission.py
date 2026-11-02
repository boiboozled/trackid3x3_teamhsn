"""Package challenge predictions for the results form.

    python scripts/make_submission.py --run-dir outputs/runs/<run> [--out submission.zip]

Validates <run-dir>/challenge/IMG_0114.txt and IMG_0115.txt against the organisers'
format (headerless, 5 columns, cm, roster identities, frames 1..N) and writes a ZIP
holding exactly those two files at its root, as the submission instructions require.
Prints SHA-256 checksums of the files and the ZIP for the record.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import cv2

from bb3x3.data import DATA_ROOT, challenge_video_path, load_games
from bb3x3.validate import validate_submission

VIDEOS = ("IMG_0114", "IMG_0115")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="default: <run-dir>/submission.zip")
    args = parser.parse_args()

    games = load_games(DATA_ROOT / "challenge")
    files = []
    ok = True
    for video in VIDEOS:
        path = args.run_dir / "challenge" / f"{video}.txt"
        n_frames = int(cv2.VideoCapture(str(challenge_video_path(video))).get(cv2.CAP_PROP_FRAME_COUNT))
        problems = validate_submission(path, games[video], n_frames) if path.exists() else ["missing"]
        print(f"{video}.txt: {'OK' if not problems else problems}  sha256 {sha256(path) if path.exists() else '-'}")
        ok &= not problems
        files.append(path)
    if not ok:
        sys.exit("not packaging: fix the problems above")

    out = args.out or args.run_dir / "submission.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.name)  # no folders inside the ZIP
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    print(f"wrote {out} containing {names}  sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
