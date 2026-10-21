"""End-to-end TrackID3x3 pipeline: detect -> track -> project -> identity -> positions.

    python scripts/run_pipeline.py run_name=e2e_v0 [tracker=botsort] [max_frames=600]

Training videos (720p transcodes) are scored with TI-HOTA (full identity, team only,
identity-free) at each tau; challenge videos get submission files + validation.
Detections are cached per detector config; everything else is recomputed. Each
invocation is one MLflow run in experiment `trackid3x3`.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import hydra
import mlflow
import numpy as np
from handball.common.mlflow_utils import log_metrics_dict, start_run
from handball.identity.team_features import HsvTorsoFeatureExtractor
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from bb3x3.data import (
    POSITION_DTYPE,
    challenge_video_path,
    image_to_court_homography,
    load_games,
    load_positions,
    project,
    training_paths,
    write_positions,
)
from bb3x3.assign import SlotConfig, assign_slots, slots_to_positions
from bb3x3.camera import calibrate
from bb3x3.crops import padded_crop
from bb3x3.detect import UltralyticsPersonDetector, detect_video, load_detections, save_detections
from bb3x3.identity import SQUADS, assign_track_squads, fit_squad_prototypes, game_squads, placeholder_numbers
from bb3x3.player_classifier import PLAYERS, PlayerClassifier
from bb3x3.ti_hota import collapse_identity, ti_hota
from bb3x3.track import run_tracker
from bb3x3.validate import validate_submission

SCALE_720_TO_4K = np.diag([3.0, 3.0, 1.0])
INFEASIBLE_NLL = 50.0


def _key(cfg_part) -> str:
    blob = json.dumps(OmegaConf.to_container(cfg_part, resolve=True), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def video_path(cfg, split: str, video: str) -> Path:
    if split == "training":
        return Path(cfg.data_root) / "training" / "videos_720p" / f"{video}.mp4"
    return challenge_video_path(video, Path(cfg.data_root))


def homography(cfg, split: str, video: str) -> np.ndarray:
    root = Path(cfg.data_root)
    if split == "training":
        # The official training registration is in 4K pixels and defines the GT.
        return image_to_court_homography(root / "training") @ SCALE_720_TO_4K
    if cfg.registration.challenge == "official":
        return image_to_court_homography(root / "challenge")
    return np.load(Path(cfg.registration.challenge) / f"{video}.npy")


def squad_prototypes(cfg, exclude_video: str | None):
    data = np.load(cfg.identity.squad_features)
    videos = sorted({k.split("/")[0] for k in data.files})
    feats = np.concatenate([data[f"{v}/feats"] for v in videos if v != exclude_video])
    squads = np.concatenate([data[f"{v}/squads"] for v in videos if v != exclude_video])
    return fit_squad_prototypes({s: feats[squads == s] for s in np.unique(squads)}, slack=cfg.identity.prototype_slack)


def assemble(cfg, dets, feats, track_ids, H, game, protos) -> tuple[np.ndarray, dict]:
    a = cfg.assembly
    keep = (track_ids >= 0) & (dets["conf"] >= a.min_conf)
    lengths = dict(zip(*np.unique(track_ids[keep], return_counts=True)))
    keep &= np.array([lengths.get(t, 0) >= a.min_track_len for t in track_ids])

    squads_in_game = game_squads(game)  # squad -> team
    numbers_of = {s: game.roster[t] for s, t in squads_in_game.items()}
    track_squad = assign_track_squads(track_ids[keep], feats[keep], protos, cfg.identity.min_squad_share)
    numbers = placeholder_numbers(track_ids[keep], dets["frame"][keep], track_squad, numbers_of)

    foot = np.c_[(dets["x1"] + dets["x2"]) / 2, dets["y2"]]
    court = project(H, foot)
    m = a.court_margin_cm
    in_gate = (court[:, 0] > -m) & (court[:, 0] < 1505 + m) & (court[:, 1] > -m) & (court[:, 1] < 1105 + m)

    best: dict[tuple[int, str, str], tuple[float, int]] = {}
    for i in np.flatnonzero(keep & in_gate):
        tid = int(track_ids[i])
        squad = track_squad.get(tid)
        if squad not in squads_in_game or tid not in numbers:
            continue
        key = (int(dets["frame"][i]), squads_in_game[squad], numbers[tid])
        if key not in best or dets["conf"][i] > best[key][0]:
            best[key] = (float(dets["conf"][i]), i)
    rows = [(f, court[i, 0], court[i, 1], team, num) for (f, team, num), (_, i) in best.items()]
    stats = {
        "n_tracks": len(lengths),
        "n_tracks_kept": len(numbers),
        "n_tracks_off_roster_squad": sum(1 for s in track_squad.values() if s is not None and s not in squads_in_game),
        "n_tracks_no_squad": sum(1 for s in track_squad.values() if s is None),
    }
    return np.array(rows, dtype=POSITION_DTYPE), stats


def classify_detections(cfg, path: Path, dets: np.ndarray, cache: Path) -> np.ndarray:
    """(N, 12) player log-probabilities; NaN rows for detections below min_conf."""
    if cache.exists():
        return np.load(cache)["logp"]
    classifier = PlayerClassifier(cfg.classifier.checkpoint)
    logp = np.full((len(dets), len(PLAYERS)), np.nan, np.float32)
    wanted = np.flatnonzero(dets["conf"] >= cfg.assembly.min_conf)
    by_frame: dict[int, list[int]] = {}
    for i in wanted:
        by_frame.setdefault(int(dets["frame"][i]), []).append(i)
    pending_idx, pending_crops = [], []
    cap = cv2.VideoCapture(str(path))
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_id >= int(dets["frame"].max()):
            break
        frame_id += 1
        for i in by_frame.get(frame_id, []):
            d = dets[i]
            pending_crops.append(padded_crop(frame, np.array([d["x1"], d["y1"], d["x2"], d["y2"]])))
            pending_idx.append(i)
        if len(pending_crops) >= 1024:
            logp[pending_idx] = classifier.log_probs(np.stack(pending_crops))
            pending_idx, pending_crops = [], []
    if pending_crops:
        logp[pending_idx] = classifier.log_probs(np.stack(pending_crops))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, logp=logp)
    return logp


def detection_squads(cfg, dets, feats, logp, court, game, protos) -> tuple[np.ndarray, np.ndarray | None]:
    """Squad per detection ("" = not a player of this game) and (N, 3) jersey NLL within it.

    Squad scores come from the classifier when available (log-sum over the squad's
    players), else from colour prototypes. A detection whose best squad is not in
    this game is a waiting squad unless it stands on court, where only the two game
    squads play; there it takes its best in-game squad.
    """
    squads_in_game = game_squads(game)
    names = list(SQUADS)
    dists = protos.distances(feats)
    colour_score = -np.stack([dists[s] for s in names], 1)
    # The classifier always names a squad, even for a spectator: whether a detection
    # wears a bib at all is decided by colour distance in both modes.
    bib_like = (-colour_score.max(1)) <= protos.max_distance if cfg.identity.bib_filter else np.ones(len(dets), bool)
    if logp is not None and cfg.identity.squad_from == "classifier":
        players = np.array(PLAYERS)
        score = np.stack([np.logaddexp.reduce(logp[:, [p.startswith(s + "_") for p in players]], axis=1) for s in names], 1)
        bib_like &= np.isfinite(score).all(1)
    else:
        score = colour_score
    on_court = (court[:, 0] > -50) & (court[:, 0] < 1555) & (court[:, 1] > -50) & (court[:, 1] < 1155)
    in_game = np.array([s in squads_in_game for s in names])
    best_any = score.argmax(1)
    best_game = np.where(in_game[None], score, -np.inf).argmax(1)
    det_squad = np.where(in_game[best_any], np.array(names)[best_any], np.where(on_court, np.array(names)[best_game], ""))
    det_squad = np.where(bib_like & (dets["conf"] >= cfg.assembly.min_conf), det_squad, "")

    number_nll = None
    if logp is not None:
        number_nll = np.full((len(dets), 3), INFEASIBLE_NLL, np.float32)
        for squad, team in squads_in_game.items():
            m = det_squad == squad
            cols = [PLAYERS.index(f"{squad}_{n}") for n in game.roster[team]]
            sub = logp[m][:, cols]
            number_nll[m] = -(sub - np.logaddexp.reduce(sub, axis=1, keepdims=True))
    return det_squad, number_nll


def foot_points(dets, H, bottom_shift_frac: float = 0.0) -> np.ndarray:
    """Court position per detection: bbox bottom-centre, a ground-contact point in any
    view, optionally moved up by a fraction of the box height (to undo a detector's
    learned annotation margin in views other than the one it was trained on)."""
    bottom = dets["y2"] - bottom_shift_frac * (dets["y2"] - dets["y1"])
    return project(H, np.c_[(dets["x1"] + dets["x2"]) / 2, bottom])


def gt_camera_ground_position(cfg) -> np.ndarray:
    """Court (x, y) under the training (GT) camera, from its official registration."""
    H = image_to_court_homography(Path(cfg.data_root) / "training") @ SCALE_720_TO_4K
    cam, _ = calibrate(H, (1280, 720))
    return cam.centre[:2]


def assemble_slots(cfg, dets, feats, logp, H, game, protos, foot_cfg) -> tuple[np.ndarray, dict]:
    court = foot_points(dets, H, foot_cfg.bottom_shift_frac)
    if foot_cfg.gt_camera_bias_cm:
        # GT boxes (annotated in the GT camera) end ~1.5 px below COCO-detector boxes:
        # GT positions sit ~10 cm toward that camera (measured on training, 20 Oct).
        u = gt_camera_ground_position(cfg)[None] - court
        court = court + foot_cfg.gt_camera_bias_cm * u / np.linalg.norm(u, axis=1, keepdims=True)
    m = cfg.assembly.court_margin_cm
    in_gate = (court[:, 0] > -m) & (court[:, 0] < 1505 + m) & (court[:, 1] > -m) & (court[:, 1] < 1105 + m)
    det_squad, number_nll = detection_squads(cfg, dets, feats, logp, court, game, protos)
    det_squad = np.where(in_gate, det_squad, "")
    squads_in_game = game_squads(game)
    tracks = assign_slots(
        dets["frame"], court, det_squad, number_nll,
        {s: game.roster[t] for s, t in squads_in_game.items()}, SlotConfig(**cfg.assembly.slots),
    )
    positions = slots_to_positions(tracks, squads_in_game, cfg.assembly.slots.max_gap_fill)
    stats = {"n_candidates": int((det_squad != "").sum())}
    return positions, stats


@hydra.main(config_path="../configs", config_name="pipeline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    det_key = _key(cfg.detector)
    detector = None
    extractor = HsvTorsoFeatureExtractor()

    with start_run(cfg, run_name=cfg.run_name):
        mlflow.set_tag("output_dir", str(out_dir))
        mlflow.log_param("detector_cache_key", det_key)
        table = ["| video | tau | identity | TI-HOTA | DetA | AssA |", "|---|---|---|---|---|---|"]
        agg: dict[str, list[float]] = {}
        for split in ("training", "challenge"):
            games = load_games(Path(cfg.data_root) / split)
            for video in cfg.videos[split] or []:
                t0 = time.time()
                path = video_path(cfg, split, video)
                H = homography(cfg, split, video)
                cache = Path(cfg.cache_dir) / "det" / f"{cfg.detector.name}_{det_key}" / f"{video}.npz"
                if cfg.max_frames:
                    cache = cache.with_name(f"{video}_first{cfg.max_frames}.npz")
                if cache.exists():
                    dets, feats, _ = load_detections(cache)
                else:
                    if detector is None:
                        detector = UltralyticsPersonDetector(**{k: v for k, v in cfg.detector.items() if k != "name"})
                    dets, feats, keypoints = detect_video(path, detector, extractor, cfg.max_frames)
                    save_detections(cache, dets, feats, keypoints)  # keypoints kept for pose detectors
                n_frames = int(dets["frame"].max()) if len(dets) else 0

                loo = split == "training" and cfg.identity.leave_one_out
                protos = squad_prototypes(cfg, exclude_video=video if loo else None)
                if cfg.assembly.method == "tracks":
                    track_ids = run_tracker(instantiate(cfg.tracker), dets, n_frames)
                    positions, stats = assemble(cfg, dets, feats, track_ids, H, games[video], protos)
                else:
                    logp = None
                    if cfg.classifier.checkpoint:
                        cls_key = hashlib.sha1(f"{cfg.classifier.checkpoint}|{cfg.assembly.min_conf}".encode()).hexdigest()[:10]
                        logp = classify_detections(cfg, path, dets, cache.parent / f"cls_{cls_key}" / cache.name)
                    positions, stats = assemble_slots(cfg, dets, feats, logp, H, games[video], protos, cfg.foot[split])
                write_positions(out_dir / split / f"{video}.txt", positions)
                log_metrics_dict(stats, prefix=f"{video}/")
                log_metrics_dict({"seconds": time.time() - t0}, prefix=f"{video}/")

                if split == "training":
                    gt = load_positions(training_paths(video, Path(cfg.data_root))["positions"])
                    if cfg.max_frames:
                        gt = gt[gt["frame"] <= cfg.max_frames]
                    for tau in cfg.tau_cm:
                        for mode in ("full", "team", "none"):
                            m = ti_hota(collapse_identity(gt, mode), collapse_identity(positions, mode), tau=tau)
                            name = f"tau{tau}_{mode}"
                            log_metrics_dict(
                                {"TI-HOTA": m["TI-HOTA"], "TI-DetA": m["TI-DetA"], "TI-AssA": m["TI-AssA"]},
                                prefix=f"{video}/{name}/",
                            )
                            agg.setdefault(name, []).append(m["TI-HOTA"])
                            table.append(
                                f"| {video} | {tau} | {mode} | {100 * m['TI-HOTA']:.2f} | "
                                f"{100 * m['TI-DetA']:.2f} | {100 * m['TI-AssA']:.2f} |"
                            )
                else:
                    problems = validate_submission(out_dir / split / f"{video}.txt", games[video], n_frames)
                    mlflow.log_metric(f"{video}/format_problems", len(problems))
                    rows_per_frame = len(positions) / max(n_frames, 1)
                    mlflow.log_metric(f"{video}/rows_per_frame", rows_per_frame)
                    table.append(f"| {video} | - | rows/frame {rows_per_frame:.2f} | problems: {problems or 'none'} | | |")
                print(table[-1], flush=True)

        for name, vals in sorted(agg.items()):
            mlflow.log_metric(f"mean/{name}", float(np.mean(vals)))
            table.append(f"| **mean** | {name} | | **{100 * np.mean(vals):.2f}** | | |")
        (out_dir / "scores.md").write_text("\n".join(table) + "\n")
        print("\n".join(table))
        mlflow.log_artifact(str(out_dir / "scores.md"))
        if (out_dir / "challenge").exists():
            mlflow.log_artifacts(str(out_dir / "challenge"), artifact_path="challenge_positions")
        OmegaConf.save(cfg, out_dir / "config.yaml")
        mlflow.log_artifact(str(out_dir / "config.yaml"))


if __name__ == "__main__":
    main()
