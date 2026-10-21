"""Train the 12-way player classifier on training GT crops; logged to MLflow.

    python scripts/train_player_classifier.py run_name=cls_v1 [val_videos=[]] [backbone=resnet50.a1_in1k]

Validation reports overall accuracy and the accuracy that matters at inference:
argmax restricted to the six players of the crop's own game (roster prior), plus
squad accuracy and accuracy by box height.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import cv2
import hydra
import mlflow
import numpy as np
import torch
from handball.common.mlflow_utils import log_metrics_dict, start_run
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from bb3x3.data import DATA_ROOT, load_games
from bb3x3.identity import game_squads
from bb3x3.player_classifier import INPUT_HW, MEAN, PLAYERS, build_model


class CropDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict], aug: DictConfig | None):
        self.root, self.rows = root, rows
        self.labels = [PLAYERS.index(r["player"]) for r in rows]
        self.aug = aug
        if aug is not None:
            self.color = v2.ColorJitter(aug.brightness, aug.contrast, aug.saturation, aug.hue)
            self.erase = v2.RandomErasing(p=aug.erase_p, scale=(0.02, 0.15), value=0)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        img = cv2.imread(str(self.root / self.rows[i]["path"]))[..., ::-1]
        x = torch.from_numpy(img.copy()).permute(2, 0, 1)  # uint8 CHW RGB
        a = self.aug
        if a is not None:
            h, w = x.shape[1:]
            if random.random() < a.downscale_p:
                s = random.uniform(a.downscale_min, 1.0)
                x = v2.functional.resize(x, [max(8, int(h * s)), max(4, int(w * s))], antialias=True)
                x = v2.functional.resize(x, [h, w], antialias=True)
            dx, dy = (int(random.uniform(-a.shift, a.shift) * d) for d in (w, h))
            x = v2.functional.affine(x, angle=0.0, translate=[dx, dy], scale=random.uniform(0.9, 1.1), shear=[0.0])
            x = self.color(x)
            if random.random() < a.blur_p:
                x = v2.functional.gaussian_blur(x, kernel_size=5, sigma=random.uniform(0.3, 1.2))
        x = v2.functional.resize(x, list(INPUT_HW), antialias=True).float() / 255.0
        x = (x - MEAN[0]) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        if a is not None:
            x = self.erase(x)
        return x, self.labels[i], i


def roster_masks(rows: list[dict]) -> np.ndarray:
    """(N, 12) bool: the six players of each crop's game."""
    games = load_games(DATA_ROOT / "training")
    allowed = {}
    for video, game in games.items():
        squads = game_squads(game)
        allowed[video] = np.array([p.split("_")[0] in squads for p in PLAYERS])
    return np.stack([allowed[r["video"]] for r in rows])


@torch.no_grad()
def evaluate(model, loader, rows, device) -> dict[str, float]:
    model.eval()
    logits = np.zeros((len(rows), len(PLAYERS)), np.float32)
    for x, _, idx in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits[idx.numpy()] = model(x.to(device, non_blocking=True)).float().cpu().numpy()
    labels = np.array([PLAYERS.index(r["player"]) for r in rows])
    masked = np.where(roster_masks(rows), logits, -np.inf)
    pred_all, pred_roster = logits.argmax(1), masked.argmax(1)
    squad = np.array([PLAYERS[i].split("_")[0] for i in range(len(PLAYERS))])
    heights = np.array([float(r["box_h"]) for r in rows])
    out = {
        "acc": float((pred_all == labels).mean()),
        "acc_roster": float((pred_roster == labels).mean()),
        "squad_acc_roster": float((squad[pred_roster] == squad[labels]).mean()),
    }
    for lo, hi in [(0, 70), (70, 100), (100, 1e9)]:
        m = (heights >= lo) & (heights < hi)
        if m.any():
            out[f"acc_roster_h{lo}"] = float((pred_roster[m] == labels[m]).mean())
    return out


@hydra.main(config_path="../configs", config_name="player_classifier", version_base="1.3")
def main(cfg: DictConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    root = Path(cfg.crops_dir)
    with open(root / "index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    train_rows = [r for r in rows if r["video"] not in cfg.val_videos]
    val_rows = [r for r in rows if r["video"] in cfg.val_videos][::3]  # every 9th frame is plenty for val
    if cfg.max_train_samples:
        train_rows = random.sample(train_rows, cfg.max_train_samples)

    device = "cuda"
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(
        CropDataset(root, train_rows, cfg.augment), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    val_loader = (
        DataLoader(CropDataset(root, val_rows, None), batch_size=256, num_workers=cfg.workers, pin_memory=True)
        if val_rows else None
    )
    model = build_model(cfg.backbone, cfg.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(train_loader)
    warm = max(1, int(cfg.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    with start_run(cfg, run_name=cfg.run_name):
        mlflow.set_tag("task", "player_classifier")
        mlflow.set_tag("checkpoint_dir", str(out_dir))
        mlflow.log_params({"n_train": len(train_rows), "n_val": len(val_rows)})
        best, step = -1.0, 0
        for epoch in range(cfg.epochs):
            model.train()
            running = 0.0
            for x, y, _ in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = loss_fn(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                running += loss.item()
                step += 1
                if step % 100 == 0:
                    mlflow.log_metric("train_loss", running / 100, step=step)
                    running = 0.0
            state = {"model": model.state_dict(), "backbone": cfg.backbone, "players": PLAYERS,
                     "config": OmegaConf.to_container(cfg, resolve=True), "epoch": epoch}
            torch.save(state, out_dir / "last.pt")
            if val_loader is not None:
                metrics = evaluate(model, val_loader, val_rows, device)
                log_metrics_dict(metrics, step=epoch, prefix="val_")
                print(f"epoch {epoch}: {metrics}", flush=True)
                if metrics["acc_roster"] > best:
                    best = metrics["acc_roster"]
                    torch.save(state, out_dir / "best.pt")
            else:
                print(f"epoch {epoch} done", flush=True)
        if val_loader is not None:
            mlflow.log_metric("best_val_acc_roster", best)
        OmegaConf.save(cfg, out_dir / "config.yaml")
        mlflow.log_artifact(str(out_dir / "config.yaml"))


if __name__ == "__main__":
    main()
