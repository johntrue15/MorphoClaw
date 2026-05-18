"""3D U-Net student model wrapper.

The student is a MONAI ``UNet`` trained on (volume, pseudo-label) pairs
produced by :mod:`pseudo_label_generator`. Compared to the upstream
nnInteractive backbone, this model:

- specialises on the *specific* anatomy the iterative loop has been
  asked to segment (e.g. cranial cavity, cortical bone), and
- is fully autonomous at inference time — no prompts.

Heavy dependencies (torch / MONAI / SimpleITK) are imported lazily so
the rest of the ``seg_train`` package can be unit-tested without them.
The implementation is deliberately small and patch-based so it can run
on the Apple-Silicon Mac mini runner; production-scale runs should use
the same code on a CUDA box (just point ``NNINTERACTIVE_HOME`` at a
venv built with the CUDA torch wheels).

Inference exposes both a deterministic forward pass and a Monte-Carlo
dropout pass that yields a per-voxel uncertainty map — used by
:mod:`confidence_router` and recorded in the ledger.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


log = logging.getLogger("seg_train.student")


class StudentUnavailable(RuntimeError):
    """Raised when torch / MONAI / SimpleITK cannot be imported."""


def _require_backend():
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
        import SimpleITK as sitk  # noqa: F401
        from monai.networks.nets import UNet  # noqa: F401
        from monai.transforms import (  # noqa: F401
            Compose, EnsureChannelFirstd, LoadImaged, NormalizeIntensityd,
            RandSpatialCropd, ScaleIntensityRanged, SpatialPadd, ToTensord,
        )
    except ImportError as exc:
        raise StudentUnavailable(
            "Student model requires torch, monai, SimpleITK. Install with:\n"
            "    "
            f"{os.environ.get('NNINTERACTIVE_HOME', '~/.autoresearchclaw/nninteractive')}/bin/pip "
            "install 'monai[nibabel,einops]>=1.3'\n"
            f"(import error: {exc})"
        ) from exc


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class StudentConfig:
    """Hyperparameters for the student model."""

    in_channels: int = 1
    out_channels: int = 2
    spatial_dims: int = 3
    channels: tuple = (16, 32, 64, 128, 256)
    strides: tuple = (2, 2, 2, 2)
    num_res_units: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    batch_size: int = 1
    patch_size: tuple = (96, 96, 96)
    intensity_clip: tuple = (-1000.0, 2000.0)
    intensity_scale: tuple = (0.0, 1.0)
    val_every: int = 5
    seed: int = 42
    device: str = ""  # "" → auto, "cuda:0", "mps", "cpu"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Model artefact
# ---------------------------------------------------------------------------


@dataclass
class StudentArtifact:
    """A trained student model on disk."""

    version: str
    weights_path: str
    config_path: str
    train_log_path: str = ""
    val_dice_history: list = field(default_factory=list)
    epochs_trained: int = 0
    train_samples: int = 0
    val_samples: int = 0
    best_val_dice: Optional[float] = None
    parent_version: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StudentArtifact":
        known = {f.name for f in dataclasses.fields(cls)}
        cleaned = {k: v for k, v in d.items() if k in known}
        return cls(**cleaned)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class StudentTrainer:
    """Train a MONAI U-Net on the iterative dataset."""

    def __init__(self, config: Optional[StudentConfig] = None) -> None:
        self.config = config or StudentConfig()

    # ------------------------------------------------------------------

    def _select_device(self):
        import torch
        forced = (self.config.device or os.environ.get(
            "STUDENT_DEVICE", "")).strip().lower()
        if forced:
            return torch.device(forced)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps",
                   None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_model(self):
        from monai.networks.nets import UNet
        return UNet(
            spatial_dims=self.config.spatial_dims,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            channels=self.config.channels,
            strides=self.config.strides,
            num_res_units=self.config.num_res_units,
            dropout=self.config.dropout,
        )

    def _build_pipeline(self, training: bool):
        from monai.transforms import (
            Compose, EnsureChannelFirstd, LoadImaged, NormalizeIntensityd,
            RandFlipd, RandSpatialCropd, ScaleIntensityRanged, SpatialPadd,
            ToTensord,
        )
        keys = ("image", "label")
        clip_lo, clip_hi = self.config.intensity_clip
        scale_lo, scale_hi = self.config.intensity_scale
        steps = [
            LoadImaged(keys=keys, image_only=True),
            EnsureChannelFirstd(keys=keys),
            ScaleIntensityRanged(
                keys=("image",), a_min=clip_lo, a_max=clip_hi,
                b_min=scale_lo, b_max=scale_hi, clip=True,
            ),
            NormalizeIntensityd(keys=("image",), nonzero=False, channel_wise=True),
            SpatialPadd(keys=keys, spatial_size=self.config.patch_size),
        ]
        if training:
            steps.extend([
                RandSpatialCropd(
                    keys=keys, roi_size=self.config.patch_size,
                    random_size=False,
                ),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            ])
        steps.append(ToTensord(keys=keys))
        return Compose(steps)

    def _prepare_dataset(self, samples, training: bool):
        from monai.data import CacheDataset, DataLoader
        items = [
            {
                "image": s.volume_path,
                "label": s.label_path,
                "weight": float(s.reliability),
                "sample_id": s.sample_id,
            }
            for s in samples
        ]
        if not items:
            return None
        ds = CacheDataset(
            data=items,
            transform=self._build_pipeline(training=training),
            cache_rate=0.0,
            num_workers=0,
        )
        return DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=training,
            num_workers=0,
        )

    # ------------------------------------------------------------------

    def train(
        self,
        train_samples: list,
        val_samples: list,
        *,
        output_dir: str,
        version: str,
        parent_version: str = "",
    ) -> StudentArtifact:
        """Train a fresh model on *train_samples*, validating on *val_samples*."""
        _require_backend()
        import numpy as np
        import torch
        from monai.losses import DiceCELoss

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        device = self._select_device()
        log.info(
            "Training student %s — train=%d val=%d device=%s",
            version, len(train_samples), len(val_samples), device,
        )
        model = self._build_model().to(device)
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        train_loader = self._prepare_dataset(train_samples, training=True)
        val_loader = self._prepare_dataset(val_samples, training=False)
        if train_loader is None:
            raise RuntimeError("No training samples; cannot train student.")

        history: list[dict] = []
        best_val_dice = -1.0

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            model.train()
            train_losses: list[float] = []
            for batch in train_loader:
                image = batch["image"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True).long()
                weight = float(batch["weight"][0])
                optim.zero_grad()
                logits = model(image)
                loss = loss_fn(logits, label) * weight
                loss.backward()
                optim.step()
                train_losses.append(float(loss.item()))

            mean_train_loss = sum(train_losses) / max(len(train_losses), 1)

            val_dice = None
            if val_loader is not None and (
                epoch % self.config.val_every == 0 or epoch == self.config.epochs
            ):
                val_dice = self._validate(model, val_loader, device)
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    torch.save(model.state_dict(), out / f"{version}.pt")

            history.append({
                "epoch": epoch,
                "train_loss": round(mean_train_loss, 6),
                "val_dice": round(val_dice, 6) if val_dice is not None else None,
                "duration_s": round(time.time() - t0, 2),
            })
            log.info(
                "  epoch %d: loss=%.4f val_dice=%s (%.1fs)",
                epoch, mean_train_loss,
                f"{val_dice:.4f}" if val_dice is not None else "-",
                time.time() - t0,
            )

        weights_path = out / f"{version}.pt"
        if not weights_path.exists():
            torch.save(model.state_dict(), weights_path)

        config_path = out / f"{version}.config.json"
        config_payload = {
            "config": self.config.to_dict(),
            "version": version,
            "parent_version": parent_version,
            "history": history,
        }
        config_path.write_text(json.dumps(config_payload, indent=2, default=str))

        train_log = out / f"{version}.train_log.json"
        train_log.write_text(json.dumps(history, indent=2, default=str))

        artifact = StudentArtifact(
            version=version,
            weights_path=str(weights_path),
            config_path=str(config_path),
            train_log_path=str(train_log),
            val_dice_history=[h["val_dice"] for h in history
                              if h["val_dice"] is not None],
            epochs_trained=self.config.epochs,
            train_samples=len(train_samples),
            val_samples=len(val_samples),
            best_val_dice=round(best_val_dice, 6) if best_val_dice >= 0 else None,
            parent_version=parent_version,
        )
        (out / f"{version}.artifact.json").write_text(
            json.dumps(artifact.to_dict(), indent=2, default=str)
        )
        return artifact

    def _validate(self, model, val_loader, device) -> float:
        import torch
        from monai.metrics import DiceMetric

        metric = DiceMetric(include_background=False, reduction="mean")
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True).long()
                logits = model(image)
                pred = logits.argmax(dim=1, keepdim=True)
                # one-hot
                one_hot_pred = torch.nn.functional.one_hot(
                    pred.squeeze(1), num_classes=self.config.out_channels
                ).permute(0, 4, 1, 2, 3).float()
                one_hot_label = torch.nn.functional.one_hot(
                    label.squeeze(1), num_classes=self.config.out_channels
                ).permute(0, 4, 1, 2, 3).float()
                metric(y_pred=one_hot_pred, y=one_hot_label)
        return float(metric.aggregate().item())


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@dataclass
class StudentInferenceResult:
    label_path: str
    probability_path: str
    uncertainty_path: str
    voxel_count: int
    total_voxels: int
    mean_confidence: float
    mean_entropy: float
    bounding_box_xyz: Optional[list]
    duration_s: float
    model_version: str


class StudentInference:
    """Sliding-window 3D inference with optional MC-dropout uncertainty."""

    def __init__(
        self,
        artifact_path: str,
        config: Optional[StudentConfig] = None,
    ) -> None:
        _require_backend()
        import torch

        self.artifact_path = Path(artifact_path)
        if self.artifact_path.is_dir():
            # Pick most recent artifact in directory.
            arts = sorted(self.artifact_path.glob("*.artifact.json"))
            if not arts:
                raise FileNotFoundError(
                    f"No *.artifact.json under {self.artifact_path}"
                )
            self.artifact_path = arts[-1]
        artifact_data = json.loads(self.artifact_path.read_text())
        self.artifact = StudentArtifact.from_dict(artifact_data)
        config_data = json.loads(
            Path(self.artifact.config_path).read_text()
        )
        self.config = config or StudentConfig(**config_data["config"])
        self.device = self._select_device()

        from monai.networks.nets import UNet
        self.model = UNet(
            spatial_dims=self.config.spatial_dims,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            channels=self.config.channels,
            strides=self.config.strides,
            num_res_units=self.config.num_res_units,
            dropout=self.config.dropout,
        ).to(self.device)
        state = torch.load(
            str(self.artifact.weights_path), map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state)
        self.model.eval()

    # ------------------------------------------------------------------

    def _select_device(self):
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------

    def predict(
        self,
        volume_path: str,
        output_dir: str,
        *,
        media_id: str = "unknown",
        n_dropout_samples: int = 0,
        roi_size: Optional[tuple] = None,
        sw_batch_size: int = 4,
        threshold: float = 0.5,
    ) -> StudentInferenceResult:
        import numpy as np
        import SimpleITK as sitk
        import torch
        from monai.inferers import sliding_window_inference

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        sitk_image = sitk.ReadImage(str(volume_path))
        image = sitk.GetArrayFromImage(sitk_image)  # z, y, x
        clip_lo, clip_hi = self.config.intensity_clip
        scale_lo, scale_hi = self.config.intensity_scale
        image = np.clip(image, clip_lo, clip_hi)
        image = (image - clip_lo) / (clip_hi - clip_lo + 1e-9)
        image = (image - image.mean()) / (image.std() + 1e-9)

        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(
            self.device
        )

        roi = roi_size or self.config.patch_size
        with torch.no_grad():
            logits = sliding_window_inference(
                tensor, roi, sw_batch_size, self.model, overlap=0.25,
            )
            prob = torch.softmax(logits, dim=1)[:, 1, ...]  # foreground prob

            uncertainty = None
            if n_dropout_samples > 0:
                self.model.train()  # enable dropout
                samples = []
                for _ in range(n_dropout_samples):
                    li = sliding_window_inference(
                        tensor, roi, sw_batch_size, self.model, overlap=0.25,
                    )
                    samples.append(torch.softmax(li, dim=1)[:, 1, ...])
                self.model.eval()
                stack = torch.stack(samples, dim=0)
                uncertainty = stack.std(dim=0)
                prob = stack.mean(dim=0)

        prob_np = prob.detach().cpu().squeeze().numpy()
        unc_np = (
            uncertainty.detach().cpu().squeeze().numpy()
            if uncertainty is not None else None
        )
        binary = (prob_np > threshold).astype(np.uint8)

        # Persist outputs
        prob_img = sitk.GetImageFromArray(prob_np.astype(np.float32))
        prob_img.CopyInformation(sitk_image)
        prob_path = out_dir / f"{media_id}_student_prob.nii.gz"
        sitk.WriteImage(prob_img, str(prob_path), useCompression=True)

        label_img = sitk.GetImageFromArray(binary)
        label_img.CopyInformation(sitk_image)
        label_path = out_dir / f"{media_id}_student_labelmap.nii.gz"
        sitk.WriteImage(label_img, str(label_path), useCompression=True)

        unc_path = out_dir / f"{media_id}_student_uncertainty.nii.gz"
        if unc_np is not None:
            unc_img = sitk.GetImageFromArray(unc_np.astype(np.float32))
            unc_img.CopyInformation(sitk_image)
            sitk.WriteImage(unc_img, str(unc_path), useCompression=True)
        else:
            unc_path = ""

        voxel_count = int(binary.sum())
        total = int(binary.size)
        mean_conf = float(prob_np[binary > 0].mean()) if voxel_count > 0 else float(
            prob_np.max()
        )
        # binary entropy on the foreground probability
        eps = 1e-9
        safe = np.clip(prob_np, eps, 1 - eps)
        h = -(safe * np.log(safe) + (1 - safe) * np.log(1 - safe))
        mean_entropy = float(h.mean())

        bbox = None
        if voxel_count > 0:
            zs, ys, xs = np.where(binary > 0)
            bbox = [
                [int(xs.min()), int(xs.max())],
                [int(ys.min()), int(ys.max())],
                [int(zs.min()), int(zs.max())],
            ]

        return StudentInferenceResult(
            label_path=str(label_path),
            probability_path=str(prob_path),
            uncertainty_path=str(unc_path),
            voxel_count=voxel_count,
            total_voxels=total,
            mean_confidence=round(mean_conf, 6),
            mean_entropy=round(mean_entropy, 6),
            bounding_box_xyz=bbox,
            duration_s=round(time.time() - t0, 2),
            model_version=self.artifact.version,
        )
