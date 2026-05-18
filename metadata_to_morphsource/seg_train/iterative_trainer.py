"""Round-based orchestrator for the iterative segmentation pipeline.

A *round* takes a list of MorphoSource specimens and:

1. Downloads the CT volume (and, if present, a curated GT mesh).
2. Voxelises any GT mesh onto the CT grid for evaluation.
3. Generates a pseudo-label using the appropriate path:

   - **Round 0** — always nnInteractive (we have no student yet).
   - **Round n>0** — student inference first, then the
     :class:`~.confidence_router.ConfidenceRouter` decides whether to
     accept it or invoke nnInteractive for correction.

4. Computes metrics against any available GT, records every event in
   the :class:`~.experiment_ledger.ExperimentLedger`, and adds the new
   sample to the :class:`~.dataset.DatasetManifest`.
5. Trains a fresh student on the accumulated dataset (or skips training
   when ``train=False`` is passed).
6. Emits a per-round summary report.

Graduation is checked after every training step: if the student's
validation Dice on the held-out human pool exceeds
``graduation_dice_threshold`` for two consecutive rounds, the
``graduated`` flag is flipped and subsequent rounds skip the
nnInteractive correction step (still logging both for the paper).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from .confidence_router import (
    ConfidenceRouter, RouteResult, RouterDecision, RouterPolicy,
    StudentPrediction,
)
from .dataset import (
    DatasetManifest, SegmentationSample, SplitName, reliability_for,
)
from .experiment_ledger import (
    EpisodeRecord, ExperimentLedger, SegmentationMode,
)
from .pseudo_label_generator import (
    PseudoLabelResult, run_paint_loop, voxelize_curated_mesh,
)


log = logging.getLogger("seg_train.trainer")


# ---------------------------------------------------------------------------


@dataclass
class SpecimenInput:
    """Input description for one specimen processed in a round."""

    media_id: str
    volume_path: str
    physical_object_id: str = ""
    taxonomy: str = ""
    morphosource_query: str = ""
    gt_mesh_path: str = ""           # raw mesh (before voxelisation)
    gt_label_path: str = ""          # already-voxelised label, if available
    gt_media_id: str = ""

    @property
    def has_gt(self) -> bool:
        return bool(self.gt_mesh_path or self.gt_label_path)


@dataclass
class RoundConfig:
    """Hyperparameters for one iterative round."""

    goal: str
    output_root: str
    paper_tag: str = ""
    max_paint_steps: int = 12
    train_after_round: bool = True
    student_epochs: int = 30
    n_dropout_samples: int = 4
    graduation_dice_threshold: float = 0.85
    student_min_reliability: float = 0.7
    require_human_review_for_holdout: bool = True
    seed: int = 1234


@dataclass
class RoundReport:
    round_index: int
    n_specimens: int
    episodes: list = field(default_factory=list)
    student_artifact_path: str = ""
    student_val_dice: Optional[float] = None
    graduated: bool = False
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d


# ---------------------------------------------------------------------------


class IterativeTrainer:
    """Drives many rounds of pseudo-label generation + student training."""

    def __init__(
        self,
        *,
        ledger: ExperimentLedger,
        manifest: DatasetManifest,
        router: Optional[ConfidenceRouter] = None,
        round_config: RoundConfig,
    ) -> None:
        self.ledger = ledger
        self.manifest = manifest
        self.router = router or ConfidenceRouter(RouterPolicy())
        self.config = round_config
        self.output_root = Path(round_config.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.current_round_index: int = self._infer_round_index()
        self.graduated: bool = False
        self.consecutive_passes: int = 0
        self._latest_artifact: Optional[dict] = None  # student artifact

    def _infer_round_index(self) -> int:
        existing = sorted(
            int(p.name.replace("round_", ""))
            for p in self.output_root.glob("round_*")
            if p.is_dir() and p.name.replace("round_", "").isdigit()
        )
        return (existing[-1] + 1) if existing else 0

    # ------------------------------------------------------------------

    def run_round(
        self,
        specimens: list[SpecimenInput],
        *,
        student_artifact_path: str = "",
        skip_training: bool = False,
    ) -> RoundReport:
        """Process one batch of specimens.

        ``student_artifact_path`` should point at an artifact JSON written
        by :class:`StudentTrainer.train` if the round should make use of
        student inference. When empty, the round falls back to pure
        nnInteractive (Round 0 behaviour).
        """
        round_index = self.current_round_index
        round_dir = self.output_root / f"round_{round_index:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        log.info("=" * 60)
        log.info("Round %d — %d specimens; student=%s",
                 round_index, len(specimens),
                 student_artifact_path or "(none)")
        log.info("=" * 60)

        report = RoundReport(round_index=round_index,
                             n_specimens=len(specimens))
        t0 = time.time()

        student_inference = None
        if student_artifact_path:
            try:
                from .student_model import StudentInference
                student_inference = StudentInference(student_artifact_path)
                log.info("Loaded student v=%s",
                         student_inference.artifact.version)
            except Exception as exc:
                log.warning("Could not load student %s: %s; "
                            "falling back to nnInteractive only",
                            student_artifact_path, exc)
                student_inference = None

        for specimen in specimens:
            try:
                episode_records = self._process_specimen(
                    specimen=specimen,
                    round_dir=round_dir,
                    round_index=round_index,
                    student_inference=student_inference,
                )
                report.episodes.extend(episode_records)
            except Exception as exc:
                log.exception("Specimen %s failed: %s", specimen.media_id, exc)

        # ---- Training ----
        if (not skip_training) and self.config.train_after_round:
            artifact = self._train_student(round_index=round_index,
                                           round_dir=round_dir)
            if artifact is not None:
                report.student_artifact_path = str(
                    Path(artifact.weights_path).with_suffix(".artifact.json")
                )
                report.student_val_dice = artifact.best_val_dice
                if (artifact.best_val_dice is not None
                        and artifact.best_val_dice
                        >= self.config.graduation_dice_threshold):
                    self.consecutive_passes += 1
                else:
                    self.consecutive_passes = 0
                if self.consecutive_passes >= 2:
                    self.graduated = True
                report.graduated = self.graduated
                self._latest_artifact = dataclasses.asdict(artifact)

        report.duration_s = round(time.time() - t0, 2)
        (round_dir / "round_report.json").write_text(
            json.dumps(report.to_dict(), indent=2, default=str)
        )
        log.info(
            "Round %d done in %.1fs — student_val_dice=%s graduated=%s",
            round_index, report.duration_s, report.student_val_dice,
            self.graduated,
        )

        self.current_round_index += 1
        # Persist manifest after each round.
        self.manifest.save()
        return report

    # ------------------------------------------------------------------

    def _process_specimen(
        self,
        *,
        specimen: SpecimenInput,
        round_dir: Path,
        round_index: int,
        student_inference,
    ) -> list[EpisodeRecord]:
        """Generate one or more segmentations for *specimen*, log them,
        and add the chosen pseudo-label to the dataset manifest.
        """
        records: list[EpisodeRecord] = []
        spec_dir = round_dir / specimen.media_id
        spec_dir.mkdir(parents=True, exist_ok=True)

        # ---- 1. Voxelise GT if a mesh was supplied ----
        gt_label_path = specimen.gt_label_path
        if specimen.gt_mesh_path and not gt_label_path:
            target = spec_dir / "gt_voxelized.nii.gz"
            log.info("Voxelising curated mesh for %s", specimen.media_id)
            vox = voxelize_curated_mesh(
                reference_volume=specimen.volume_path,
                mesh_path=specimen.gt_mesh_path,
                output_path=str(target),
            )
            records.append(self._record_pseudo_label(
                specimen=specimen, result=vox,
                mode=SegmentationMode.HUMAN_GT,
                round_index=round_index,
                router_result=None,
                operator="human_curator",
            ))
            if vox.success:
                gt_label_path = vox.label_path

        # Promote the GT into the manifest as a holdout sample.
        if gt_label_path and Path(gt_label_path).exists():
            self.manifest.add_sample(SegmentationSample(
                sample_id=f"{specimen.media_id}__gt",
                media_id=specimen.media_id,
                physical_object_id=specimen.physical_object_id,
                volume_path=specimen.volume_path,
                label_path=gt_label_path,
                mode="human_gt",
                round_index=round_index,
                reliability=reliability_for("human_gt"),
                has_human_review=True,
                taxonomy=specimen.taxonomy,
                source_run_id=self.ledger.run_id,
                notes="Voxelised MorphoSource curated mesh",
            ))

        # ---- 2. Student inference (if available) ----
        student_label_path: str = ""
        student_prediction: Optional[StudentPrediction] = None
        student_record: Optional[EpisodeRecord] = None
        if student_inference is not None:
            log.info("Running student inference on %s", specimen.media_id)
            try:
                pred = student_inference.predict(
                    volume_path=specimen.volume_path,
                    output_dir=str(spec_dir / "student"),
                    media_id=specimen.media_id,
                    n_dropout_samples=self.config.n_dropout_samples,
                )
                student_prediction = StudentPrediction(
                    label_path=pred.label_path,
                    voxel_count=pred.voxel_count,
                    total_voxels=pred.total_voxels,
                    mean_confidence=pred.mean_confidence,
                    mean_entropy=pred.mean_entropy,
                    bounding_box_xyz=pred.bounding_box_xyz,
                )
                student_label_path = pred.label_path
                student_record = self._record_student_episode(
                    specimen=specimen, pred=pred,
                    gt_label_path=gt_label_path,
                    round_index=round_index,
                )
                records.append(student_record)
            except Exception as exc:
                log.warning("Student inference failed for %s: %s",
                            specimen.media_id, exc)

        # ---- 3. Routing decision ----
        route_result: Optional[RouteResult] = None
        if student_prediction is not None and not self.graduated:
            route_result = self.router.decide(student_prediction)
            log.info("Router decision for %s: %s — %s",
                     specimen.media_id, route_result.decision.value,
                     route_result.reason)
        elif student_prediction is not None and self.graduated:
            route_result = RouteResult(
                decision=RouterDecision.ACCEPT_STUDENT,
                reason="graduated — student is autonomous",
                policy_name=self.router.policy.name,
            )
        else:
            route_result = RouteResult(
                decision=RouterDecision.REJECT,
                reason="no student available — fall back to nnInteractive",
                policy_name="bootstrap",
            )

        # ---- 4. Possibly invoke nnInteractive ----
        chosen_label: str = ""
        chosen_mode: str = ""
        if route_result.routes_to_nninteractive:
            paint = run_paint_loop(
                volume_path=specimen.volume_path,
                goal=self.config.goal,
                output_dir=str(spec_dir / "nninteractive"),
                media_id=specimen.media_id,
                max_steps=self.config.max_paint_steps,
            )
            paint_record = self._record_pseudo_label(
                specimen=specimen,
                result=paint,
                mode=(
                    SegmentationMode.STUDENT_CORRECTED
                    if (student_prediction is not None
                        and route_result.decision
                        == RouterDecision.CORRECT_WITH_NNINTERACTIVE)
                    else SegmentationMode.NNINTERACTIVE
                ),
                round_index=round_index,
                router_result=route_result,
                operator=os.environ.get(
                    "NNINTERACTIVE_VISION_MODEL", "llm:gpt-4o"
                ),
                gt_label_path=gt_label_path,
            )
            records.append(paint_record)
            if paint.success:
                chosen_label = paint.label_path
                chosen_mode = paint_record.mode

        # If the student was accepted, that's the chosen label.
        if (route_result.decision == RouterDecision.ACCEPT_STUDENT
                and student_label_path):
            chosen_label = student_label_path
            chosen_mode = SegmentationMode.STUDENT.value

        # ---- 5. Add chosen label to dataset manifest ----
        if chosen_label and Path(chosen_label).exists():
            sid = f"{specimen.media_id}__r{round_index:03d}__{chosen_mode}"
            sample = SegmentationSample(
                sample_id=sid,
                media_id=specimen.media_id,
                physical_object_id=specimen.physical_object_id,
                volume_path=specimen.volume_path,
                label_path=chosen_label,
                mode=chosen_mode,
                round_index=round_index,
                taxonomy=specimen.taxonomy,
                source_run_id=self.ledger.run_id,
            )
            self.manifest.add_sample(sample)
            log.info("Added sample %s (%s) to manifest", sid, chosen_mode)

        return records

    # ------------------------------------------------------------------
    # Episode-record builders
    # ------------------------------------------------------------------

    def _record_pseudo_label(
        self,
        *,
        specimen: SpecimenInput,
        result: PseudoLabelResult,
        mode: SegmentationMode,
        round_index: int,
        router_result: Optional[RouteResult],
        operator: str = "",
        gt_label_path: str = "",
    ) -> EpisodeRecord:
        rec = EpisodeRecord(
            run_id=self.ledger.run_id,
            round_index=round_index,
            paper_tag=self.config.paper_tag,
            media_id=specimen.media_id,
            physical_object_id=specimen.physical_object_id,
            taxonomy=specimen.taxonomy,
            morphosource_query=specimen.morphosource_query,
            volume_path=specimen.volume_path,
            mode=mode.value,
            goal=self.config.goal,
            prediction_path=result.label_path if result.success else "",
            n_prompts=result.n_prompts,
            prompt_kinds=result.prompt_kinds,
            operator=operator,
            model_version=result.model_version,
            duration_s=result.duration_s,
            has_ground_truth=bool(
                gt_label_path and Path(gt_label_path).exists()
            ),
            ground_truth_path=gt_label_path or "",
            ground_truth_source=("morphosource_mesh"
                                 if gt_label_path else ""),
        )
        if router_result is not None:
            rec.router_policy = router_result.policy_name
            rec.router_threshold = router_result.threshold
            rec.routed_to_nninteractive = router_result.routes_to_nninteractive
        if not result.success:
            rec.extra["error"] = result.error
        # Compute metrics if GT is available and prediction exists.
        if rec.has_ground_truth and rec.prediction_path:
            self._fill_metrics(rec)
        return self.ledger.record(rec)

    def _record_student_episode(
        self,
        *,
        specimen: SpecimenInput,
        pred,  # StudentInferenceResult
        gt_label_path: str,
        round_index: int,
    ) -> EpisodeRecord:
        rec = EpisodeRecord(
            run_id=self.ledger.run_id,
            round_index=round_index,
            paper_tag=self.config.paper_tag,
            media_id=specimen.media_id,
            physical_object_id=specimen.physical_object_id,
            taxonomy=specimen.taxonomy,
            morphosource_query=specimen.morphosource_query,
            volume_path=specimen.volume_path,
            mode=SegmentationMode.STUDENT.value,
            goal=self.config.goal,
            prediction_path=pred.label_path,
            operator=f"student:{pred.model_version}",
            model_version=pred.model_version,
            duration_s=pred.duration_s,
            voxel_count=pred.voxel_count,
            confidence_score=pred.mean_confidence,
            entropy_score=pred.mean_entropy,
            has_ground_truth=bool(
                gt_label_path and Path(gt_label_path).exists()
            ),
            ground_truth_path=gt_label_path or "",
            ground_truth_source=("morphosource_mesh"
                                 if gt_label_path else ""),
        )
        if rec.has_ground_truth:
            self._fill_metrics(rec)
        return self.ledger.record(rec)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _fill_metrics(self, rec: EpisodeRecord) -> None:
        """Compute Dice/IoU/etc using the existing segmentation_metrics."""
        try:
            sys.path.insert(0, str(
                Path(__file__).resolve().parent.parent.parent
                / ".github" / "scripts"
            ))
            from segmentation_metrics import compare_labelmaps  # type: ignore
        except Exception as exc:
            log.debug("Cannot import segmentation_metrics: %s", exc)
            return
        try:
            metrics = compare_labelmaps(
                rec.prediction_path, rec.ground_truth_path,
                compute_surface_distances=True,
            )
        except Exception as exc:
            log.warning("Metric computation failed for %s: %s",
                        rec.media_id, exc)
            return
        rec.dice = metrics.dice
        rec.iou = metrics.iou
        rec.precision = metrics.precision
        rec.recall = metrics.recall
        rec.hausdorff_mm = metrics.hausdorff_mm
        rec.hausdorff_95_mm = metrics.hausdorff_95_mm
        rec.average_surface_dist_mm = metrics.average_surface_dist_mm
        rec.centroid_distance_mm = metrics.centroid_distance_mm
        rec.volume_pred_mm3 = metrics.volume_mm3_pred
        rec.volume_gt_mm3 = metrics.volume_mm3_gt
        rec.volume_difference_pct = metrics.volume_difference_pct
        rec.voxel_count = int(metrics.voxel_count_pred)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_student(self, *, round_index: int, round_dir: Path):
        try:
            from .student_model import (
                StudentArtifact, StudentConfig, StudentTrainer,
            )
        except Exception as exc:
            log.warning("Student model unavailable (%s); "
                        "skipping training step.", exc)
            return None

        train_samples = self.manifest.split_samples(
            SplitName.TRAIN.value,
            min_reliability=self.config.student_min_reliability,
        )
        val_samples = self.manifest.split_samples(SplitName.VAL.value)
        if not train_samples:
            log.warning("No train samples in manifest; skipping training.")
            return None

        cfg = StudentConfig(epochs=self.config.student_epochs,
                            seed=self.config.seed)
        trainer = StudentTrainer(cfg)
        version = f"student_r{round_index:03d}"
        try:
            artifact = trainer.train(
                train_samples=train_samples,
                val_samples=val_samples,
                output_dir=str(round_dir / "student_weights"),
                version=version,
                parent_version=(
                    self._latest_artifact.get("version", "")
                    if self._latest_artifact else ""
                ),
            )
        except Exception as exc:
            log.exception("Student training failed: %s", exc)
            return None
        return artifact
