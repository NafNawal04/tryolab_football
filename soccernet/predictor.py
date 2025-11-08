from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class SoccerNetPredictorError(RuntimeError):
    """Raised when the SoccerNet inference pipeline cannot be executed."""


@dataclass
class SoccerNetPredictorConfig:
    """Configuration for running SoccerNet CALF inference."""

    root_dir: Path = Path("soccernet")
    output_dir: Path = Path("soccernet/generated")
    repo_subdir: Path = Path("SoccerNetv2-DevKit/Task1-ActionSpotting/CALF")
    model_name: str = "CALF_benchmark"
    preprocessing_patch_marker: str = "tryolab_autopatch"


class SoccerNetPredictor:
    """
    Wrapper that orchestrates running the SoccerNet CALF inference pipeline.

    The predictor expects that the SoccerNet repository has been cloned locally
    (following the same layout as in the notebook) and that all required third-party
    dependencies have been installed manually by the user. The class provides
    guardrails and helpful error messages so the integration can fail fast when
    the runtime environment is not ready.
    """

    def __init__(self, config: Optional[SoccerNetPredictorConfig] = None) -> None:
        self.config = config or SoccerNetPredictorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def repo_dir(self) -> Path:
        return (self.config.root_dir / self.config.repo_subdir).resolve()

    def generate_predictions(self, video_path: Path) -> Path:
        """
        Run SoccerNet inference for ``video_path`` and return the generated JSON file.

        Parameters
        ----------
        video_path: Path
            Path to the input video that should be analysed by SoccerNet.

        Returns
        -------
        Path
            Location of the copied predictions JSON file within ``config.output_dir``.
        """
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise SoccerNetPredictorError(f"Input video not found: {video_path}")

        self.logger.info("Preparing SoccerNet repository at %s", self.repo_dir)
        self._ensure_repo_ready()
        self._ensure_preprocessing_patch()

        # Run inference
        command = [
            sys.executable,
            "inference/main.py",
            "--video_path",
            str(video_path),
            "--model_name",
            self.config.model_name,
        ]

        self.logger.info("Running SoccerNet CALF inference...")
        start = time.perf_counter()
        try:
            subprocess.run(
                command,
                cwd=self.repo_dir,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        except FileNotFoundError as exc:
            raise SoccerNetPredictorError(
                "Could not execute SoccerNet inference. Ensure Python dependencies "
                "are installed and the repository is available as expected."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise SoccerNetPredictorError(
                "SoccerNet inference command failed. Inspect the console output above "
                "for details."
            ) from exc
        finally:
            elapsed = time.perf_counter() - start
            self.logger.info("SoccerNet inference finished in %.2fs", elapsed)

        predictions_path = self._extract_predictions(video_path)

        target_dir = (self.config.output_dir / video_path.stem).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / predictions_path.name

        shutil.copy2(predictions_path, target_path)
        self.logger.info("Copied SoccerNet predictions to %s", target_path)

        return target_path

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _ensure_repo_ready(self) -> None:
        inference_script = self.repo_dir / "inference" / "main.py"
        if not inference_script.exists():
            raise SoccerNetPredictorError(
                "SoccerNet repository not found.\n"
                "Expected to find the CALF code at:\n"
                f"  {self.repo_dir}\n\n"
                "Please clone https://github.com/SilvioGiancola/SoccerNetv2-DevKit "
                "into the soccernet/ folder (same layout as in the provided notebook) "
                "and install the dependencies listed there before re-running."
            )

    def _ensure_preprocessing_patch(self) -> None:
        """
        Apply the defensive copy patch described in the notebook when necessary.
        """
        preprocessing = self.repo_dir / "inference" / "preprocessing.py"
        if not preprocessing.exists():
            return

        marker = f"# {self.config.preprocessing_patch_marker}"
        content = preprocessing.read_text(encoding="utf-8")
        if marker in content:
            return  # Already patched

        updated_lines = []
        for line in content.splitlines():
            if "timestamps_long[0:chunk_size-receptive_field]" in line:
                updated_lines.append(marker)
                updated_lines.append(
                    "            copy_end = min(chunk_size-receptive_field, "
                    "video_size, tmp_timestamps.size(0))"
                )
                updated_lines.append(
                    "            timestamps_long[0:copy_end] = "
                    "tmp_timestamps[0:copy_end]"
                )
                continue
            if "segmentation_long[0:chunk_size-receptive_field]" in line:
                updated_lines.append(marker)
                updated_lines.append(
                    "            copy_end = min(chunk_size-receptive_field, "
                    "video_size, tmp_segmentation.size(0))"
                )
                updated_lines.append(
                    "            segmentation_long[0:copy_end] = "
                    "tmp_segmentation[0:copy_end]"
                )
                continue
            updated_lines.append(line)

        preprocessing.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
        self.logger.info("Patched preprocessing.py to avoid out-of-range assignments.")

    def _extract_predictions(self, video_path: Path) -> Path:
        """
        Look for the predictions JSON emitted by CALF and return its path.
        """
        output_roots = [
            self.repo_dir / "inference" / "output",
            self.repo_dir / "inference" / "outputs",
        ]
        candidates: list[Path] = []

        for root in output_roots:
            if root.exists():
                candidates.extend(path for path in root.rglob("*.json"))

        if not candidates:
            raise SoccerNetPredictorError(
                "SoccerNet inference finished but no predictions JSON was found. "
                "Please verify the inference script output directory."
            )

        video_stem = video_path.stem

        # Prefer JSON files located inside a directory named after the video stem
        for root in output_roots:
            stem_dir = root / video_stem
            if stem_dir.exists():
                stem_candidates = sorted(
                    stem_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
                )
                if stem_candidates:
                    return stem_candidates[0]

        # Fall back to the most recent JSON file across all outputs
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]


def load_predictions_json(json_path: Path) -> list[dict]:
    """
    Convenience helper that loads the raw predictions list from a SoccerNet JSON file.
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise SoccerNetPredictorError(
            f"Unexpected SoccerNet predictions format in {json_path}"
        )
    return predictions


