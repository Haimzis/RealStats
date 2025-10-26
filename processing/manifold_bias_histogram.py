import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, List, Set
from tqdm import tqdm

from processing.histograms import BaseHistogram

PATH_BASED_STATISTICS: Set[str] = {"LatentNoiseCriterion_original"}


class PathBasedStatistic(BaseHistogram):
    """Utility base class for statistics that rely on sample metadata instead of pixels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # These statistics do not need a GPU device.
        self.device = torch.device("cpu")

    def preprocess(self, image_batch):
        raise NotImplementedError("PathBasedStatistic does not process image batches.")

    def lookup(self, paths: Iterable[str]) -> List[float]:
        """Return the statistic for each input path."""
        raise NotImplementedError

    def create_histogram(self, data_loader):
        results: Dict[str, float] = {}

        for _, _, paths in tqdm(data_loader, desc="Generating histograms", leave=False):
            scores = self.lookup(paths)
            for path, score in zip(paths, scores):
                results[path] = float(score)

        return results


class LatentNoiseCriterionOriginal(PathBasedStatistic):
    """Loads pre-computed Latent Noise Criterion scores from disk."""

    def __init__(self, scores_csv):
        super().__init__()
        self.scores_csv = Path(scores_csv).expanduser().resolve()
        if not self.scores_csv.exists():
            raise FileNotFoundError(f"Could not find CSV with scores at {self.scores_csv}")

        self._scores = self._load_scores(self.scores_csv)

    def _load_scores(self, csv_path: Path) -> Dict[str, float]:
        df = pd.read_csv(
            csv_path,
            usecols=["image_path", "criterion"],
            dtype={"image_path": str, "criterion": float},
        )
        if "image_path" not in df.columns or "criterion" not in df.columns:
            raise ValueError(
                "LatentNoiseCriterion_original CSV must contain 'image_path' and 'criterion' columns."
            )

        return dict(zip(df["image_path"], df["criterion"]))

    def lookup(self, paths: Iterable[str]) -> List[float]:
        scores: List[float] = []

        for path in paths:
            score = self._scores.get(path)
            if score is None:
                raise KeyError(
                    f"No pre-computed LatentNoiseCriterion score found for '{path}'."
                )

            scores.append(score)

        return scores
