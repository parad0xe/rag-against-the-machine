import os

from src.config import settings
from src.domain.models.dataset import MinimalSource


class Evaluator:
    def calculate_recall(
        self,
        retrieved_sources: list[MinimalSource],
        expected_sources: list[MinimalSource],
    ) -> float:
        if not expected_sources:
            return 0.0

        found_count = sum(
            1
            for expected_source in expected_sources
            if self._is_source_found(expected_source, retrieved_sources)
        )

        if len(expected_sources) > 1:
            return float(found_count)

        return found_count / len(expected_sources)

    def _is_source_found(
        self,
        expected_source: MinimalSource,
        retrieved_sources: list[MinimalSource],
    ) -> bool:
        expected_length = self._compute_source_length(expected_source)
        if expected_length == 0:
            return False

        expected_path = os.path.normpath(expected_source.file_path)

        for retrieved_source in retrieved_sources:
            retrieved_path = os.path.normpath(retrieved_source.file_path)

            if not (
                retrieved_path.endswith(expected_path)
                or expected_path.endswith(retrieved_path)
            ):
                continue

            retrieved_length = self._compute_source_length(retrieved_source)
            if retrieved_length == 0:
                continue

            intersection_length = self._compute_intersection_length(
                expected_source, retrieved_source
            )
            union_length = (
                expected_length + retrieved_length - intersection_length
            )

            if union_length > 0:
                ratio = intersection_length / union_length
                if ratio >= settings.overlap_threshold:
                    return True

        return False

    def _compute_source_length(self, source: MinimalSource) -> int:
        return max(
            0, source.last_character_index - source.first_character_index
        )

    def _compute_intersection_length(
        self,
        expected_source: MinimalSource,
        retrieved_source: MinimalSource,
    ) -> int:
        start_index = max(
            expected_source.first_character_index,
            retrieved_source.first_character_index,
        )
        end_index = min(
            expected_source.last_character_index,
            retrieved_source.last_character_index,
        )
        return max(0, end_index - start_index)
