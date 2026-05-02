import logging
import os

from src.domain.models.dataset import MinimalSource

logger = logging.getLogger(__file__)


class EvaluatorService:
    """
    Service to calculate evaluation metrics like recall for search results.
    """

    def __init__(self, overlap_threshold: float = 0.05) -> None:
        """
        Initializes the evaluator with an overlap threshold.

        Args:
            overlap_threshold: The minimum IoU (Intersection over Union)
                ratio to consider a retrieved source as matching an
                expected source.
        """
        self._overlap_threshold = overlap_threshold

    def calculate_recall(
        self,
        retrieved_sources: list[MinimalSource],
        expected_sources: list[MinimalSource],
    ) -> float:
        """
        Calculates the recall for a set of retrieved sources.

        Args:
            retrieved_sources: Sources returned by the retriever.
            expected_sources: Ground truth sources.

        Returns:
            The recall score as a float.
        """
        if not expected_sources:
            logger.debug("No expected sources provided. Recall is 0.0.")
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
        """
        Checks if an expected source is present in the retrieved sources.

        Args:
            expected_source: The ground truth source to find.
            retrieved_sources: List of sources to search within.

        Returns:
            True if the source is found, False otherwise.
        """
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
                if ratio >= self._overlap_threshold:
                    logger.debug(
                        f"Match found: {expected_path} "
                        f"(overlap ratio={ratio:.2f})."
                    )
                    return True

        return False

    def _compute_source_length(self, source: MinimalSource) -> int:
        """
        Computes the character length of a source.

        Args:
            source: The source to measure.

        Returns:
            The number of characters in the source range.
        """
        return max(
            0, source.last_character_index - source.first_character_index
        )

    def _compute_intersection_length(
        self,
        expected_source: MinimalSource,
        retrieved_source: MinimalSource,
    ) -> int:
        """
        Computes the length of the intersection between two sources.

        Args:
            expected_source: The first source.
            retrieved_source: The second source.

        Returns:
            The number of overlapping characters.
        """
        start_index = max(
            expected_source.first_character_index,
            retrieved_source.first_character_index,
        )
        end_index = min(
            expected_source.last_character_index,
            retrieved_source.last_character_index,
        )
        return max(0, end_index - start_index)
