import os

from src.domain.models.source import MinimalSource


class Evaluator:
    def calculate_recall(
        self,
        retrieved: list[MinimalSource],
        expected: list[MinimalSource],
    ) -> float:
        if not expected:
            return 0.0

        found = 0
        for exp in expected:
            exp_len = exp.last_character_index - exp.first_character_index
            if exp_len <= 0:
                continue

            exp_n = os.path.normpath(exp.file_path)

            is_found = False
            for ret in retrieved:
                ret_n = os.path.normpath(ret.file_path)

                if not ret_n.endswith(exp_n) and not exp_n.endswith(ret_n):
                    continue

                start = max(
                    ret.first_character_index,
                    exp.first_character_index,
                )
                end = min(
                    ret.last_character_index,
                    exp.last_character_index,
                )
                overlap = max(0, end - start)

                if (overlap / exp_len) >= 0.05:
                    is_found = True
                    break

            if is_found:
                found += 1

        if len(expected) > 1:
            return float(found)

        return found / len(expected)
