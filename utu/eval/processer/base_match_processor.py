import re
import string

from ..data import EvaluationSample
from .base_llm_processor import BaseLLMJudgeProcesser


class BaseMatchProcesser(BaseLLMJudgeProcesser):
    """Base class for processers that use match-based judging."""

    async def judge_one(self, data: EvaluationSample) -> EvaluationSample:
        """Judge a single sample."""
        # question = data.raw_question
        response = data.response
        correct_answer = data.correct_answer or "unknown"

        if_correct = False
        # if gt is a number
        if self._is_float(correct_answer):
            normalized_answer = self._normalize_number_str(str(response))
            if_correct = normalized_answer == float(correct_answer)

        # if gt is a list
        elif any(char in correct_answer for char in [",", ";"]):
            # question with the fish: normalization removes punct

            gt_elems = self._split_string(correct_answer)
            ma_elems = self._split_string(response)

            # check length is the same
            if len(gt_elems) != len(ma_elems):
                data.update(correct=False)
                return data

            # compare each element as float or str
            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems, strict=False):
                if self._is_float(gt_elem):
                    normalized_ma_elem = self._normalize_number_str(ma_elem)
                    comparisons.append(normalized_ma_elem == float(gt_elem))
                else:
                    # we do not remove punct since comparisons can include punct
                    comparisons.append(
                        self._normalize_str(ma_elem, remove_punct=False)
                        == self._normalize_str(gt_elem, remove_punct=False)
                    )
            if_correct = all(comparisons)

        # if gt is a str
        else:
            if_correct = self._normalize_str(response) == self._normalize_str(correct_answer)

        data.update(correct=if_correct)
        return data

    def _is_float(self, s: str) -> bool:
        """Check if a string is a float."""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _normalize_number_str(self, s: str) -> float:
        """Normalize a number string to a float."""
        for char in ["$", "%", ","]:
            s = s.replace(char, "")
        try:
            return float(s)
        except ValueError:
            print(f"String {s} cannot be normalized to number str.")
            return float("inf")

    def _split_string(self, s: str, char_list: list[str] = None) -> list[str]:
        """Split a string by a list of characters."""
        if char_list is None:
            char_list = [",", ";"]
        pattern = f"[{''.join(char_list)}]"
        return re.split(pattern, s)

    def _normalize_str(self, s: str, remove_punct: bool = True) -> str:
        """
        Normalize a string by:
        - Removing all white spaces
        - Optionally removing punctuation (if remove_punct is True)
        - Converting to lowercase

        Parameters:
        - input_str: str, the string to normalize
        - remove_punct: bool, whether to remove punctuation (default: True)
        Returns:
        - str, the normalized string
        """
        # Remove all white spaces. Required e.g for seagull vs. sea gull
        no_spaces = re.sub(r"\s", "", s)

        # Remove punctuation, if specified.
        if remove_punct:
            translator = str.maketrans("", "", string.punctuation)
            return no_spaces.lower().translate(translator)
        else:
            return no_spaces.lower()
