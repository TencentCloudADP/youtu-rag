import re

from ..data import EvaluationSample as Datapoint
from .base_llm_processor import BaseLLMJudgeProcesser


class XBenchProcesser(BaseLLMJudgeProcesser):
    """Processer for XBench evaluation."""

    name: str = "XBench"

    def calculate_metrics(self, samples: list[Datapoint]) -> dict:
        """Calculate metrics for XBench evaluation."""
        # 1. calculate level metrics
        level_bin = {}
        invalid_count = 0
        for item in samples:
            level = item.level
            if level not in level_bin:
                level_bin[level] = {"correct": 0, "wrong": 0, "unknown": 0}
            if item.judged_response == "invalid":
                level_bin[level]["unknown"] += 1
                invalid_count += 1
                continue
            if item.correct:
                level_bin[level]["correct"] += 1
            else:
                level_bin[level]["wrong"] += 1
        # calculate overall metrics
        for _, counts in level_bin.items():
            total = counts["correct"] + counts["wrong"]
            if total > 0:
                counts["accuracy"] = round(counts["correct"] / total * 100, 4)
            else:
                counts["accuracy"] = 0.0
        # 2. calculate overall accuracy
        total = len(samples)
        correct_count = sum(item.correct for item in samples)
        incorrect_count = total - correct_count - invalid_count

        # 3. Calculate confidence metrics
        for item in samples:
            if item.confidence is None:
                item.confidence = 100 if item.judged_response == "Exact match" else 0
        confidence_scores = [item.confidence for item in samples if item.judged_response != "invalid"]

        return {
            "Accuracy (%)": round(correct_count / total * 100, 2),
            "Average Confidence (%)": round(sum(confidence_scores) / total, 2),
            "Details": {
                "correct": correct_count,
                "wrong": incorrect_count,
                "unknown": invalid_count,
                "total": total,
                "level_metrics": level_bin,
            },
        }

    def _parse_judge_response(self, response: str) -> dict:
        """Parse the judge response into a structured format."""
        pattern = re.compile(
            r"(?=.*?最终答案:\s*(?P<extracted_final_answer>.*?)(?=\n\s*\w+:|$))?"
            r"(?=.*?解释:\s*(?P<reasoning>.*?)(?=\n\s*\w+:|$))?"
            r"(?=.*?结论:\s*(?P<correct>.*?)(?=\n\s*\w+:|$))?",
            re.DOTALL,
        )
        # remove the bold formatting
        response = response.replace("**", "")
        match = pattern.search(response)
        if not match:
            raise ValueError("Invalid judge response format.")

        return {
            "extracted_final_answer": match.group("extracted_final_answer").strip()
            if match.group("extracted_final_answer")
            else "",
            "reasoning": match.group("reasoning").strip() if match.group("reasoning") else "",
            "correct": match.group("correct").strip() == "正确" if match.group("correct") else False,
        }

    def _extract_exact_answer(self, response: str) -> str:
        """Earse the exact answer from the response."""
        pattern = re.compile(r"最终答案:\s*(.*)")
        match = pattern.search(response)
        if not match or not match.group(1):
            return ""
        return match.group(1).strip()
