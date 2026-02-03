from ..data import EvaluationSample


class MetricsUtils:
    @staticmethod
    def calculate_overall_metrics(samples: list[EvaluationSample]) -> dict:
        """calculate overall metrics"""
        invalid_count = 0
        for item in samples:
            if item.judged_response == "invalid":
                invalid_count += 1
        total = len(samples)
        correct_count = sum(item.correct for item in samples)
        incorrect_count = total - correct_count - invalid_count
        # confidence_scores = [item.confidence for item in samples if item.judged_response != "invalid"]
        return {
            "Accuracy (%)": round(correct_count / total * 100, 2),
            # "Average Confidence (%)": round(sum(confidence_scores) / total, 2),
            "Details": {
                "correct": correct_count,
                "wrong": incorrect_count,
                "unknown": invalid_count,
                "total": total,
            },
        }

    @staticmethod
    def calculate_level_metrics(samples: list[EvaluationSample]) -> dict:
        """calculate level metrics"""
        level_bin = {}
        for item in samples:
            level = item.level
            if level not in level_bin:
                level_bin[level] = {"correct": 0, "wrong": 0, "unknown": 0}
            if item.judged_response == "invalid":
                level_bin[level]["unknown"] += 1
                continue
            if item.correct:
                level_bin[level]["correct"] += 1
            else:
                level_bin[level]["wrong"] += 1
        for _, counts in level_bin.items():
            total = counts["correct"] + counts["wrong"]
            if total > 0:
                counts["accuracy"] = round(counts["correct"] / total * 100, 4)
            else:
                counts["accuracy"] = 0.0
        return {
            "level_metrics": level_bin,
        }

    @staticmethod
    def calculate_calibration(
        samples: list[EvaluationSample],
        confidence_bins: list[tuple[int, int]] = None,
    ) -> dict:
        """Calculate calibration statistics"""
        confidence_bins = confidence_bins or [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]
        calibration = [{"samples": 0, "correct": 0, "conf_sum": 0} for _ in confidence_bins]
        for record in samples:
            if record.judged_response == "invalid":
                continue
            confidence = record.confidence or 0
            bin_idx = min(confidence // 20, len(confidence_bins) - 1)
            bin_stats = calibration[bin_idx]
            bin_stats["samples"] += 1
            bin_stats["conf_sum"] += confidence
            if record.get("correct", False):
                bin_stats["correct"] += 1
        calibration_error = round(MetricsUtils._calculate_calibration(calibration, len(samples)), 2)
        return {
            "Calibration Error (%)": calibration_error,
        }

    @staticmethod
    def _calculate_calibration(stats: list[dict], total: int) -> float:
        """calculate calibration statistics"""
        error = 0.0
        for bin_stats in stats:
            samples = bin_stats["samples"]
            if not samples:
                continue
            accuracy = bin_stats["correct"] / samples
            avg_conf = bin_stats["conf_sum"] / samples / 100  # convert to 0-1 decimal
            error += (samples / total) * abs(accuracy - avg_conf)
        return error * 100  # convert to percentage
