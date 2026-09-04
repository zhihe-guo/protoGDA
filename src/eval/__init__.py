from src.eval.metrics import compute_metrics, format_metrics

# Backwards-compat alias (some callers used this name).
evaluate_predictions = compute_metrics

__all__ = ["compute_metrics", "format_metrics", "evaluate_predictions"]
