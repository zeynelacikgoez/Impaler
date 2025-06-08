"""Fitness function for evolutionary simulations."""

from typing import Dict, Any

import numpy as np


DEFAULT_WEIGHTS: Dict[str, float] = {
    "welfare": 1.5,
    "sustainability": 1.2,
    "stability": 0.8,
    "resilience": 1.0,
}


def calculate_fitness(simulation_results: Dict[str, Any], weights: Dict[str, float] | None = None) -> float:
    """Calculate a fitness score for a finished simulation.

    Parameters
    ----------
    simulation_results:
        Result dictionary produced by ``DataCollector.export_data()``.
    weights:
        Optional weighting for the fitness components.

    Returns
    -------
    float
        The resulting fitness score. Higher is better.
    """

    if weights is None:
        weights = DEFAULT_WEIGHTS

    model_data = simulation_results.get("model_data", [])
    if not model_data:
        return -1000.0

    start_index = len(model_data) // 2
    relevant_data = model_data[start_index:]
    if not relevant_data:
        return -1000.0

    def _extract(path: str, default: float = 0.0) -> list[float]:
        res = []
        for entry in relevant_data:
            current = entry
            for part in path.split("."):
                if not isinstance(current, dict):
                    current = {}
                    break
                current = current.get(part, {})
            if isinstance(current, (int, float)):
                res.append(float(current))
            else:
                res.append(default)
        return res

    satisfactions = _extract("welfare.avg_satisfaction", 0.0)
    satisfaction_ginis = _extract("welfare.satisfaction_gini", 1.0)
    sustainability_indices = _extract("environmental.sustainability_index", 0.0)
    total_outputs = _extract("production_metrics.total_output", 0.0)

    avg_satisfaction = np.mean(satisfactions)
    avg_fairness = 1.0 - np.mean(satisfaction_ginis)
    sustainability_score = np.mean(sustainability_indices)

    if np.mean(total_outputs) > 1e-6:
        coeff_of_variation = np.std(total_outputs) / np.mean(total_outputs)
        stability_score = np.exp(-2.0 * coeff_of_variation)
    else:
        stability_score = 0.0

    resilience_score = np.min(satisfactions) if satisfactions else 0.0

    welfare_score = avg_satisfaction * avg_fairness

    fitness = (
        weights.get("welfare", 1.0) * welfare_score
        + weights.get("sustainability", 1.0) * sustainability_score
        + weights.get("stability", 1.0) * stability_score
        + weights.get("resilience", 1.0) * resilience_score
    )

    if sustainability_score < 0.2 or resilience_score < 0.1:
        fitness -= 5.0

    return float(fitness)
