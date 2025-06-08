# tests/test_fitness.py
"""Tests for the calculate_fitness function."""

import pytest
from impaler.evolution.fitness import calculate_fitness


def test_calculate_fitness_basic():
    """Ensure fitness is computed from valid simulation results."""
    model_data = [
        {
            "welfare": {"avg_satisfaction": 0.7, "satisfaction_gini": 0.2},
            "environmental": {"sustainability_index": 0.5},
            "production_metrics": {"total_output": 100},
        },
        {
            "welfare": {"avg_satisfaction": 0.8, "satisfaction_gini": 0.2},
            "environmental": {"sustainability_index": 0.6},
            "production_metrics": {"total_output": 110},
        },
        {
            "welfare": {"avg_satisfaction": 0.9, "satisfaction_gini": 0.2},
            "environmental": {"sustainability_index": 0.7},
            "production_metrics": {"total_output": 120},
        },
        {
            "welfare": {"avg_satisfaction": 1.0, "satisfaction_gini": 0.2},
            "environmental": {"sustainability_index": 0.8},
            "production_metrics": {"total_output": 130},
        },
    ]

    result = {"model_data": model_data}

    score = calculate_fitness(result, {"welfare": 1, "sustainability": 1, "stability": 0.5, "resilience": 0.5})

    assert isinstance(score, float)
    assert score == pytest.approx(2.42156, rel=1e-3)


def test_calculate_fitness_missing_data():
    """If model_data is missing, a heavy penalty is returned."""
    assert calculate_fitness({}, {}) == -1000.0
