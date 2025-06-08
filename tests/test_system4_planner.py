# tests/test_system4planner.py

import pytest
import numpy as np

# Falls deine Imports ggf. andere Pfade haben, passe sie hier an!
from impaler.core.config import SimulationConfig
from impaler.core.model import EconomicModel
from impaler.vsm.system4 import System4Planner


@pytest.fixture
def example_config():
    """
    Einfaches Fixture, das ein Grund-Config zurückgibt.
    Du kannst hier Parameter einstellen, die für Tests nützlich sind.
    """
    config = SimulationConfig(
        # Beispielwerte anpassen:
        simulation_steps=5,
        agent_counts={"producers": 3, "regions": 1},
        # ADMM-Parameter:
        admm_parameters={
            "lambdas": {"co2": 1.0, "energy": 0.5, "water": 0.3, "fairness": 0.2},
            "resource_limits": {"co2": 200.0, "energy": 400.0, "water": 300.0},
            "min_fairness": 0.7,
            "rho": 0.1,
            "tolerance": 2.0,
            "max_iterations": 5
        }
    )
    return config


@pytest.fixture
def example_model(example_config):
    """
    Stellt ein kleines EconomicModel mit wenigen Produzenten
    und einem System4Planner zur Verfügung.
    """
    model = EconomicModel(config=example_config.to_dict())
    return model


def test_planner_initialization(example_model):
    """
    Testet, ob der System4Planner korrekt im Modell
    vorhanden ist und Initialwerte gesetzt sind.
    """
    planner = example_model.system4planner
    assert planner is not None, "System4Planner sollte im Modell existieren."
    assert isinstance(planner, System4Planner), "Planner muss vom Typ System4Planner sein."

    # Prüfe einige Standardwerte
    assert planner.rho == 0.1, "ADMM-Penalty-Faktor sollte 0.1 sein (laut Config)."
    assert "co2" in planner.lambdas, "Lambda für co2 muss konfiguriert sein."


def test_planning_methods(example_model):
    """
    Testet, ob verschiedene Planungsmethoden (iterative, hierarchical, distributed)
    ohne Fehler durchlaufen.
    """
    planner = example_model.system4planner
    
    # 1) Iterative
    planner.active_planning_method = "iterative"
    planner.planning_methods["iterative"](max_iter=2)
    # Hier könntest du z.B. prüfen, ob Production Targets gesetzt wurden.
    for p in example_model.producers:
        for g in p.can_produce:
            assert p.production_target[g] >= 0.0, "Production Target darf nicht negativ sein."

    # 2) Hierarchical
    planner.active_planning_method = "hierarchical"
    planner.planning_methods["hierarchical"](max_iter=2)
    # Evtl. nochmal checken, ob Targets sich geändert haben
    for p in example_model.producers:
        total_t = sum(p.production_target[g] for g in p.can_produce)
        assert total_t >= 0.0

    # 3) Distributed
    planner.active_planning_method = "distributed"
    planner.planning_methods["distributed"](max_iter=2)
    # Und wieder eine kleine Plausibilitätsprüfung
    for p in example_model.producers:
        # z.B. Kapazität darf nicht unterschritten werden (falls Code das skaliert)
        total_t = sum(p.production_target[g] for g in p.can_produce)
        assert total_t <= p.productive_capacity + 1e-6, "Distributed planning sollte Kapazitäten respektieren."


def test_admm_iteration(example_model):
    """
    Testet den ADMM-Update-Prozess (run_admm_iteration).
    Wir prüfen, ob das ohne Fehler durchläuft und z.B.
    Lagrange-Parameter sich ändern oder in plausiblem Bereich bleiben.
    """
    planner = example_model.system4planner

    # Vorher
    old_lambda_co2 = planner.lambdas["co2"]

    # Führt ADMM-Iterationen durch
    planner.run_admm_iteration(n_iter=2)

    # Beispiel-Checks:
    new_lambda_co2 = planner.lambdas["co2"]
    # Möglicherweise hat sich der CO2-Lambda erhöht, falls Producer "zuviel" Emissionen haben
    # Wir prüfen nur, ob der Prozess durchlief und der Wert >= 0 bleibt.
    assert new_lambda_co2 >= 0.0, "Lambda für CO2 sollte nicht negativ werden."

    # Außerdem kann man checken, ob Producer nun local_solution oder production_target
    # upgedated haben.
    for p in example_model.producers:
        for g in p.can_produce:
            assert p.local_solution[g] >= 0.0, "Lokale ADMM-Lösung sollte >= 0 sein."
            assert p.production_target[g] == p.local_solution[g], (
                "Nach ADMM sollte production_target dem local_solution entsprechen."
            )

    # Optional: Prüfen, ob primal/dual residual geloggt wurde
    # -> Hier bräuchte man evtl. Einblick in admm_convergence_history
    assert len(planner.admm_convergence_history) > 0, "ADMM sollte Infos zu Konvergenz speichern."
    last_entry = planner.admm_convergence_history[-1]
    assert "final_primal_res" in last_entry, "Konvergenzinfos (final_primal_res) fehlen."
    assert last_entry["final_primal_res"] >= 0.0


def test_fed_update(example_model):
    """
    Testet das Federated-Learning-Update (fed_update).
    Prüft, ob global_params sowie Agentenparams aktualisiert werden.
    """
    planner = example_model.system4planner
    before_global_params = planner.global_params.copy()

    # Künstliches "global_vec" simulieren:
    custom_vec = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    planner.fed_update(global_vec=custom_vec)

    # Prüfen, ob global_params sich verändert hat
    after_global_params = planner.global_params
    assert not np.allclose(before_global_params, after_global_params), "global_params sollte sich ändern."
    # Erwartung: gemittelt (0.5 * old + 0.5 * custom_vec)
    expected = 0.5 * before_global_params + 0.5 * custom_vec
    assert np.allclose(expected, after_global_params), "global_params stimmt nicht mit Weighted-Average überein."

    # Produzenten-Parameter checken
    for p in example_model.producers:
        # local_model_params sollte sich angepasst haben
        assert not np.allclose(
            p.local_model_params,
            np.zeros_like(p.local_model_params)
        ), "Local model params sollten nicht mehr auf 0 sein."


@pytest.mark.parametrize("method", ["iterative", "hierarchical", "distributed"])
def test_planner_switch_methods(example_model, method):
    """
    Überprüft, ob System4Planner methodenbezogen anpassbar ist und
    adapt_planning_strategy() ohne Crash läuft.
    """
    planner = example_model.system4planner
    # Manuell setzten
    planner.active_planning_method = method
    # Teste adapt_planning_strategy
    # (das wechselt ggf. autom. bei Ungleichheit, etc.)
    planner.adapt_planning_strategy()
    # Nur check, ob Code durchläuft
    assert planner.active_planning_method in ["iterative", "hierarchical", "distributed"]
    # Evtl. logische Prüfung, ob der Switch plausibel war
