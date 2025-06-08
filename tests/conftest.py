# tests/conftest.py
"""
Globale Pytest-Konfiguration und Fixture-Definitionen.

'conftest.py' wird von Pytest automatisch eingelesen. 
Hier kannst du gemeinsam genutzte Fixtures definieren, 
die in allen Testdateien verfügbar sind.
"""

import sys
import os
import pytest
import random
import numpy as np

# Ensure the project root is on the Python path so that ``impaler`` can be
# imported when tests are executed from within the ``tests`` directory.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from impaler.core.model import EconomicModel
from impaler.core.config import SimulationConfig, PlanningPriorities, ADMMConfig
from impaler.vsm.system3 import System3Manager
from unittest.mock import MagicMock
import logging

@pytest.fixture(scope="session", autouse=False)
def set_global_seed():
    """
    Optional: Globaler Seed-Setter für reproduzierbare Tests.

    scope="session" bedeutet, dass dies maximal einmal pro Testlauf
    ausgeführt wird. Du könntest dies auch weglassen oder anpassen.
    """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    # Falls du weitere Zufalls-Generatoren nutzt (z.B. torch), 
    # könntest du sie hier ebenfalls seeden.

@pytest.fixture
def small_model():
    """
    Stellt ein EconomicModel bereit, das für kleine Tests geeignet ist:
    - 2 Producer
    - 1 Region
    - 5 Simulationsschritte
    """
    config = SimulationConfig(
        agent_counts={"producers": 2, "regions": 1},
        simulation_steps=5,
        vsm_on=False,
        admm_on=False
    )
    return EconomicModel(config=config)

@pytest.fixture
def medium_model_vsm_admm():
    """
    Stellt ein EconomicModel mit mittlerer Größe bereit:
    - 5 Producer
    - 2 Regionen
    - 10 Simulationsschritte
    - VSM & ADMM aktiv
    """
    config = SimulationConfig(
        agent_counts={"producers": 5, "regions": 2},
        simulation_steps=10,
        vsm_on=True,
        admm_on=True
    )
    return EconomicModel(config=config)

@pytest.fixture
def custom_model_factory():
    """
    Ein Factory-Fixture, um flexibel Modelle zu erzeugen.
    Z.B. test_custom_model = custom_model_factory(producers=10, regions=3, steps=20)
    """
    def _factory(
        producers=2,
        regions=1,
        steps=5,
        vsm=False,
        admm=False,
        *,
        config=None,
        config_dict=None,
        **extra,
    ):
        """Erzeugt ein EconomicModel je nach uebergebenen Parametern.

        Die Tests verwenden teilweise die alte ``config_dict`` API.  Um die
        Rueckwaertskompatibilitaet sicherzustellen, erlaubt diese Factory sowohl
        das direkte Uebergeben eines ``SimulationConfig`` Objektes als auch eines
        Dictionarys.
        """
        if config is not None:
            cfg = config
        elif config_dict is not None:
            cfg = SimulationConfig.parse_obj(config_dict)
        else:
            # Support alternate keyword names used in some tests
            vsm_flag = extra.pop("vsm_on", vsm)
            admm_flag = extra.pop("admm_on", admm)
            cfg = SimulationConfig(
                simulation_steps=steps,
                vsm_on=vsm_flag,
                admm_on=admm_flag,
            )
        return EconomicModel(config=cfg)
    return _factory


@pytest.fixture
def s3_manager():
    """Provide a basic System3Manager instance with a mocked model."""
    mock_model = MagicMock(spec=EconomicModel)
    mock_model.current_step = 0
    mock_model.logger = logging.getLogger("MockEconomicModelS3")
    mock_model.logger.getChild = lambda name: logging.getLogger(f"MockModelS3.{name}")
    mock_model.config = SimulationConfig(
        planning_priorities=PlanningPriorities(goods_priority={"A": 1.0}),
        admm_config=ADMMConfig(),
        system3_params={
            "fairness_weight": 0.5,
            "stability_factor": 0.7,
            "enable_advanced_optimization": True,
            "enable_adaptive_coordination": True,
            "max_feedback_cycles": 3,
            "audit_active": True,
            "conflict_resolution_strategies": ["proportional", "priority_based", "bargaining"],
            "initial_conflict_strategy": "proportional",
        }
    )
    mock_model.system4planner = MagicMock()
    mock_model.route_directive = MagicMock(return_value=True)
    return System3Manager(mock_model)
