# tests/test_config_loading.py
"""
Tests für das Laden, Validieren und Serialisieren der SimulationConfig
unter Verwendung von Pydantic.

Stellt sicher, dass die Konfiguration korrekt aus verschiedenen Quellen
(Dict, JSON) geladen wird, Standardwerte angewendet, Typen und Constraints
validiert werden und Fehler bei ungültigen Eingaben korrekt ausgelöst werden.
"""

import pytest
import json
import os
from typing import Dict, Any, List

# Importiere Pydantic Fehler und Konfigurationsklassen (Pfade ggf. anpassen)
from pydantic import ValidationError
from impaler.core.config import (
    SimulationConfig,
    DemandConfig,
    ScenarioConfig,
    IOConfig,
    EnvironmentConfig,
    PlanningPriorities,
    RegionalConfig,
    ADMMConfig,
    AgentPopulationConfig,
    SpecificAgentConfig,
    ResourceShockModel,  # Beispiel für verschachteltes Modell
    ProductionPath,      # Beispiel für verschachteltes Modell
    ProductionPathInput, # Beispiel für verschachteltes Modell
)
from impaler.core.config_models import PlanwirtschaftParams

# --- Testdaten Fixtures ---

@pytest.fixture
def valid_config_dict() -> Dict[str, Any]:
    """Liefert ein gültiges Konfigurations-Dictionary."""
    return {
        "simulation_name": "Valid_Test_Run",
        "goods": ["Food", "Tools", "Services"],
        "resources": ["Energy", "Metal", "Biomass"],
        "simulation_steps": 50,
        "random_seed": 123,
        "vsm_on": True,
        "admm_on": True,
        "regional_config": {
            "regions": ["North", "South"],
            "transport_costs": { # Verschachteltes Dict statt Tuple-Key
                "North": {"South": {"Food": 0.1, "Metal": 0.05}},
                "South": {"North": {"Food": 0.1, "Metal": 0.05}}
            },
             "transport_capacity": {
                "North": {"South": 500.0},
                "South": {"North": 500.0}
            }
        },
        "environment_config": {
            "resource_capacities": {"Biomass": 10000, "Metal": 5000},
            "emission_factors": {"Tools": {"co2": 2.5}},
            "sustainability_targets": {"co2": {30: 800}}
        },
        "admm_config": {
            "rho": 0.15,
            "tolerance": 1e-4
        },
        "planning_priorities": {
            "goods_priority": {"Food": 1.0, "Tools": 0.7},
            "fairness_weight": 1.2
        },
        "io_config": {
            "initial_io_matrix": {
                "Food": [ # Liste von Pfaden
                    ProductionPath(input_requirements=[ProductionPathInput(resource_type="Biomass", amount_per_unit=0.8),
                                                       ProductionPathInput(resource_type="Energy", amount_per_unit=0.2)]),
                ],
                "Tools": ProductionPath( # Einzelner Pfad
                    input_requirements=[ProductionPathInput(resource_type="Metal", amount_per_unit=1.5),
                                        ProductionPathInput(resource_type="Energy", amount_per_unit=0.5)]
                )
            }
        },
        "scenario_config": {
            "resource_shocks": [
                ResourceShockModel(
                    step=20, resource="Energy", relative_change=-0.3, duration=10,
                    name="Energy Shortage", description="Test shock"
                ).dict() # Konvertiere zu Dict für Input
            ]
        },
        "agent_populations": {
            "farmers": {
                "agent_class": "ProducerAgent",
                "count": 5,
                "id_prefix": "farmer_",
                "params": {"initial_capacity": (50, 80), "production_lines": []}, # Beispiel
                "region_distribution": {0: 1.0} # Annahme: Region 0 = North
            }
        },
        "specific_agents": [
            {
                "agent_class": "ProducerAgent",
                "unique_id": "ToolFactory_S",
                "params": {"region_id": 1, "initial_capacity": 500, "production_lines": []} # Annahme: Region 1 = South
            }
        ],
         "logging_config": {
             "log_level": "DEBUG"
         }
    }

@pytest.fixture
def minimal_config_dict() -> Dict[str, Any]:
    """Liefert ein minimales Config-Dict, das auf Defaults angewiesen ist."""
    # Enthält nur Felder ohne Defaults in SimulationConfig oder Sub-Modellen
    # (Aktuell hat SimulationConfig selbst keine Felder ohne Defaults)
    # Wir brauchen aber ggf. innere Felder für Validierung
    return {
        "goods": ["Basic"],
        "resources": ["Material"],
        "regional_config": {"regions": ["Center"]},
        "agent_populations": { # Erforderlich, damit Agenten erstellt werden können
             "basic_producer": {
                 "agent_class": "ProducerAgent", # Erforderlich in AgentPopulationConfig
                 "count": 1,                    # Erforderlich in AgentPopulationConfig
                 "params": {}                   # Kann leer sein, wenn ProducerAgent Defaults hat
             }
        },
        "io_config": { # Muss ggf. für Producer definiert sein
             "initial_io_matrix": {
                  "Basic": ProductionPath(input_requirements=[ProductionPathInput(resource_type="Material", amount_per_unit=1.0)]).dict() # Zu Dict konvertieren
             }
        }

    }

@pytest.fixture
def invalid_type_config_dict(valid_config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Liefert ein Config-Dict mit einem Typfehler."""
    invalid_dict = valid_config_dict.copy()
    invalid_dict["simulation_steps"] = "fifty" # String statt int
    return invalid_dict

@pytest.fixture
def missing_required_config_dict(valid_config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Liefert ein Config-Dict, dem ein erforderliches Feld fehlt."""
    invalid_dict = valid_config_dict.copy()
    # Entferne 'agent_class' aus der Population (erforderlich in AgentPopulationConfig)
    if "farmers" in invalid_dict["agent_populations"]:
        del invalid_dict["agent_populations"]["farmers"]["agent_class"]
    return invalid_dict

@pytest.fixture
def invalid_value_config_dict(valid_config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Liefert ein Config-Dict, das einen Constraint verletzt."""
    invalid_dict = valid_config_dict.copy()
    # Setze ungültigen Wert (z.B. negative Schritte oder Wahrscheinlichkeit > 1)
    invalid_dict["simulation_steps"] = 0 # Field erfordert ge=1
    if "scenario_config" in invalid_dict:
         invalid_dict["scenario_config"]["random_events_probability"] = 1.5 # Field erfordert le=1.0
    return invalid_dict

@pytest.fixture
def config_with_extra_field(valid_config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Liefert ein Config-Dict mit einem undefinierten Feld."""
    config = valid_config_dict.copy()
    config["unknown_parameter"] = "some_value"
    return config


# --- Test Klasse ---

class TestSimulationConfigLoading:

    def test_load_valid_dict(self, valid_config_dict: Dict[str, Any]):
        """Testet das Laden einer gültigen Konfiguration aus einem Dictionary."""
        try:
            config = SimulationConfig.from_dict(valid_config_dict)
            assert isinstance(config, SimulationConfig)
            # Prüfe einige Werte und verschachtelte Modelle
            assert config.simulation_name == "Valid_Test_Run"
            assert config.simulation_steps == 50
            assert isinstance(config.regional_config, RegionalConfig)
            assert config.regional_config.regions == ["North", "South"]
            assert isinstance(config.agent_populations["farmers"], AgentPopulationConfig)
            assert config.agent_populations["farmers"].count == 5
            assert isinstance(config.specific_agents[0], SpecificAgentConfig)
            assert config.specific_agents[0].unique_id == "ToolFactory_S"
            assert isinstance(config.scenario_config.resource_shocks[0], ResourceShockModel)
            assert config.scenario_config.resource_shocks[0].resource == "Energy"
            assert config.admm_config.rho == 0.15
            # Prüfe Transportkosten-Struktur
            assert "North" in config.regional_config.transport_costs
            assert "South" in config.regional_config.transport_costs["North"]
            assert config.regional_config.transport_costs["North"]["South"]["Food"] == 0.1

        except ValidationError as e:
            pytest.fail(f"Validierung fehlgeschlagen bei gültigem Dict: {e}")

    def test_load_minimal_dict_uses_defaults(self, minimal_config_dict: Dict[str, Any]):
        """Testet, ob beim Laden eines minimalen Dicts korrekte Defaults gesetzt werden."""
        config = SimulationConfig.from_dict(minimal_config_dict)
        # Prüfe einige Default-Werte
        assert config.simulation_steps == 100 # Default aus SimulationConfig
        assert config.vsm_on is True
        assert isinstance(config.admm_config, ADMMConfig)
        assert config.admm_config.rho == 0.1 # Default aus ADMMConfig
        assert isinstance(config.demand_config, DemandConfig)
        assert config.demand_config.population_growth_rate == 0.01 # Default
        assert len(config.specific_agents) == 0 # Default ist leere Liste
        assert config.performance_profile == "balanced"

    def test_load_config_with_planwirtschaft_params(self):
        """Testet das Laden und Validieren der neuen Planwirtschafts-Parameter."""
        config_dict = {
            "goods": ["Food", "Energy"],
            "resources": ["Water"],
            "regional_config": {"regions": ["R1"]},
            "planning_priorities": {
                "fairness_weight": 0.9,
                "planwirtschaft_params": {
                    "underproduction_penalty": 5.0,
                    "co2_penalties": {"Energy": 12.0},
                    "priority_sectors": ["Food"],
                },
            },
        }

        config = SimulationConfig.parse_obj(config_dict)
        pp = config.planning_priorities.planwirtschaft_params
        assert isinstance(pp, PlanwirtschaftParams)
        assert pp.underproduction_penalty == 5.0
        assert pp.co2_penalties["Energy"] == 12.0
        assert pp.priority_sectors == ["Food"]
        assert pp.inventory_cost == 0.05

    def test_load_invalid_type_raises_error(self, invalid_type_config_dict: Dict[str, Any]):
        """Testet, ob ein Typfehler eine ValidationError auslöst."""
        with pytest.raises(ValidationError) as excinfo:
            SimulationConfig.from_dict(invalid_type_config_dict)
        # Prüfe, ob der Fehler das richtige Feld betrifft
        assert "simulation_steps" in str(excinfo.value)
        # Pydantic v2 wording
        assert "Input should be a valid integer" in str(excinfo.value)

    def test_load_missing_required_raises_error(self, missing_required_config_dict: Dict[str, Any]):
        """Testet, ob ein fehlendes erforderliches Feld eine ValidationError auslöst."""
        with pytest.raises(ValidationError) as excinfo:
            SimulationConfig.from_dict(missing_required_config_dict)
        # Prüfe auf den Fehler im verschachtelten Modell
        assert "agent_populations.farmers.agent_class" in str(excinfo.value)
        assert "Field required" in str(excinfo.value)

    def test_load_invalid_value_raises_error(self, invalid_value_config_dict: Dict[str, Any]):
        """Testet, ob ein Wert außerhalb der Constraints eine ValidationError auslöst."""
        with pytest.raises(ValidationError) as excinfo:
            SimulationConfig.from_dict(invalid_value_config_dict)
        # Prüfe auf die beiden Fehler
        error_str = str(excinfo.value)
        assert "simulation_steps" in error_str
        assert "greater than or equal to 1" in error_str
        # Prüfe den anderen Fehler (probability > 1) - Pfad kann komplex sein
        assert "scenario_config.random_events_probability" in error_str
        assert "less than or equal to 1" in error_str


    def test_load_from_valid_json(self, valid_config_dict: Dict[str, Any], tmp_path):
        """Testet das Laden einer gültigen Konfiguration aus einer JSON-Datei."""
        file_path = tmp_path / "valid_config.json"
        with open(file_path, 'w') as f:
            # Konvertiere ProductionPath explizit, da Pydantic v1 .dict() nicht tief genug geht
            temp_dict = valid_config_dict.copy()
            temp_dict["io_config"]["initial_io_matrix"]["Food"] = [p.dict() for p in temp_dict["io_config"]["initial_io_matrix"]["Food"]]
            temp_dict["io_config"]["initial_io_matrix"]["Tools"] = temp_dict["io_config"]["initial_io_matrix"]["Tools"].dict()
            json.dump(temp_dict, f)


        config = SimulationConfig.from_json_file(str(file_path))
        assert isinstance(config, SimulationConfig)
        assert config.simulation_steps == 50
        assert config.regional_config.regions == ["North", "South"]
        # Stichprobe für verschachteltes Modell
        assert isinstance(config.io_config.initial_io_matrix["Tools"], ProductionPath)
        assert len(config.io_config.initial_io_matrix["Food"]) == 1
        assert isinstance(config.io_config.initial_io_matrix["Food"][0], ProductionPath)

    def test_load_from_missing_json_raises_error(self, tmp_path):
        """Testet, ob das Laden einer nicht existierenden JSON-Datei einen Fehler wirft."""
        file_path = tmp_path / "non_existent_config.json"
        with pytest.raises(FileNotFoundError):
            SimulationConfig.from_json_file(str(file_path))

    def test_load_from_invalid_json_raises_error(self, tmp_path):
        """Testet, ob das Laden einer ungültigen JSON-Datei einen Fehler wirft."""
        file_path = tmp_path / "invalid_config.json"
        # Schreibe ungültiges JSON
        with open(file_path, 'w') as f:
            f.write("{ 'simulation_steps': 50, ") # Fehlendes Komma, ungültige Quotes

        with pytest.raises(json.JSONDecodeError):
            SimulationConfig.from_json_file(str(file_path))

    def test_extra_fields_behavior(self, config_with_extra_field: Dict[str, Any]):
        """Testet das Verhalten bei zusätzlichen Feldern basierend auf Config.extra."""
        # Annahme: SimulationConfig.Config.extra = 'allow' (Standard oder explizit gesetzt)
        try:
            config = SimulationConfig.from_dict(config_with_extra_field)
            # Pydantic v2 speichert unbekannte Felder in ``model_extra``
            assert config.model_extra.get('unknown_parameter') == "some_value"
        except ValidationError:
             pytest.fail("ValidationError wurde bei Config.extra='allow' ausgelöst.")

        # Um 'forbid' zu testen:
        class StrictConfig(SimulationConfig):
             class Config:
                  extra = 'forbid'
        with pytest.raises(ValidationError) as excinfo:
             StrictConfig.from_dict(config_with_extra_field)
        assert "unknown_parameter" in str(excinfo.value)
        assert "Extra inputs are not permitted" in str(excinfo.value)


    def test_to_dict_and_to_json(self, valid_config_dict: Dict[str, Any], tmp_path):
        """Testet die Serialisierung nach Dict und JSON."""
        config = SimulationConfig.from_dict(valid_config_dict)

        # Nach Dict
        output_dict = config.to_dict()
        assert isinstance(output_dict, dict)
        assert output_dict["simulation_steps"] == 50
        assert isinstance(output_dict["admm_config"], dict) # Pydantic wandelt Submodelle um
        assert output_dict["admm_config"]["rho"] == 0.15
        # Prüfe verschachtelte Liste von Modellen
        assert isinstance(output_dict["scenario_config"]["resource_shocks"], list)
        assert isinstance(output_dict["scenario_config"]["resource_shocks"][0], dict)
        assert output_dict["scenario_config"]["resource_shocks"][0]["resource"] == "Energy"

        # Nach JSON
        file_path = tmp_path / "output_config.json"
        config.to_json_file(str(file_path))
        assert file_path.exists()

        # Lade zurück und vergleiche (einfacher Check)
        with open(file_path, 'r') as f:
             loaded_data = json.load(f)
        assert loaded_data["simulation_name"] == "Valid_Test_Run"
        assert loaded_data["admm_config"]["tolerance"] == 1e-4

    # Optional: Test für spezifische Validatoren (z.B. den root_validator)
    def test_root_validator_consistency(self, valid_config_dict):
        """Testet den root_validator auf Konsistenz (Beispiel)."""
        # Test 1: Gültig (sollte durchgehen)
        SimulationConfig.from_dict(valid_config_dict) # Kein Fehler erwartet

        # Test 2: Ungültig (Gut in IO nicht in 'goods' definiert)
        invalid_dict = valid_config_dict.copy()
        # Füge ungültigen IO-Eintrag hinzu
        invalid_dict["io_config"]["initial_io_matrix"]["NewGood"] = ProductionPath(input_requirements=[]).dict()
        # In der aktuellen Implementierung werden nur Warnungen geloggt
        # und keine ValidationError ausgelöst
        SimulationConfig.from_dict(invalid_dict)

    def test_apply_performance_profile(self, minimal_config_dict):
        minimal_config_dict["performance_profile"] = "fast_prototype"
        cfg = SimulationConfig.from_dict(minimal_config_dict)
        assert cfg.performance_profile == "fast_prototype"
        assert cfg.admm_config.max_iterations <= 5
