# tests/test_crisismanager_integration.py
"""
Integrationstests für den CrisisManager und seine Auswirkungen auf das EconomicModel.

Diese Tests prüfen, ob das Auslösen von Krisen die erwarteten Effekte im
Modellzustand (Agenten, Ressourcen, VSM-Systeme) hervorruft und ob die
Krisenmechanismen (Start, Update, Ende, Adaption) wie vorgesehen funktionieren.
"""

import pytest
import numpy as np
import math
import logging
from unittest.mock import MagicMock, patch, ANY
from typing import Optional, Union

# Importiere notwendige Klassen (Pfade ggf. anpassen)
from impaler.core.model import EconomicModel
from impaler.core.config import SimulationConfig
from impaler.core.crisismanager import CrisisManager, BaseCrisis
# Importiere spezifische Krisen für Tests (oder mocke sie)
from impaler.core.crisismanager import (
    ResourceShortageCrisis,
    NaturalDisasterCrisis,
    InfrastructureFailureCrisis,
    TechnologicalDisruptionCrisis,
    EnvironmentalCatastropheCrisis
)
# Importiere Agenten für Zustandsprüfungen
from impaler.agents import ProducerAgent, ResourceAgent
# from impaler.agents.infrastructure import InfrastructureAgent # Falls vorhanden


# --- Hilfsfunktionen ---

def run_model_steps(model: EconomicModel, num_steps: int):
    """Führt das Modell für eine gegebene Anzahl Schritte aus."""
    for _ in range(num_steps):
        if not model.step():
            # Simulation vorzeitig beendet (z.B. wegen Fehler oder Endbedingung)
            break

def get_avg_producer_capacity(model: EconomicModel, region_id: Optional[Union[int, str]] = None) -> float:
    """Berechnet die durchschnittliche Kapazität der Producer (optional pro Region)."""
    producers_to_avg = []
    if region_id is None:
        producers_to_avg = getattr(model, 'producers', [])
    else:
        producers_to_avg = getattr(model, 'producers_by_region', {}).get(region_id, [])

    if not producers_to_avg:
        return 0.0
    capacities = [getattr(p, 'productive_capacity', 0.0) for p in producers_to_avg]
    return float(np.mean(capacities))

# TODO: Füge ggf. weitere Helfer hinzu (z.B. get_avg_infra_capacity)


# --- Testklasse ---

class TestCrisisManagerIntegration:

    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Stellt sicher, dass Logging für Tests sichtbar ist."""
        logging.basicConfig(level=logging.DEBUG) # Setze auf DEBUG für detaillierte Logs

    @pytest.fixture
    def crisis_model(self, custom_model_factory) -> EconomicModel:
        """Erstellt ein Standardmodell mit aktiviertem VSM für Krisentests."""
        # Konfiguration mit einigen Agenten und Ressourcen
        cfg = SimulationConfig(
            simulation_steps=20,
            agent_populations={
                "producers_r0": {"agent_class": "ProducerAgent", "count": 3, "id_prefix": "p_r0_", "params": {"region_id": 0, "initial_capacity": 100, "can_produce": ["A", "B"]}},
                "producers_r1": {"agent_class": "ProducerAgent", "count": 2, "id_prefix": "p_r1_", "params": {"region_id": 1, "initial_capacity": 120, "can_produce": ["C"]}},
                "consumers": {"agent_class": "ConsumerAgent", "count": 10, "id_prefix": "c_", "params": {}}
            },
            specific_agents=[
                # Resource Agent explizit definieren (optional, wird sonst standardmäßig erstellt)
                # {"agent_class": "ResourceAgent", "unique_id": "global_resources", "params": {}}
            ],
            regional_config={"regions": ["R0", "R1"]}, # Verwende Namen statt Indizes
            environment_config={
                "resource_capacities": {"iron": 500, "wood": 800, "energy": 1000},
                "resource_regeneration_rates": {"wood": 0.05, "energy": 0.1}
                },
            vsm_on=True,
            # Crisis Config (Beispiel)
            crisis_config={
                "crisis_probability": 0.0, # Deaktiviere zufällige Krisen für deterministische Tests
                "adaptation_mitigation_factor": 0.7,
                 # Füge hier ggf. spezifische Dauern/Schweregrade hinzu, falls nötig
            }
        )
        model = custom_model_factory(config_dict=cfg.dict()) # Annahme: factory akzeptiert dict
        return model

    def test_crisis_manager_initialization(self, crisis_model: EconomicModel):
        """Prüft, ob der CrisisManager korrekt initialisiert wird."""
        assert hasattr(crisis_model, 'crisis_manager')
        cm = crisis_model.crisis_manager
        assert isinstance(cm, CrisisManager)
        assert cm.crisis_active is False
        assert cm.current_crisis is None
        assert cm.adaptation_level == 0.0
        assert cm.crisis_probability == 0.0 # Aus unserer Test-Config

    def test_trigger_resource_shortage(self, crisis_model: EconomicModel, mocker):
        """Testet das manuelle Auslösen und die direkten Effekte einer Ressourcenkrise."""
        cm = crisis_model.crisis_manager
        resource_agent = crisis_model.resource_agent
        assert resource_agent is not None

        initial_capacity = resource_agent.resource_capacity.get("iron", 500.0) # Hol den Wert aus Agent/Config

        # Mocke VSM-Benachrichtigungen
        mock_s4_start = mocker.patch.object(crisis_model.system4planner, 'handle_crisis_start', autospec=True)
        mock_s5_start = mocker.patch.object(crisis_model.system5policy, 'handle_crisis_start', autospec=True)

        # Krise auslösen
        crisis_params = {"resource": "iron", "severity": 0.5, "duration": 5}
        cm.trigger_crisis("resource_shortage", manual_params=crisis_params)

        # Prüfe Zustand des Managers
        assert cm.crisis_active is True
        assert isinstance(cm.current_crisis, ResourceShortageCrisis)
        assert cm.current_crisis.duration == 5
        assert cm.current_crisis.initial_severity == 0.5
        assert cm.adaptation_level == 0.0

        # Prüfe direkten Effekt auf ResourceAgent
        expected_capacity = initial_capacity * (1.0 - 0.5)
        assert resource_agent.resource_capacity["iron"] == pytest.approx(expected_capacity)

        # Prüfe VSM-Benachrichtigung
        mock_s4_start.assert_called_once()
        mock_s5_start.assert_called_once()
        # Prüfe übergebene Effekte (optional, detaillierter Test)
        call_args_s4 = mock_s4_start.call_args[0]
        assert call_args_s4[0] == "resource_shortage" # crisis_type
        assert call_args_s4[1]['resource'] == "iron"   # effects dict

    def test_crisis_update_and_adaptation(self, crisis_model: EconomicModel):
        """Testet das Update der Krise über Zeit und den Anstieg der Adaption."""
        cm = crisis_model.crisis_manager
        crisis_params = {"resource": "wood", "severity": 0.6, "duration": 10}
        cm.trigger_crisis("resource_shortage", manual_params=crisis_params)

        initial_severity = cm.current_crisis.initial_severity
        assert cm.adaptation_level == 0.0

        # Simuliere einige Schritte
        run_model_steps(crisis_model, 3) # Läuft Schritte 1, 2, 3

        # Prüfe Zustand nach 3 Schritten
        assert cm.crisis_active is True
        assert cm.current_crisis.steps_active == 3
        # Adaption sollte gestiegen sein
        assert cm.adaptation_level > 0.0
        # Aktueller Schweregrad sollte der Logistikkurve folgen
        dt = 3
        logistic = 1.0 / (1 + math.exp(-0.35 * (dt - 6)))
        expected = logistic * (1.0 - cm.adaptation_level * 0.7)
        assert cm.current_crisis.get_severity() == pytest.approx(expected, rel=0.3)

    def test_crisis_end_and_recovery(self, crisis_model: EconomicModel, mocker):
        """Testet das korrekte Beenden einer Krise und die teilweise Wiederherstellung."""
        cm = crisis_model.crisis_manager
        resource_agent = crisis_model.resource_agent
        resource = "energy"
        initial_capacity = resource_agent.resource_capacity.get(resource, 1000.0)
        crisis_params = {"resource": resource, "severity": 0.4, "duration": 3}

        # Mocke End-Benachrichtigungen
        mock_s4_end = mocker.patch.object(crisis_model.system4planner, 'handle_crisis_end', autospec=True)
        mock_s5_end = mocker.patch.object(crisis_model.system5policy, 'handle_crisis_end', autospec=True)

        # Krise auslösen
        cm.trigger_crisis("resource_shortage", manual_params=crisis_params)
        capacity_during_crisis = resource_agent.resource_capacity[resource]
        assert capacity_during_crisis == pytest.approx(initial_capacity * (1.0 - 0.4))

        # Simuliere bis zum Ende der Krise (3 Schritte laufen lassen)
        run_model_steps(crisis_model, 3) # Läuft Step 1, 2, 3

        # Im nächsten Step (Step 4) sollte die Krise enden
        assert cm.crisis_active is True # Noch aktiv am Ende von Step 3
        run_model_steps(crisis_model, 1) # Führe Step 4 aus
        assert cm.crisis_active is False # Sollte jetzt beendet sein
        assert cm.current_crisis is None

        # Prüfe Wiederherstellung (partiell)
        # Annahme: Standard Recovery Factor = 0.6 in ResourceShortageCrisis.end_crisis
        expected_recovery = (initial_capacity - capacity_during_crisis) * 0.6
        expected_capacity_after_end = capacity_during_crisis + expected_recovery
        assert resource_agent.resource_capacity[resource] == pytest.approx(expected_capacity_after_end)

        # Prüfe History
        assert len(cm.crisis_history) == 1
        history_entry = cm.crisis_history[0]
        assert history_entry["type"] == "resource_shortage"
        assert history_entry["start_step"] == 0 # Krise startete in Step 0 (vor Step 1)
        assert history_entry["end_step"] == 3 # Krise endete nach Step 3
        assert history_entry["duration"] == 3
        assert history_entry["initial_severity"] == 0.4

        # Prüfe VSM End-Benachrichtigungen
        mock_s4_end.assert_called_once_with("resource_shortage")
        mock_s5_end.assert_called_once_with("resource_shortage")

    def test_natural_disaster_regional_impact(self, crisis_model: EconomicModel):
        """Testet, ob eine Naturkatastrophe nur die betroffenen Regionen schädigt."""
        cm = crisis_model.crisis_manager
        # Finde Producer in verschiedenen Regionen
        producers_r0 = [p for p in crisis_model.producers if p.region_id == 0]
        producers_r1 = [p for p in crisis_model.producers if p.region_id == 1]
        assert len(producers_r0) > 0
        assert len(producers_r1) > 0

        initial_cap_r0 = get_avg_producer_capacity(crisis_model, region_id=0)
        initial_cap_r1 = get_avg_producer_capacity(crisis_model, region_id=1)

        # Krise nur für Region 0 auslösen
        crisis_params = {"affected_regions": [0], "severity": 0.5, "production_damage": 0.6} # 60% Produktionsschaden
        cm.trigger_crisis("natural_disaster", manual_params=crisis_params)

        # Prüfe Kapazität direkt nach Auslösung
        current_cap_r0 = get_avg_producer_capacity(crisis_model, region_id=0)
        current_cap_r1 = get_avg_producer_capacity(crisis_model, region_id=1)

        # Erwartung: Kapazität in R0 reduziert, in R1 unverändert
        expected_cap_r0 = initial_cap_r0 * (1.0 - 0.6 * 0.5) # production_damage * severity
        assert current_cap_r0 == pytest.approx(expected_cap_r0, rel=0.5)
        assert current_cap_r1 == pytest.approx(initial_cap_r1, rel=0.01) # Unverändert

    # TODO: Füge Tests für die anderen Krisentypen hinzu:
    # - test_trigger_infrastructure_failure (prüft Infra Agents)
    # - test_trigger_technological_disruption (prüft Producer Effizienz für bestimmtes Gut)
    # - test_trigger_environmental_catastrophe (prüft globale Effekte auf Producer/Ressourcen)

    def test_crisis_prevents_new_crisis(self, crisis_model: EconomicModel):
        """Testet, dass keine neue Krise startet, während eine aktiv ist."""
        cm = crisis_model.crisis_manager
        cm.trigger_crisis("resource_shortage", {"duration": 5})
        assert cm.crisis_active is True

        # Setze Wahrscheinlichkeit für neue Krise auf 100%
        cm.crisis_probability = 1.0
        # Versuche, neue Krise auszulösen (passiert in update_crisis_states)
        cm.update_crisis_states() # Sollte nichts tun, da Krise aktiv

        # Die ursprüngliche Krise sollte immer noch aktiv sein
        assert cm.crisis_active is True
        assert isinstance(cm.current_crisis, ResourceShortageCrisis)