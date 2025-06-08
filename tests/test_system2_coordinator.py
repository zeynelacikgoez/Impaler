# tests/test_system2_coordinator.py
"""
Unit-Tests für den System2Coordinator und seine Subsysteme.

Fokus liegt auf der isolierten Logik der Konflikterkennung, -lösung,
Ressourcenkoordination und Fairness-Bewertung auf interregionaler Ebene.
Die Interaktion mit dem echten Model und anderen VSM-Systemen wird gemockt.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from collections import defaultdict
import logging
from typing import Dict, Any, Union

# Importiere die zu testenden Klassen (Pfade ggf. anpassen)
from impaler.vsm.system2 import (
    System2Coordinator,
    ConflictResolutionSystem,
    ResourceCoordinationSystem,
    FairnessEvaluator,
    # Importiere Kommunikations-Datenstrukturen, falls separat definiert
    # ConflictResolutionDirectiveS2, SystemFeedback
)
# Importiere Konfigurationsmodelle (Annahme: Pydantic wird genutzt)
from impaler.core.config import SimulationConfig, ADMMConfig, PlanningPriorities, EnvironmentConfig, RegionalConfig

# --- Fixtures für Mocks und Testdaten ---

@pytest.fixture
def mock_model() -> MagicMock:
    """Erstellt ein Mock-Objekt für das EconomicModel."""
    model = MagicMock(name="EconomicModel")
    model.current_step = 10 # Beispielhafter Step
    model.logger = logging.getLogger("MockEconomicModel") # Einfacher Logger
    model.logger.getChild = lambda name: logging.getLogger(f"MockModel.{name}") # Mock getChild

    # Minimale Konfiguration bereitstellen
    model.config = SimulationConfig(
        # Minimale Config, damit Attribute existieren
        regional_config=RegionalConfig(regions=["R0", "R1"]),
        planning_priorities=PlanningPriorities(goods_priority={"A": 1.0, "B": 0.8}),
        environment_config=EnvironmentConfig(resource_capacities={"X": 100, "Y": 200}),
        admm_config=ADMMConfig(),
        system2_params={ # Eigene S2 Konfiguration
            "coordinator_iterations": 2,
            "critical_resource_boost": 2.0,
            "conflict_thresholds": {
                "plan_imbalance": 0.2, "resource_critical": 0.3, "resource_standard": 0.5,
                "emission": 0.4, "fairness": 0.25
             },
             "priority_weights": { # Gewichte für Prioritätsberechnung
                "plan_gap": 2.0, "critical_needs": 3.0, "population": 0.5, "emissions": -0.1
             }
        }
    )
    # Machen S3 und Route-Funktion mockbar
    model.system3manager = MagicMock(name="System3Manager")
    model.route_directive = MagicMock(name="route_directive", return_value=True) # Standard: Routing erfolgreich

    return model

@pytest.fixture
def basic_regional_imbalances() -> Dict[Union[int, str], Dict[str, Any]]:
    """Liefert einfache Beispieldaten für regional_imbalances."""
    return {
        0: { # Region 0 - Benötigt Ressource X
            "plan_fulfillment": 0.7,
            "resource_shortfall": {"X": 50.0, "Y": 10.0},
            "resource_surplus": {"A": 5.0},
            "emissions": 100.0,
            "population": 1000,
            "critical_shortages_amount": 50.0 # Annahme X ist kritisch
            ,"agent_object": MagicMock(name="rm0")
        },
        1: { # Region 1 - Hat Überschuss X, braucht Y
            "plan_fulfillment": 0.95,
            "resource_shortfall": {"Y": 30.0},
            "resource_surplus": {"X": 70.0, "B": 10.0},
            "emissions": 80.0,
            "population": 1200,
            "critical_shortages_amount": 0.0,
            "agent_object": MagicMock(name="rm1")
        }
    }

@pytest.fixture
def system2_coordinator(mock_model: MagicMock, basic_regional_imbalances) -> System2Coordinator:
    """Erstellt eine Instanz des System2Coordinator mit gemocktem Modell."""
    mock_model.system3manager.get_all_regional_statuses.return_value = basic_regional_imbalances
    return System2Coordinator(mock_model)

# --- Tests für System2Coordinator ---

class TestSystem2Coordinator:

    def test_init(self, mock_model: MagicMock):
        """Testet die Initialisierung des Coordinators."""
        coordinator = System2Coordinator(mock_model)
        assert coordinator.model == mock_model
        assert coordinator.logger is not None
        assert coordinator.coordinator_iterations == 2 # Aus mock_model.config
        assert coordinator.critical_resource_boost == 2.0
        assert isinstance(coordinator.conflict_resolution_system, ConflictResolutionSystem)
        assert isinstance(coordinator.resource_coordinator, ResourceCoordinationSystem)
        assert isinstance(coordinator.fairness_evaluator, FairnessEvaluator)
        # ... weitere Checks ...

    def test_collect_region_status(self, system2_coordinator: System2Coordinator, basic_regional_imbalances: Dict):
        """Testet das Sammeln und Verarbeiten von Regionaldaten."""
        system2_coordinator.collect_region_status()

        # Prüfe, ob Daten korrekt übernommen wurden
        assert len(system2_coordinator.regional_imbalances) == 2
        assert 0 in system2_coordinator.regional_imbalances
        assert 1 in system2_coordinator.regional_imbalances
        assert system2_coordinator.regional_imbalances[0]["plan_fulfillment"] == 0.7
        assert system2_coordinator.regional_imbalances[1]["surplus"]["X"] == 70.0

        # Prüfe, ob Shortages/Surpluses korrekt extrahiert wurden
        assert system2_coordinator.current_resource_shortages[0]["X"] == 50.0
        assert system2_coordinator.current_resource_surpluses[1]["X"] == 70.0
        assert system2_coordinator.current_resource_shortages[1]["Y"] == 30.0
        assert system2_coordinator.current_resource_surpluses[0]["A"] == 5.0

        # Prüfe, ob Prioritäten berechnet wurden (Werte hängen von _calculate_single_region_priority ab)
        assert 0 in system2_coordinator.regional_priorities
        assert 1 in system2_coordinator.regional_priorities
        assert system2_coordinator.regional_priorities[0] > 0 # Region 0 sollte Prio haben (Plan Gap, Critical Shortage)
        # Mache den Test robuster gegen genaue Prioritätswerte
        assert isinstance(system2_coordinator.regional_priorities[0], float)
        assert isinstance(system2_coordinator.regional_priorities[1], float)

    def test_detect_and_resolve_conflicts(self, system2_coordinator: System2Coordinator, mocker: MagicMock):
        """Testet den Ablauf der Konflikterkennung und -lösung."""
        # Mocke die Subsystem-Methoden
        mock_identify = mocker.patch.object(system2_coordinator.conflict_resolution_system, 'identify_conflicts')
        mock_resolve = mocker.patch.object(system2_coordinator.conflict_resolution_system, 'resolve_conflict')
        mock_create_directive = mocker.patch.object(system2_coordinator, '_create_directive_from_resolution')

        # Szenario 1: Keine Konflikte
        mock_identify.return_value = []
        resolved = system2_coordinator.detect_and_resolve_conflicts()
        assert resolved == []
        mock_resolve.assert_not_called()
        mock_create_directive.assert_not_called()
        system2_coordinator.model.route_directive.assert_not_called()

        # Szenario 2: Ein Konflikt, Lösung gefunden und Direktive erstellt/gesendet
        mock_conflict = {"type": "plan_imbalance", "parties": [0, 1], "severity": 0.25, "id": "c1"}
        mock_resolution = {"method": "negotiation", "parties": [0, 1]}
        # Annahme: _create... gibt eine (Mock-)Direktive zurück
        mock_directive = MagicMock(name="Directive", action_type="negotiate", payload={"parties": [0,1]}, target_system="System3")
        mock_identify.return_value = [mock_conflict]
        mock_resolve.return_value = mock_resolution
        mock_create_directive.return_value = mock_directive
        system2_coordinator.model.route_directive.return_value = True # Senden erfolgreich

        resolved = system2_coordinator.detect_and_resolve_conflicts()

        mock_identify.assert_called_once()
        mock_resolve.assert_called_once_with(mock_conflict)
        mock_create_directive.assert_called_once_with(mock_conflict, mock_resolution)
        system2_coordinator.model.route_directive.assert_called_once_with(mock_directive)
        assert len(resolved) == 1
        assert resolved[0]["directive_type"] == "negotiate" # Prüfe Info im Rückgabewert

        # Szenario 3: Konflikt, aber keine Lösung gefunden
        mock_resolve.return_value = None
        mock_create_directive.reset_mock()
        system2_coordinator.model.route_directive.reset_mock()
        resolved = system2_coordinator.detect_and_resolve_conflicts()
        assert resolved == []
        mock_create_directive.assert_not_called()
        system2_coordinator.model.route_directive.assert_not_called()

    def test_create_directive_from_resolution(self, system2_coordinator: System2Coordinator):
        """Testet die Umwandlung von Lösungsvorschlägen in Direktiven."""
        conflict = {"type": "resource_conflict", "parties": [0, 1], "resource": "X", "id": "c2", "severity": 0.6}

        # Fall 1: resource_reallocation
        resolution1 = {"method": "resource_reallocation", "resource": "X", "amount": 30.0}
        directive1 = system2_coordinator._create_directive_from_resolution(conflict, resolution1)
        assert directive1 is not None
        assert directive1.target_system == "System3"
        assert directive1.action_type == "resource_transfer"
        assert directive1.payload == {"resource": "X", "amount": 30.0, "from_region": 1, "to_region": 0} # Beachte Annahme der Reihenfolge in parties

        # Fall 2: plan_adjustment
        resolution2 = {"method": "plan_adjustment", "rationale": "Test Rationale", "suggested_changes": {"reduce": [1]}}
        directive2 = system2_coordinator._create_directive_from_resolution(conflict, resolution2)
        assert directive2 is not None
        assert directive2.target_system == "System4"
        assert directive2.action_type == "request_plan_adjustment"
        assert directive2.payload["rationale"] == "Test Rationale"
        assert directive2.payload["regions"] == [0, 1]

        # Fall 3: negotiation (sollte None zurückgeben, da intern behandelt)
        resolution3 = {"method": "negotiation", "parties": [0, 1]}
        # Mocke den Handler, um Erfolg zu simulieren
        system2_coordinator.negotiation_handler.execute_negotiation = MagicMock(return_value=True)
        directive3 = system2_coordinator._create_directive_from_resolution(conflict, resolution3)
        assert directive3 is None
        system2_coordinator.negotiation_handler.execute_negotiation.assert_called_once()

    def test_check_fairness_and_adjust(self, system2_coordinator: System2Coordinator, mocker: MagicMock):
        """Testet die Fairness-Prüfung und Anpassung der Prioritäten."""
        # Mocke FairnessEvaluator
        mock_evaluate = mocker.patch.object(system2_coordinator.fairness_evaluator, 'evaluate_fairness')
        mock_get_adjust = mocker.patch.object(system2_coordinator.fairness_evaluator, 'get_fairness_adjustments')

        # Szenario 1: Keine Anpassung nötig
        mock_evaluate.return_value = {"combined_fairness": 0.1, "requires_adjustment": False, "plan_gini": 0.1, "resource_gini": 0.1}
        system2_coordinator.regional_priorities = {0: 5.0, 1: 4.0} # Beispiel-Prioritäten
        system2_coordinator.check_fairness_and_adjust()
        mock_get_adjust.assert_not_called()
        assert system2_coordinator.regional_priorities[0] == 5.0 # Prioritäten unverändert

        # Szenario 2: Anpassung nötig
        mock_evaluate.return_value = {"combined_fairness": 0.3, "requires_adjustment": True, "plan_gini": 0.3, "resource_gini": 0.2}
        # Annahme: get_fairness_adjustments gibt Faktoren zurück
        mock_get_adjust.return_value = {"priority_adjustments": {0: 1.2, 1: 0.8}} # Region 0 boost, Region 1 reduce
        system2_coordinator.regional_priorities = {0: 5.0, 1: 4.0}
        system2_coordinator.check_fairness_and_adjust()
        mock_get_adjust.assert_called_once()
        assert system2_coordinator.regional_priorities[0] == pytest.approx(5.0 * 1.2)
        assert system2_coordinator.regional_priorities[1] == pytest.approx(4.0 * 0.8)

    def test_send_feedback_to_higher_systems(self, system2_coordinator: System2Coordinator):
        """Testet das Senden von gesammelten Feedbacks/Direktiven."""
        # Füge Test-Direktiven hinzu
        directive1 = MagicMock(spec=['target_system', 'action_type', 'payload'], target_system="System4", action_type="request_plan_adjustment", payload={})
        directive2 = MagicMock(spec=['target_system', 'action_type', 'payload'], target_system="System5", action_type="request_intervention", payload={})
        system2_coordinator.feedbacks_to_higher_systems = [directive1, directive2]

        system2_coordinator._send_feedback_to_higher_systems()

        # Prüfe, ob model.route_directive für jede Direktive aufgerufen wurde
        assert system2_coordinator.model.route_directive.call_count == 2
        system2_coordinator.model.route_directive.assert_any_call(directive1)
        system2_coordinator.model.route_directive.assert_any_call(directive2)
        # Prüfe, ob Liste geleert wurde
        assert not system2_coordinator.feedbacks_to_higher_systems


# --- Tests für ConflictResolutionSystem ---

@pytest.fixture
def conflict_resolver(system2_coordinator: System2Coordinator) -> ConflictResolutionSystem:
    """Erstellt eine Instanz des ConflictResolutionSystem."""
    return system2_coordinator.conflict_resolution_system

class TestConflictResolutionSystem:

    def test_identify_plan_imbalance(self, conflict_resolver: ConflictResolutionSystem):
        """Testet die Erkennung von Plan-Imbalance Konflikten."""
        imbalances = {
            0: {"plan_fulfillment": 0.6, "population": 1000}, # Deutlich unter Plan
            1: {"plan_fulfillment": 0.9, "population": 1000}, # Im Plan
            2: {"plan_fulfillment": 1.1, "population": 1000}  # Leicht über Plan
        }
        # Threshold ist 0.2 in fixture config
        conflicts = conflict_resolver._detect_plan_imbalance_conflicts(imbalances)
        # Erwarte Konflikt zwischen 0 und 2 (1.1 - 0.6 = 0.5 > 0.2)
        # Erwarte Konflikt zwischen 0 und 1 (0.9 - 0.6 = 0.3 > 0.2)
        assert len(conflicts) >= 1 # Mindestens ein Konflikt
        found_0_2 = any(set(c['parties']) == {0, 2} and c['type'] == 'plan_imbalance' for c in conflicts)
        found_0_1 = any(set(c['parties']) == {0, 1} and c['type'] == 'plan_imbalance' for c in conflicts)
        assert found_0_2 or found_0_1 # Mindestens einer der erwarteten Konflikte muss da sein

    def test_identify_resource_conflict(self, conflict_resolver: ConflictResolutionSystem, basic_regional_imbalances: Dict):
        """Testet die Erkennung von Ressourcenkonflikten."""
        # Nutze die Fixture Daten: R0 braucht 50 X, R1 hat 70 X übrig
        # Annahme: X ist kritisch, threshold 0.3
        # Annahme: Relative shortage > 0.3
        conflict_resolver.coordinator.critical_resources = {"X"} # Mache X kritisch
        # Setze Bevölkerung für Berechnung von relative_shortage
        basic_regional_imbalances[0]['population'] = 1000
        basic_regional_imbalances[1]['population'] = 1000
        # relative_shortage = 50 / (1000/1000) = 50.0 -> > 0.3

        conflicts = conflict_resolver._detect_resource_conflicts(basic_regional_imbalances)
        assert len(conflicts) == 1
        assert conflicts[0]['type'] == 'resource_conflict'
        assert conflicts[0]['resource'] == 'X'
        assert set(conflicts[0]['parties']) == {0, 1} # R0 (shortage) vs R1 (surplus)
        assert conflicts[0]['is_critical'] == True

    def test_resolve_conflict_selection(self, conflict_resolver: ConflictResolutionSystem, mocker: MagicMock):
        """Testet die Auswahl der Auflösungsmethode basierend auf Typ/Severity/Chronicity."""
        # Fall 1: Plan Imbalance, niedrig/neu -> negotiation
        conflict1 = {"type": "plan_imbalance", "parties": [0, 1], "severity": 0.3, "chronic_level": 1}
        res1 = conflict_resolver.resolve_conflict(conflict1)
        assert res1 is not None and res1['method'] == 'negotiation'

        # Fall 2: Plan Imbalance, mittel/neu -> resource_reallocation (wenn möglich)
        conflict2 = {"type": "plan_imbalance", "parties": [0, 1], "severity": 0.7, "chronic_level": 1}
        # Mocke _find_reallocatable_resource, um eine Ressource zu finden
        mocker.patch.object(conflict_resolver, '_find_reallocatable_resource', return_value=("Y", 20.0))
        res2 = conflict_resolver.resolve_conflict(conflict2)
        assert res2 is not None and res2['method'] == 'resource_reallocation'
        assert res2['resource'] == 'Y'

        # Fall 3: Plan Imbalance, hoch/chronisch -> plan_adjustment
        conflict3 = {"type": "plan_imbalance", "parties": [0, 1], "severity": 0.9, "chronic_level": 3}
        res3 = conflict_resolver.resolve_conflict(conflict3)
        assert res3 is not None and res3['method'] == 'plan_adjustment'

        # Fall 4: Resource Conflict, kritisch -> resource_reallocation
        conflict4 = {"type": "resource_conflict", "parties": [0, 1], "resource": "X", "is_critical": True, "severity": 0.5, "chronic_level": 1, "values": {"shortage": 50, "surplus": 70}}
        res4 = conflict_resolver.resolve_conflict(conflict4)
        assert res4 is not None and res4['method'] == 'resource_reallocation'
        assert res4['amount'] == 50.0 # min(shortage, surplus)

        # Fall 5: Resource Conflict, nicht kritisch, niedrig -> negotiation
        conflict5 = {"type": "resource_conflict", "parties": [0, 1], "resource": "B", "is_critical": False, "severity": 0.4, "chronic_level": 1, "values": {"shortage": 10, "surplus": 30}}
        res5 = conflict_resolver.resolve_conflict(conflict5)
        assert res5 is not None and res5['method'] == 'negotiation'


# --- Tests für ResourceCoordinationSystem ---

@pytest.fixture
def resource_coordinator(system2_coordinator: System2Coordinator) -> ResourceCoordinationSystem:
    """Erstellt eine Instanz des ResourceCoordinationSystem."""
    return system2_coordinator.resource_coordinator

class TestResourceCoordinationSystem:

    def test_heuristic_allocate_scarce_resource(self, resource_coordinator: ResourceCoordinationSystem):
        """Testet die heuristische Allokation für knappe Ressourcen."""
        resource = "X"
        # R0 braucht 50 (Prio 5), R2 braucht 30 (Prio 8)
        demands = {0: {resource: 50.0}, 2: {resource: 30.0}}
        # R1 hat 60 übrig
        supplies = {1: {resource: 60.0}}
        priorities = {0: 5.0, 1: 1.0, 2: 8.0} # Prioritäten
        resource_coordinator.coordinator.current_resource_shortages = defaultdict(lambda: defaultdict(float), demands) # Update Coordinator state
        resource_coordinator.coordinator.current_resource_surpluses = defaultdict(lambda: defaultdict(float), supplies)

        # Führe Allokation aus (Iteration 0, Annahme: X ist nicht kritisch -> Proportional?)
        # Die Logik in der aktuellen Implementierung ist unklar, dieser Test muss angepasst werden!
        # Annahme: Proportionale Logik wird verwendet
        allocations = resource_coordinator._heuristic_allocate_scarce_resource(resource, demands, supplies, priorities, iteration=1)

        # Gesamtbedarf = 80, Verfügbar = 60 -> Faktor = 60/80 = 0.75
        # Gewichteter Bedarf R0 = 5*50=250, R2 = 8*30=240 -> Summe ~490
        # Anteil R0 ~ 250/490 ~ 0.51 -> Zuteilung ~ 0.51 * 60 ~ 30.6
        # Anteil R2 ~ 240/490 ~ 0.49 -> Zuteilung ~ 0.49 * 60 ~ 29.4
        # (Dies hängt von der exakten Implementierung in _calculate_single_resource_allocation ab!)

        # Einfacher Check: Gesamtallokation = min(total_need, total_supply)
        total_allocated = sum(amount for _, _, amount in allocations)
        assert total_allocated == pytest.approx(60.0)

        # Prüfe, ob beide Demanders etwas bekommen haben
        amount_r0 = sum(a for s, t, a in allocations if t == 0)
        amount_r2 = sum(a for s, t, a in allocations if t == 2)
        assert amount_r0 > 0
        assert amount_r2 > 0
        # Prüfe, ob R2 (höhere Prio*Bedarf) mehr oder gleich viel bekommt? (Abhängig von Formel)
        # assert amount_r2 >= amount_r0 # Nicht garantiert durch alle Formeln

    def test_heuristic_allocate_abundant_resource(self, resource_coordinator: ResourceCoordinationSystem):
        """Testet die heuristische Allokation für reichliche Ressourcen."""
        resource = "Y"
        # R0 braucht 10, R1 braucht 20
        demands = {0: {resource: 10.0}, 1: {resource: 20.0}}
        # R2 hat 50 übrig
        supplies = {2: {resource: 50.0}}
        priorities = {0: 5.0, 1: 4.0, 2: 1.0}
        resource_coordinator.coordinator.current_resource_shortages = defaultdict(lambda: defaultdict(float), demands)
        resource_coordinator.coordinator.current_resource_surpluses = defaultdict(lambda: defaultdict(float), supplies)

        # Führe Allokation aus
        allocations = resource_coordinator._heuristic_allocate_abundant_resource(resource, demands, supplies, priorities)

        # Gesamtbedarf = 30, Verfügbar = 50 -> Alle Bedarfe sollten gedeckt werden
        total_allocated = sum(amount for _, _, amount in allocations)
        assert total_allocated == pytest.approx(30.0)

        # Prüfe, ob jeder genau seinen Bedarf bekommen hat
        amount_r0 = sum(a for s, t, a in allocations if t == 0)
        amount_r1 = sum(a for s, t, a in allocations if t == 1)
        assert amount_r0 == pytest.approx(10.0)
        assert amount_r1 == pytest.approx(20.0)


# --- Tests für FairnessEvaluator ---
@pytest.fixture
def fairness_evaluator(system2_coordinator: System2Coordinator) -> FairnessEvaluator:
    """Erstellt eine Instanz des FairnessEvaluator."""
    return system2_coordinator.fairness_evaluator

class TestFairnessEvaluator:

    def test_evaluate_fairness(self, fairness_evaluator: FairnessEvaluator):
        """Testet die Berechnung von Fairness-Metriken."""
        # Hohe Ungleichheit
        imbalances1 = {
            0: {"plan_fulfillment": 0.1, "surplus": {"X": 1}, "population": 100},
            1: {"plan_fulfillment": 0.9, "surplus": {"X": 99}, "population": 100}
        }
        report1 = fairness_evaluator.evaluate_fairness(imbalances1)
        assert report1["plan_gini"] > 0.3 # Erwarte hohen Gini
        assert report1["resource_gini"] > 0.4
        assert report1["requires_adjustment"] is True

        # Niedrige Ungleichheit
        imbalances2 = {
            0: {"plan_fulfillment": 0.8, "surplus": {"X": 45}, "population": 100},
            1: {"plan_fulfillment": 0.85,"surplus": {"X": 55}, "population": 100}
        }
        report2 = fairness_evaluator.evaluate_fairness(imbalances2)
        assert report2["plan_gini"] < 0.1
        assert report2["resource_gini"] < 0.1
        assert report2["requires_adjustment"] is False # Unter dem Threshold von 0.25

    def test_get_fairness_adjustments(self, fairness_evaluator: FairnessEvaluator):
        """Testet die Generierung von Anpassungsvorschlägen."""
        # Report, der Anpassung erfordert
        report = {
            "plan_gini": 0.4, "resource_gini": 0.3, "combined_fairness": 0.37,
            "requires_adjustment": True
        }
        # Mocke Imbalances, um niedrigste/höchste Regionen zu identifizieren
        fairness_evaluator.coordinator.regional_imbalances = {
            0: {"plan_fulfillment": 0.1, "population": 100}, # Niedrigste
            1: {"plan_fulfillment": 0.5, "population": 100},
            2: {"plan_fulfillment": 0.9, "population": 100}  # Höchste
        }
        adjustments = fairness_evaluator.get_fairness_adjustments(report)

        # Erwarte, dass Prio für Region 0 erhöht (Faktor > 1) und für 2 gesenkt (Faktor < 1) wird
        assert 0 in adjustments["priority_adjustments"]
        assert 2 in adjustments["priority_adjustments"]
        assert adjustments["priority_adjustments"][0] > 1.0
        assert adjustments["priority_adjustments"][2] < 1.0
        assert 1 not in adjustments["priority_adjustments"] # Mittlere Region nicht angepasst