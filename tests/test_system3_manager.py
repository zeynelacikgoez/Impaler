# tests/test_system3_manager.py
"""
Unit-Tests für System3Manager und RegionalSystem3Manager aus vsm/system3.py.

Fokus liegt auf der isolierten Logik der operativen Koordination,
Ressourcenallokation (innerhalb und zwischen Regionen), Feedback-Verarbeitung,
Auditierung und Interaktion mit anderen VSM-Systemen (gemockt).
"""

import pytest
import numpy as np
import logging
from unittest.mock import MagicMock, patch, call, ANY
from collections import defaultdict, deque
from typing import Union, Dict, Any

# Importiere die zu testenden Klassen (Pfade ggf. anpassen)
from impaler.vsm.system3 import System3Manager, RegionalSystem3Manager

# Importiere abhängige Typen (oder Mocks davon)
# Annahme: Diese existieren und sind importierbar
from impaler.core.model import EconomicModel
from impaler.core.config import SimulationConfig, PlanningPriorities, ADMMConfig # Beispiel
from impaler.agents.producer import ProducerAgent, ProductionLine, ResourceRequirement # Beispiel
# Importiere Kommunikationsverträge (oder Dummies)
try:
    from impaler.vsm.communication_contracts import OperationalReportS3, StrategicDirectiveS4, ConflictResolutionDirectiveS2
except ImportError:
    # Dummy-Klassen, falls die echten nicht existieren
    OperationalReportS3 = dict
    StrategicDirectiveS4 = dict
    ConflictResolutionDirectiveS2 = dict


# --- Fixtures ---

@pytest.fixture
def mock_model_s3(mocker) -> MagicMock:
    """Erstellt ein Mock EconomicModel speziell für System3 Tests."""
    model = MagicMock(spec=EconomicModel)
    model.current_step = 5 # Beispiel Step
    model.logger = logging.getLogger("MockEconomicModelS3")
    model.logger.getChild = lambda name: logging.getLogger(f"MockModelS3.{name}")

    # Minimale Konfiguration bereitstellen
    model.config = SimulationConfig(
        planning_priorities=PlanningPriorities(goods_priority={"A": 1.2, "B": 0.8}),
        admm_config=ADMMConfig(), # Default ADMM Config
        system3_params={ # Spezifische S3 Parameter
            "fairness_weight": 0.5,
            "stability_factor": 0.7,
            "enable_advanced_optimization": True,
            "enable_adaptive_coordination": True,
            "max_feedback_cycles": 3,
            "audit_active": True,
            "conflict_resolution_strategies": ["proportional", "priority_based", "bargaining"],
            "initial_conflict_strategy": "proportional"
        }
    )
    # Mock S4 Planner
    model.system4planner = MagicMock(name="System4Planner")
    model.system4planner.get_strategic_directives.return_value = StrategicDirectiveS4(step=5, production_targets={"A": 100, "B": 80}) # Beispiel-Direktiven

    # Mock Producer Liste/Dict
    model.producers = [] # Wird in Tests gefüllt
    model.agents_by_id = {}

    # Mock route_directive
    model.route_directive = MagicMock(return_value=True)

    # Mock NetworkX, falls benötigt (Standard: Nicht verfügbar für Fallback-Tests)
    mocker.patch('impaler.vsm.system3.NETWORKX_AVAILABLE', False)

    return model

@pytest.fixture
def mock_regional_manager(mocker) -> MagicMock:
    """Erstellt einen Mock für den RegionalSystem3Manager."""
    rm = MagicMock(spec=RegionalSystem3Manager)
    rm.region_id = mocker.sentinel.region_id # Eindeutige ID
    rm.logger = logging.getLogger(f"MockRM.{rm.region_id}")
    # Mocke Methoden, die von System3Manager aufgerufen werden
    rm.collect_regional_resources.return_value = {}
    rm.calculate_resource_needs.return_value = {}
    rm.calculate_resource_surpluses.return_value = {}
    rm.calculate_regional_priority.return_value = 1.0
    rm.coordinate_regional_producers.return_value = None
    rm.get_performance_metrics.return_value = {'stress_level': 0.2}
    rm.generate_feedback.return_value = {'stress_level': 0.2}
    rm.get_summary_status.return_value = {'stress_level': 0.2}
    rm.transfer_resource.return_value = True # Standard: Transfer erfolgreich
    rm.receive_resource.return_value = None
    rm.audit_producer.return_value = {"resource_waste": 0.1, "capacity_underutilization": 0.1}
    rm.apply_producer_penalties.return_value = None
    rm.receive_production_targets.return_value = None
    rm.receive_priority_updates.return_value = None
    return rm

@pytest.fixture
def mock_producer(mocker, unique_id: str, region_id: Union[int, str]) -> MagicMock:
    """Erstellt einen Mock für einen ProducerAgent."""
    p = MagicMock(spec=ProducerAgent)
    p.unique_id = unique_id
    p.region_id = region_id
    p.resource_stock = defaultdict(float)
    p.production_target = defaultdict(float)
    p.productive_capacity = 100.0
    p.tech_level = 1.0
    p.can_produce = {'A'}
    # Mocke benötigte Methoden
    p.calculate_resource_needs.return_value = {} # Standard: keine Bedürfnisse
    # ... weitere Attribute/Methoden nach Bedarf mocken ...
    return p


# --- Tests für System3Manager ---

class TestSystem3Manager:

    def test_init(self, mock_model_s3: MagicMock):
        """Testet die Initialisierung des System3Manager."""
        s3_manager = System3Manager(mock_model_s3)
        assert s3_manager.model == mock_model_s3
        assert s3_manager.logger is not None
        assert s3_manager.fairness_weight == 0.5 # Aus Config
        assert s3_manager.stability_factor == 0.7
        assert s3_manager.audit_active is True
        assert s3_manager.current_conflict_strategy == "proportional"
        assert s3_manager.dependency_graph is None # Da NetworkX gemockt ist

    def test_register_regional_manager(self, mock_model_s3: MagicMock, mock_regional_manager: MagicMock):
        """Testet das Registrieren eines RegionalManagers."""
        s3_manager = System3Manager(mock_model_s3)
        rm1 = mock_regional_manager
        rm1.region_id = 0
        rm2 = MagicMock(spec=RegionalSystem3Manager) # Zweiter Mock
        rm2.region_id = "RegionA"

        s3_manager.register_regional_manager(0, rm1)
        s3_manager.register_regional_manager("RegionA", rm2)

        assert s3_manager.regions == [0, "RegionA"]
        assert s3_manager.regional_managers[0] == rm1
        assert s3_manager.regional_managers["RegionA"] == rm2

    @patch('impaler.vsm.system3.SCIPY_AVAILABLE', True) # Teste mit SciPy
    @patch('impaler.vsm.system3.minimize') # Mocke die minimize Funktion
    def test_optimize_inter_regional_transfers_scipy(self, mock_minimize: MagicMock, mock_model_s3: MagicMock):
        """Testet den Aufruf der SciPy-Optimierung für interregionale Transfers."""
        s3_manager = System3Manager(mock_model_s3)
        # Deaktiviere Fallback für diesen Test
        s3_manager.enable_advanced_optimization = True

        # Mock-Ergebnis für minimize
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.x = np.array([15.0, 5.0]) # Beispiel-Lösung: Transfer von S0->N0=15, S0->N1=5
        mock_result.fun = -10.0 # Beispiel Objective Value
        mock_minimize.return_value = mock_result

        # Testdaten
        resource = "X"
        needy = {0: 20.0, 1: 10.0} # R0 braucht 20, R1 braucht 10
        suppliers = {2: 30.0}      # R2 hat 30
        priorities = {0: 1.5, 1: 1.0, 2: 0.5}
        total_transfer = 30.0 # min(need=30, supply=30)

        transfers = s3_manager._calculate_inter_regional_transfers(resource, needy, suppliers, priorities, total_transfer)

        # Prüfe, ob minimize aufgerufen wurde
        mock_minimize.assert_called_once()
        # Prüfe, ob das Ergebnis von minimize korrekt verarbeitet wurde
        # Erwartet: [(from_region, to_region, amount)]
        assert len(transfers) == 2
        # Die Reihenfolge der Variablen in x hängt von der Reihenfolge in needy_ids, supplier_ids ab!
        # Annahme: var_indices[(2, 0)] = 0, var_indices[(2, 1)] = 1
        assert (2, 0, pytest.approx(15.0)) in transfers # S2 -> N0
        assert (2, 1, pytest.approx(5.0)) in transfers  # S2 -> N1

    @patch('impaler.vsm.system3.SCIPY_AVAILABLE', False) # Teste ohne SciPy
    def test_proportional_inter_regional_transfers_fallback(self, mock_model_s3: MagicMock):
        """Testet den proportionalen Fallback für interregionale Transfers."""
        s3_manager = System3Manager(mock_model_s3)
        # Stelle sicher, dass Fallback genutzt wird
        s3_manager.enable_advanced_optimization = False

        # Testdaten
        resource = "X"
        needy = {0: 50.0, 1: 30.0} # R0 braucht 50 (Prio 2), R1 braucht 30 (Prio 1)
        suppliers = {2: 40.0}      # R2 hat 40
        priorities = {0: 2.0, 1: 1.0, 2: 0.5}
        total_transfer = 40.0 # min(need=80, supply=40)

        # Mocke die interne proportionale Methode, um ihren Aufruf zu prüfen
        with patch.object(s3_manager, '_proportional_inter_regional_transfers', wraps=s3_manager._proportional_inter_regional_transfers) as mock_proportional:
            transfers = s3_manager._calculate_inter_regional_transfers(resource, needy, suppliers, priorities, total_transfer)
            mock_proportional.assert_called_once() # Prüfen, ob Fallback aufgerufen wurde

        # Prüfe das Ergebnis der proportionalen Logik (hier: einfache Implementierung)
        # Gesamtmenge sollte stimmen
        assert sum(amount for _, _, amount in transfers) == pytest.approx(total_transfer)
        # Mehr sollte an die höhere Prio*Bedarf gehen (R0: 2*50=100 vs R1: 1*30=30)
        amount_r0 = sum(a for _, t, a in transfers if t == 0)
        amount_r1 = sum(a for _, t, a in transfers if t == 1)
        assert amount_r0 > amount_r1 # Erwarte mehr für Region 0
        assert amount_r0 <= 50.0 # Nicht mehr als Bedarf
        assert amount_r1 <= 30.0

    def test_feedback_loop_adjustments(self, s3_manager: System3Manager, mocker: MagicMock):
        """Testet, ob der Feedback-Loop Anpassungen korrekt berechnet und anwendet."""
        # Mocke RM get_performance_metrics
        mock_metrics_r0 = {'resource_utilization': {'X': 0.4}, 'production_fulfillment': {'A': 0.7}, 'stress_level': 0.1}
        mock_metrics_r1 = {'resource_utilization': {'X': 0.95}, 'production_fulfillment': {'A': 1.3}, 'stress_level': 0.6}
        rm0 = MagicMock(spec=RegionalSystem3Manager); rm0.region_id = 0; rm0.get_performance_metrics.return_value = mock_metrics_r0
        rm1 = MagicMock(spec=RegionalSystem3Manager); rm1.region_id = 1; rm1.get_performance_metrics.return_value = mock_metrics_r1
        s3_manager.regional_managers = {0: rm0, 1: rm1}
        s3_manager.regions = [0, 1]
        s3_manager.production_priorities = defaultdict(lambda: 1.0, {"A": 1.0}) # Start-Priorität

        # Mocks für apply_adjustments Helfer
        mock_apply_redir = mocker.patch.object(s3_manager, '_apply_resource_redistribution')
        mock_apply_prio = mocker.patch.object(s3_manager, '_apply_priority_adjustments')
        mock_send_target_feedback = mocker.patch.object(s3_manager, '_send_target_feedback_to_s4')

        # Führe einen Feedback-Zyklus aus
        s3_manager._run_internal_feedback_cycle()

        # Prüfe, ob Anpassungen berechnet wurden (basierend auf den Mock-Metriken)
        # 1. Ressource X: Hohe Imbalance (0.4 vs 0.95) -> Erwarte Redistribution von R0 nach R1
        assert 'X' in s3_manager.last_calculated_adjustments['resource_redistribution']
        redir_x = s3_manager.last_calculated_adjustments['resource_redistribution']['X']
        assert redir_x['from_region'] == 0
        assert redir_x['to_region'] == 1
        assert redir_x['amount'] > 0

        # 2. Gut A: Avg Fulfillment = (0.7+1.3)/2 = 1.0 -> Keine große Prio-Änderung erwartet?
        #    Doch, die _Berechnung_ sollte es vorschlagen: 0.7 -> Prio hoch, 1.3 -> Prio runter
        assert 'A' in s3_manager.last_calculated_adjustments['priority_adjustments']
        new_prio_A = s3_manager.last_calculated_adjustments['priority_adjustments']['A']
        assert new_prio_A != 1.0 # Priorität sollte sich geändert haben

        # 3. Ziel A: Avg Fulfillment = 1.0 -> Kein Target-Faktor erwartet
        assert 'A' not in s3_manager.last_calculated_adjustments['target_adjustments']

        # Prüfe, ob _apply_adjustments die Helfer aufgerufen hat
        mock_apply_redir.assert_called_once()
        mock_apply_prio.assert_called_once()
        mock_send_target_feedback.assert_called_once()


    def test_interface_with_system4(self, s3_manager: System3Manager, mocker: MagicMock):
        """Testet die Interaktion mit System 4."""
        # Mocke Helfermethoden
        mock_report = OperationalReportS3(step=5, coordination_effectiveness=0.8) # Dummy Report
        mocker.patch.object(s3_manager, '_prepare_operational_report', return_value=mock_report)
        mocker.patch.object(s3_manager, '_process_strategic_directives')

        # Führe Interface-Schritt aus
        s3_manager.interface_with_system4()

        # Prüfe Aufrufe an S4 Planner
        s3_manager.model.system4planner.receive_operational_report.assert_called_once_with(mock_report)
        s3_manager.model.system4planner.get_strategic_directives.assert_called_once()
        # Prüfe, ob Direktiven verarbeitet wurden
        s3_manager._process_strategic_directives.assert_called_once()

    def test_apply_resolution_directive(self, s3_manager: System3Manager, mock_regional_manager: MagicMock):
        """Testet die Anwendung einer Direktive von System 2."""
        rm0 = mock_regional_manager; rm0.region_id = 0
        rm1 = MagicMock(spec=RegionalSystem3Manager); rm1.region_id = 1
        rm1.transfer_resource.return_value = True # Mock für rm1
        s3_manager.regional_managers = {0: rm0, 1: rm1}
        s3_manager.regions = [0, 1]

        # Erstelle eine Resource-Transfer-Direktive
        directive = ConflictResolutionDirectiveS2(
            step=5,
            target_system="System3",
            action_type="resource_transfer",
            payload={"resource": "Y", "amount": 25.0, "from_region": 1, "to_region": 0}
        )

        success = s3_manager.apply_resolution_directive(directive)

        assert success is True
        # Prüfe, ob die korrekten RM-Methoden aufgerufen wurden
        rm1.transfer_resource.assert_called_once_with("Y", 25.0, 0)
        rm0.receive_resource.assert_called_once_with("Y", 25.0)


# --- Tests für RegionalSystem3Manager ---

@pytest.fixture
def s3_manager_real(mock_model_s3: MagicMock) -> System3Manager:
     """Erstellt eine echte Instanz des System3Manager für RM Tests."""
     return System3Manager(mock_model_s3)

@pytest.fixture
def regional_manager(s3_manager_real: System3Manager) -> RegionalSystem3Manager:
    """Erstellt eine Instanz des RegionalSystem3Manager."""
    rm = RegionalSystem3Manager(parent_system3=s3_manager_real, region_id=0)
    # Füge Mock-Producer hinzu
    rm.producers = {
        "P0": MagicMock(spec=ProducerAgent, unique_id="P0", region_id=0, can_produce={"A"}, productive_capacity=50, resource_stock=defaultdict(float), production_target=defaultdict(float)),
        "P1": MagicMock(spec=ProducerAgent, unique_id="P1", region_id=0, can_produce={"B"}, productive_capacity=60, resource_stock=defaultdict(float), production_target=defaultdict(float))
    }
    # Mock calculate_resource_needs für Producer
    rm.producers["P0"].calculate_resource_needs.return_value = {"X": 10.0}
    rm.producers["P1"].calculate_resource_needs.return_value = {"X": 15.0, "Y": 5.0}
    return rm


class TestRegionalSystem3Manager:

    def test_init(self, s3_manager_real: System3Manager):
        """Testet die Initialisierung des RegionalManagers."""
        rm = RegionalSystem3Manager(s3_manager_real, region_id=0)
        assert rm.parent == s3_manager_real
        assert rm.region_id == 0
        assert rm.logger is not None
        assert len(rm.producers) == 0
        assert rm.stress_level == 0.0

    def test_calculate_resource_requirements(self, regional_manager: RegionalSystem3Manager):
        """Testet die Berechnung des regionalen Ressourcenbedarfs."""
        # Setze Ziele für Producer (normalerweise durch RM - _assign_producer_targets)
        # Hier direkt setzen für Test von calculate_resource_requirements
        regional_manager.producers["P0"].production_target = defaultdict(float, {"A": 40.0})
        regional_manager.producers["P1"].production_target = defaultdict(float, {"B": 50.0})
        # Annahme: calculate_resource_needs im Producer nutzt production_target
        # Mock ist bereits entsprechend konfiguriert

        regional_manager._calculate_resource_requirements()

        assert regional_manager.resource_needs["X"] == pytest.approx(10.0 + 15.0) # P0 braucht 10, P1 braucht 15
        assert regional_manager.resource_needs["Y"] == pytest.approx(5.0) # Nur P1 braucht Y

    @pytest.mark.parametrize("strategy", ["proportional", "priority_based", "bargaining"])
    def test_allocate_resources_to_producers_scarce(self, regional_manager: RegionalSystem3Manager, strategy: str):
        """Testet die Ressourcenallokation an Producer bei Knappheit mit verschiedenen Strategien."""
        # Setze Knappheit: Verfügbar X=20, Bedarf X=25 (P0=10, P1=15)
        regional_manager.resource_pool = defaultdict(float, {"X": 20.0, "Y": 10.0})
        regional_manager.resource_needs = defaultdict(float, {"X": 25.0, "Y": 5.0}) # Annahme: wurde vorher berechnet
        regional_manager.producers["P0"].calculate_resource_needs.return_value = {"X": 10.0}
        regional_manager.producers["P1"].calculate_resource_needs.return_value = {"X": 15.0, "Y": 5.0}

        # Setze Prioritäten (Beispiel)
        regional_manager.good_priorities = defaultdict(lambda: 1.0, {"A": 1.5, "B": 1.0}) # A ist wichtiger
        # Mocke _calculate_producer_priority (oder teste es separat)
        regional_manager._calculate_producer_priority = lambda p, r: 1.5 if p.unique_id == "P0" else 1.0 # P0 hat höhere Prio

        # Setze Strategie im Parent
        regional_manager.parent.current_conflict_strategy = strategy

        regional_manager._allocate_resources_to_producers()

        # Prüfe Allokation für X (knapp)
        alloc_p0 = regional_manager.producers["P0"].resource_stock["X"]
        alloc_p1 = regional_manager.producers["P1"].resource_stock["X"]
        assert alloc_p0 >= 0
        assert alloc_p1 >= 0
        assert alloc_p0 <= 10.0 # Nicht mehr als Bedarf
        assert alloc_p1 <= 15.0
        assert alloc_p0 + alloc_p1 == pytest.approx(20.0) # Gesamte verfügbare Menge verteilt

        # Prüfe spezifische Strategie-Auswirkungen (grobe Richtung)
        if strategy == "priority_based":
            # P0 hat höhere Prio -> sollte relativ mehr bekommen als bei proportional
            prop_alloc_p0 = (10.0 / 25.0) * 20.0 # = 8.0
            assert alloc_p0 >= prop_alloc_p0 # Mindestens proportional, eher mehr
        elif strategy == "proportional":
             assert alloc_p0 == pytest.approx( (10.0 / 25.0) * 20.0 )
             assert alloc_p1 == pytest.approx( (15.0 / 25.0) * 20.0 )

        # Prüfe Allokation für Y (ausreichend)
        alloc_y_p1 = regional_manager.producers["P1"].resource_stock["Y"]
        assert alloc_y_p1 == pytest.approx(5.0) # Voller Bedarf gedeckt

    def test_assign_producer_targets_limited(self, regional_manager: RegionalSystem3Manager):
        """Testet, ob finale Ziele an Ressourcenlimits angepasst werden."""
         # P0 soll 40 A produzieren, braucht 10 X. P1 soll 50 B produzieren, braucht 15 X.
        regional_manager.production_targets = defaultdict(float, {"A": 40.0, "B": 50.0}) # Regionale Ziele
        regional_manager.producers["P0"].production_target = defaultdict(float, {"A": 40.0}) # Initiale Zuweisung
        regional_manager.producers["P1"].production_target = defaultdict(float, {"B": 50.0})

        # Mocke Ressourcenberechnung im Producer
        regional_manager.producers["P0"].calculate_inputs_for_output = lambda amount: {"X": amount * (10.0/40.0)} # Braucht 0.25 X pro A
        regional_manager.producers["P1"].calculate_inputs_for_output = lambda amount: {"X": amount * (15.0/50.0)} # Braucht 0.3 X pro B

        # Gib P0 nur genug X für 30 A (Bedarf 7.5), aber P1 genug X
        regional_manager.producers["P0"].resource_stock = defaultdict(float, {"X": 7.5})
        regional_manager.producers["P1"].resource_stock = defaultdict(float, {"X": 20.0}) # Genug für 50 B (braucht 15)

        regional_manager._assign_producer_targets()

        # P0's Ziel sollte auf 30 reduziert werden
        assert regional_manager.producers["P0"].production_target["A"] == pytest.approx(30.0)
        # P1's Ziel sollte 50 bleiben
        assert regional_manager.producers["P1"].production_target["B"] == pytest.approx(50.0)

    # TODO: Weitere Tests für RegionalSystem3Manager hinzufügen:
    # - transfer_resource / receive_resource
    # - calculate_regional_priority unter verschiedenen Bedingungen
    # - generate_feedback Struktur und Inhalt
    # - audit_producer Ergebnis
    # - apply_producer_penalties Wirkung