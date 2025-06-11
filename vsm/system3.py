# Impaler/vsm/system3.py
# =====================================================================
# System 3: Operative Koordination, Ressourcen-Ausgleich, VSM-Implementierung
# =====================================================================
# REFACTORED VERSION: Logic delegated to specialized components.

import numpy as np
import logging
import random
import math
from collections import defaultdict, deque
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Set,
    Tuple,
    Union,
    Deque,
    DefaultDict,
    TYPE_CHECKING,
)
from .communication_contracts import GovernanceMode, SystemFeedback  # ADDED FOR RUNTIME

# --- Import Abhängigkeiten ---

# Optional: Netzwerk-Analyse
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Optional SciPy optimisation
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - environment may lack scipy
    SCIPY_AVAILABLE = False
    minimize = None  # type: ignore

# Importiere spezialisierte Komponenten
try:
    from .system3_components import (
        InterRegionalResourceBalancer,
        OperationalFeedbackLoop,
        System3AuditController,
    )
except ImportError as e:
    # Fallback, falls Komponenten nicht gefunden werden (sollte nicht passieren bei korrekter Struktur)
    logging.getLogger(__name__).critical(
        f"Fehler beim Importieren der System3-Komponenten: {e}. System 3 wird nicht korrekt funktionieren."
    )

    # Definiere Dummy-Klassen, um Code lauffähig zu halten (aber funktionslos)
    class DummyComponent:
        def __init__(self, *args, **kwargs):
            pass

        def balance_resources(self, *args, **kwargs):
            return []

        def run_cycles(self, *args, **kwargs):
            return {}

        def run_audit_cycle(self, *args, **kwargs):
            return []

    InterRegionalResourceBalancer = OperationalFeedbackLoop = System3AuditController = (
        DummyComponent
    )


# Typ-Prüfung Imports / Zirkuläre Abhängigkeiten vermeiden
if TYPE_CHECKING:
    from ..core.model import EconomicModel
    from ..agents.producer import ProducerAgent, ResourceRequirement
    from ..agents.consumer import ConsumerAgent

    # Importiere Kommunikations-Datenstrukturen
    from .communication_contracts import (
        OperationalReportS3,
        StrategicDirectiveS4,
        ConflictResolutionDirectiveS2,
        SystemFeedback,
    )


# --- Hilfsfunktionen (falls benötigt) ---
# Beispiel: Sigmoid-Funktion (wird in analyze_feedback verwendet)
def sigmoid(x: float, k: float = 1.0) -> float:
    """Sigmoid-Funktion für weiche Übergänge."""
    if x * -k > 700:
        return 0.0
    return 1.0 / (1.0 + math.exp(-k * x))


# --- Hauptklasse: System3Manager (Refactored) ---


class System3Manager:
    """
    System 3: Operative Koordination in einem planwirtschaftlichen VSM-Modell. (Refactored)

    Verantwortlich für die *Orchestrierung* der operativen Koordination. Delegiert
    spezifische Aufgaben wie interregionalen Ressourcenausgleich, interne
    Feedback-Schleifen und Audits an spezialisierte Komponenten. Verwaltet die
    Schnittstellen zu den RegionalSystem3Managern und zu System 2, 4.

    Behält die Verantwortung für:
    - Verwaltung der RegionalSystem3Manager-Instanzen.
    - Initialisierung und Aufruf der spezialisierten Komponenten.
    - Durchführung der intra-regionalen Koordination (via RMs).
    - Analyse des globalen Feedbacks und Anpassung der *eigenen* Koordinationsparameter.
    - Schnittstellenkommunikation mit S2 und S4.
    - Verwaltung des Abhängigkeitsgraphen (optional).

    Attributes:
        model (EconomicModel): Referenz auf das Hauptmodell.
        logger (logging.Logger): Logger für dieses Modul.
        config (Dict[str, Any]): Spezifische Konfiguration für S3 (und seine Komponenten).
        regions (List[Union[int, str]]): Liste der IDs der verwalteten Regionen.
        regional_managers (Dict[Union[int, str], 'RegionalSystem3Manager']): Manager pro Region.
        system4planner (Optional[Any]): Referenz auf System 4 (für Interface).

        # --- Eigene Koordinationsparameter ---
        fairness_weight (float): Gewichtung Fairness vs. Priorität (angepasst durch analyze_feedback).
        stability_factor (float): Dämpfungsfaktor für adaptive Anpassungen (angepasst durch analyze_feedback).
        current_conflict_strategy (str): Aktuell verwendete Strategie bei Ressourcenkonflikten (angepasst durch analyze_feedback).

        # --- Zustand & Tracking ---
        production_priorities (DefaultDict[str, float]): Globale Prioritäten für Güter.
        critical_goods (Set[str]): Als kritisch markierte Güter.
        resource_flows (DefaultDict[str, Dict[str, float]]): Tracking von Allokationen (wird von Balancer befüllt?).
        global_shortage_history (DefaultDict[str, Deque[float]]): Historie von Knappheiten (für Trendanalyse).
        coordination_effectiveness (float): Gesamtmetrik für die Leistung von System 3.
        strategic_targets (Dict[str, Any]): Aktuelle strategische Vorgaben von System 4.
        autonomy_level (float): Autonomiegrad von System 3 gegenüber System 4 Vorgaben.
        audit_results (List[Dict[str, Any]]): Globale Historie der Audit-Ergebnisse.

        # --- Optionale Komponenten / Features ---
        dependency_graph (Optional[nx.DiGraph]): Produktionsabhängigkeitsgraph.
        bottleneck_identification_active (bool): Ob Engpässe aktiv gesucht werden.

        # --- Delegierte Komponenten ---
        inter_regional_balancer (InterRegionalResourceBalancer): Verwaltet Ressourcenausgleich zwischen Regionen.
        feedback_loop_handler (OperationalFeedbackLoop): Verwaltet interne Feedback-Zyklen.
        audit_controller (System3AuditController): Verwaltet S3* Audits.
    """

    model: "EconomicModel"
    logger: logging.Logger
    config: Dict[str, Any]
    regions: List[Union[int, str]]
    regional_managers: Dict[Union[int, str], "RegionalSystem3Manager"]
    system4planner: Optional[Any]  # Sollte spezifischer Typ sein, z.B. 'System4Planner'

    fairness_weight: float
    stability_factor: float
    current_conflict_strategy: str
    conflict_resolution_strategies: List[str]  # Wird von analyze_feedback genutzt

    production_priorities: DefaultDict[str, float]
    critical_goods: Set[str]
    resource_flows: DefaultDict[str, Dict[str, float]]
    global_shortage_history: DefaultDict[str, Deque[float]]
    coordination_effectiveness: float
    strategic_targets: Dict[str, Any]
    autonomy_level: float
    audit_results: List[Dict[str, Any]]

    dependency_graph: Optional["nx.DiGraph"]  # Keep NetworkX logic here
    bottleneck_identification_active: bool

    # --- Delegierte Komponenten ---
    inter_regional_balancer: "InterRegionalResourceBalancer"
    feedback_loop_handler: "OperationalFeedbackLoop"
    audit_controller: "System3AuditController"
    last_calculated_adjustments: Dict[str, Dict[str, Any]]

    def __init__(self, model: "EconomicModel") -> None:
        """
        Initialisiert den System3Manager und seine Komponenten.

        Args:
            model: Referenz auf das Hauptmodell (EconomicModel).
        """
        self.model = model
        self.logger = model.logger  # Nutzt den zentralen Logger des Modells
        # Lade Konfigurationen aus dem Modell-Config
        self.config = getattr(self.model.config, "system3_params", {})
        planning_prio = getattr(self.model.config, "planning_priorities", None)

        # --- Parameter für S3 selbst (werden z.T. durch analyze_feedback angepasst) ---
        self.fairness_weight = self.config.get("fairness_weight", 0.6)
        self.stability_factor = self.config.get("stability_factor", 0.8)
        self.enable_adaptive_coordination = self.config.get(
            "enable_adaptive_coordination", True
        )
        self.autonomy_level = self.config.get("autonomy_level", 0.7)
        self.conflict_resolution_strategies = self.config.get(
            "conflict_resolution_strategies",
            ["proportional", "priority_based", "bargaining"],
        )
        self.current_conflict_strategy = self.config.get(
            "initial_conflict_strategy", "proportional"
        )
        # Whether System 3* audit functionality should be active
        self.audit_active = self.config.get("audit_active", False)

        # --- Zustand & Tracking ---
        self.regions = []
        self.regional_managers = {}
        self.system4planner = getattr(
            model, "system4planner", None
        )  # Holt Referenz auf S4

        self.production_priorities = defaultdict(lambda: 1.0)
        if planning_prio and hasattr(planning_prio, "goods_priority"):
            self.production_priorities.update(planning_prio.goods_priority)
        self.critical_goods = set(self.config.get("initial_critical_goods", []))

        self.resource_flows = defaultdict(lambda: defaultdict(float))
        self.global_shortage_history = defaultdict(
            lambda: deque(maxlen=self.config.get("shortage_history_length", 20))
        )
        self.coordination_effectiveness = self.config.get("initial_effectiveness", 0.8)
        self.strategic_targets = {}
        self.audit_results = []

        # --- Optionale Komponenten ---
        self.bottleneck_identification_active = (
            self.config.get("bottleneck_identification_active", True)
            and NETWORKX_AVAILABLE
        )
        if NETWORKX_AVAILABLE:
            self.dependency_graph = nx.DiGraph()
        else:
            self.dependency_graph = None
            if self.bottleneck_identification_active:
                self.logger.warning(
                    "NetworkX nicht gefunden. Engpass-Identifikation ist deaktiviert."
                )
                self.bottleneck_identification_active = False

        # --- Delegierte Komponenten instanziieren ---
        self.inter_regional_balancer = InterRegionalResourceBalancer(self)
        self.feedback_loop_handler = OperationalFeedbackLoop(self)
        self.audit_controller = System3AuditController(
            self
        )  # Nimmt S3 Manager als Argument
        self.last_calculated_adjustments = {}

        self.logger.info(
            f"System3Manager initialisiert (Adaptiv: {self.enable_adaptive_coordination})."
        )

    def register_regional_manager(
        self, region_id: Union[int, str], manager: "RegionalSystem3Manager"
    ) -> None:
        """Registriert einen neuen RegionalSystem3Manager."""
        if region_id not in self.regions:
            self.regions.append(region_id)
        self.regional_managers[region_id] = manager
        self.logger.info(f"RegionalManager für Region {region_id} registriert.")

    def initialize_regions(
        self,
        producers_by_region: Dict[Union[int, str], List["ProducerAgent"]],
        consumers_by_region: Dict[Union[int, str], List["ConsumerAgent"]],
    ) -> None:
        """Erzeugt RegionalSystem3Manager und registriert regionale Agenten."""
        region_ids = set(producers_by_region.keys()) | set(consumers_by_region.keys())
        for rid in region_ids:
            rm = RegionalSystem3Manager(self, region_id=rid)
            for p in producers_by_region.get(rid, []):
                rm.register_producer(p.unique_id, p)
            rm.local_consumers.extend(consumers_by_region.get(rid, []))
            self.register_regional_manager(rid, rm)

    # --- Methoden für Abhängigkeitsgraph (bleiben hier) ---
    def initialize_dependency_graph(self) -> None:
        """Initialisiert (oder aktualisiert) den Abhängigkeitsgraphen."""
        if not NETWORKX_AVAILABLE or not self.bottleneck_identification_active:
            return
        self.logger.info("Analysiere Produktionsabhängigkeiten...")
        self.dependency_graph.clear()
        # --- Logik von S3M.initialize_dependency_graph hier einfügen ---
        all_producers = self.model.producers
        for producer in all_producers:
            region = getattr(producer, "region_id", "unknown")
            can_produce = getattr(producer, "can_produce", set())
            self.dependency_graph.add_node(
                producer.unique_id,
                type="producer",
                region=region,
                can_produce=list(can_produce),
            )
        self._analyze_dependencies(all_producers)
        self._identify_bottlenecks()
        self.logger.info(
            f"Abhängigkeitsgraph mit {self.dependency_graph.number_of_nodes()} Knoten und {self.dependency_graph.number_of_edges()} Kanten erstellt."
        )

    def _analyze_dependencies(self, producers: List["ProducerAgent"]) -> None:
        """Analysiert Input-Output-Abhängigkeiten."""
        if not NETWORKX_AVAILABLE or not self.dependency_graph:
            return
        # --- Logik von S3M._analyze_dependencies hier einfügen ---
        for producer in producers:
            lines = getattr(producer, "production_lines", [])
            producer_id = producer.unique_id
            for line in lines:
                output_good = getattr(line, "output_good", None)
                input_reqs = getattr(line, "input_requirements", [])
                if not output_good:
                    continue
                for req in input_reqs:
                    input_good = req.resource_type
                    req_amount = req.amount_per_unit
                    for other_producer in producers:
                        other_id = other_producer.unique_id
                        if other_id != producer_id:
                            other_can_produce = getattr(
                                other_producer, "can_produce", set()
                            )
                            if input_good in other_can_produce:
                                if not self.dependency_graph.has_edge(
                                    other_id, producer_id
                                ):
                                    self.dependency_graph.add_edge(
                                        other_id,
                                        producer_id,
                                        goods=[input_good],
                                        weight=req_amount,
                                    )
                                else:
                                    edge_data = self.dependency_graph.get_edge_data(
                                        other_id, producer_id
                                    )
                                    if input_good not in edge_data["goods"]:
                                        edge_data["goods"].append(input_good)
                                        edge_data["weight"] = max(
                                            edge_data["weight"], req_amount
                                        )

    def _identify_bottlenecks(self) -> None:
        """Identifiziert kritische Knotenpunkte im Produktionsnetzwerk."""
        if (
            not NETWORKX_AVAILABLE
            or not self.dependency_graph
            or not self.bottleneck_identification_active
        ):
            return
        self.logger.debug("Identifiziere Engpässe im Produktionsnetzwerk...")
        # --- Logik von S3M._identify_bottlenecks hier einfügen ---
        identified_bottlenecks = {}
        try:
            if self.dependency_graph.number_of_nodes() < 3:
                return
            betweenness = nx.betweenness_centrality(
                self.dependency_graph, normalized=True
            )
            in_degree = {
                node: degree for node, degree in self.dependency_graph.in_degree()
            }
            max_in_degree = max(in_degree.values()) if in_degree else 1
            for node in self.dependency_graph.nodes():
                if self.dependency_graph.nodes[node].get("type") == "producer":
                    score = (
                        betweenness.get(node, 0.0) * 0.6
                        + (in_degree.get(node, 0) / max(1, max_in_degree)) * 0.4
                    )
                    if score > 0.1:
                        identified_bottlenecks[node] = score
            top_bottlenecks = sorted(
                identified_bottlenecks.items(), key=lambda item: item[1], reverse=True
            )[:5]
            if top_bottlenecks:
                self.logger.info(
                    f"Top {len(top_bottlenecks)} Engpässe: {[(p, f'{s:.2f}') for p, s in top_bottlenecks]}"
                )
                newly_critical = set()
                for producer_id, score in top_bottlenecks:
                    node_data = self.dependency_graph.nodes[producer_id]
                    for good in node_data.get("can_produce", []):
                        if good not in self.critical_goods:
                            newly_critical.add(good)
                            self.production_priorities[good] = max(
                                self.production_priorities[good], 1.5 + score
                            )
                if newly_critical:
                    self.critical_goods.update(newly_critical)
                    self.logger.warning(
                        f"Neue kritische Güter wg. Engpass: {newly_critical}"
                    )
                    self._broadcast_priority_updates(
                        {g: self.production_priorities[g] for g in newly_critical}
                    )
        except Exception as e:
            self.logger.warning(
                f"Fehler bei Engpass-Identifikation: {e}", exc_info=False
            )

    def _broadcast_priority_updates(self, priorities: Dict[str, float]) -> None:
        """Informiert alle RegionalManager über aktualisierte Güterprioritäten."""
        self.logger.debug("Sende Prioritätsupdates an RegionalManager...")
        if not priorities:
            return
        for rm in self.regional_managers.values():
            if hasattr(rm, "receive_priority_updates"):
                rm.receive_priority_updates(priorities)

    # --- Stage Execution ---
    def step_stage(self, stage: str) -> None:
        """Führt die Aktionen für die angegebene Simulationsphase aus."""
        self.logger.debug(f"Executing stage: {stage}")

        if stage == "system3_aggregation":
            # Sammelt Status, initiiert intra-regionale Koordination
            # und führt interregionalen Ausgleich durch (delegiert).
            self.logger.info(
                "Starte operative Koordination (S3 Aggregation & Balancing)..."
            )
            self._collect_all_regional_data()
            self._coordinate_regions()  # Weist RMs an
            # Interregionaler Ausgleich (wird vom Balancer durchgeführt)
            self.inter_regional_balancer.balance_resources()
            self.logger.info(
                "Operative Koordination (Aggregation & Balancing) abgeschlossen."
            )

        elif stage == "system3_feedback":
            # Führt interne Feedback-Zyklen durch (delegiert)
            # und analysiert globales Feedback zur Anpassung eigener Parameter.
            self.logger.info("Starte S3 Feedback-Phase...")
            feedback_summary = self.feedback_loop_handler.run_cycles()
            self.analyze_feedback()  # Analysiert globales Feedback und passt S3-Params an
            self.logger.info("S3 Feedback-Phase abgeschlossen.")

        elif stage == "audit":  # System 3*
            audit_results = self.audit_controller.run_audit_cycle()
            # Audit-Ergebnisse sind jetzt im audit_controller oder in self.audit_results

        elif stage == "system3_system4_interface":  # Schnittstelle zu System 4
            self.interface_with_system4()

        # ... weitere Stages nach Bedarf ...

    # --- Kern-Koordinationslogik (vereinfacht, delegiert) ---

    def _collect_all_regional_data(self) -> None:
        """Sammelt und aggregiert *Basis*-Daten von allen RegionalManagern."""
        self.logger.debug("Sammle Daten von RegionalManagern...")
        # Diese Methode dient jetzt primär dazu, sicherzustellen, dass RMs bereit sind.
        # Die spezifischen Daten (needs, surplus, metrics) werden von den Komponenten
        # bei Bedarf direkt über die RM-Instanzen geholt.
        # Beispiel: Prüfung, ob RMs vorhanden sind.
        if not self.regional_managers:
            self.logger.warning("Keine RegionalManager registriert für Datensammlung.")
        # Hier könnte man z.B. globale Ressourcen-Momentaufnahme machen, falls benötigt
        # self.global_resource_pool.clear()
        # for rm in self.regional_managers.values():
        #     if hasattr(rm, 'collect_regional_resources'): ...

    def _coordinate_regions(self) -> None:
        """Weist jeden RegionalManager an, seine Producer zu koordinieren."""
        self.logger.debug("Starte intra-regionale Koordination...")
        current_global_targets = self.strategic_targets.get("production_targets", {})

        for rm in self.regional_managers.values():
            if hasattr(rm, "coordinate_regional_producers"):
                rm.coordinate_regional_producers(
                    current_global_targets, self.production_priorities
                )
            else:
                self.logger.error(
                    f"RegionalManager {getattr(rm, 'region_id', 'Unbekannt')} fehlt Methode 'coordinate_regional_producers'."
                )

    # --- Analyse und Anpassung der S3-Parameter (bleibt hier) ---

    def analyze_feedback(self) -> None:
        """
        Analysiert aggregiertes Feedback aus den Regionen (z.B. Stress, Knappheit)
        und passt globale S3-Koordinationsparameter adaptiv an.
        (Wird in Stage 'system3_feedback' nach dem Feedback-Loop aufgerufen).
        """
        if not self.enable_adaptive_coordination:
            self.logger.debug("Adaptive Koordination deaktiviert.")
            return

        self.logger.info(
            "Analysiere globales Feedback zur Anpassung der S3-Parameter..."
        )

        # 1. Sammle Feedback-Metriken von allen Regionen
        feedback_metrics = self._collect_feedback_metrics_for_s3_adaptation()

        # 2. Adaptive Anpassung der *eigenen* Koordinationsparameter (fairness, stability, strategy)
        self._adapt_coordination_parameters(feedback_metrics)

        # 3. Suche nach langfristigen Trends bei Engpässen
        self._analyze_shortage_trends(feedback_metrics)

        # 4. Aktualisiere die Koordinationseffektivitäts-Metrik
        self._update_coordination_effectiveness(feedback_metrics)

        self.logger.info(
            f"S3-Parameter-Analyse abgeschlossen. Neue Effektivität: {self.coordination_effectiveness:.3f}"
        )

    def _collect_feedback_metrics_for_s3_adaptation(self) -> Dict[str, Any]:
        """Sammelt spezifische Feedback-Metriken von RMs für die S3-Anpassung."""
        metrics: Dict[str, Any] = {
            "resource_shortages": defaultdict(list),  # {resource: [(region, severity)]}
            "production_deviations": defaultdict(list),  # {good: [(region, deviation)]}
            "producer_satisfaction": [],  # [(region, avg_satisfaction)]
            "regional_stress": {},  # {region: stress_level}
        }
        # --- Logik von S3M._collect_feedback_metrics hier einfügen ---
        # Iteriere durch self.regional_managers und rufe rm.generate_feedback() auf
        for region_id, rm in self.regional_managers.items():
            if hasattr(rm, "generate_feedback"):
                feedback = rm.generate_feedback()
                for resource, severity in feedback.get(
                    "resource_shortages", {}
                ).items():
                    metrics["resource_shortages"][resource].append(
                        (region_id, severity)
                    )
                for good, deviation in feedback.get(
                    "production_deviations", {}
                ).items():
                    metrics["production_deviations"][good].append(
                        (region_id, deviation)
                    )
                if "producer_satisfaction" in feedback:
                    metrics["producer_satisfaction"].append(
                        (region_id, feedback["producer_satisfaction"])
                    )
                metrics["regional_stress"][region_id] = feedback.get(
                    "stress_level", 0.0
                )
            else:
                self.logger.warning(
                    f"RM {region_id} fehlt Methode 'generate_feedback'."
                )
        return metrics

    def _adapt_coordination_parameters(self, metrics: Dict[str, Any]) -> None:
        """Passt S3-Parameter (fairness, stability, strategy) adaptiv an."""
        # --- Logik von S3M._adapt_coordination_parameters hier einfügen ---
        # Passt self.fairness_weight, self.current_conflict_strategy, self.stability_factor an
        self.logger.debug("Passe S3-Koordinationsparameter adaptiv an...")
        if metrics["producer_satisfaction"]:
            avg_satisfaction = sum(
                s for _, s in metrics["producer_satisfaction"]
            ) / len(metrics["producer_satisfaction"])
            delta_fairness = (avg_satisfaction - 0.5) * 0.1
            self.fairness_weight = np.clip(
                self.fairness_weight + delta_fairness, 0.1, 0.9
            )
        if metrics["regional_stress"]:
            avg_stress = sum(metrics["regional_stress"].values()) / len(
                metrics["regional_stress"]
            )
            # Vereinfachte Strategiewahl (Beispiel)
            if avg_stress > 0.7 and self.current_conflict_strategy != "priority_based":
                self.current_conflict_strategy = "priority_based"
            elif avg_stress < 0.4 and self.current_conflict_strategy != "proportional":
                self.current_conflict_strategy = "proportional"
        avg_stress_for_stability = sum(metrics["regional_stress"].values()) / max(
            1, len(metrics["regional_stress"])
        )
        self.stability_factor = np.clip(0.5 + avg_stress_for_stability * 0.4, 0.5, 0.9)
        self.logger.info(
            f"Angepasste S3 Parameter: Fairness={self.fairness_weight:.2f}, Strategy='{self.current_conflict_strategy}', Stability={self.stability_factor:.2f}"
        )

    def _analyze_shortage_trends(self, metrics: Dict[str, Any]) -> None:
        """Analysiert Trends bei Ressourcenknappheiten und aktualisiert kritische Güter."""
        # --- Logik von S3M._analyze_shortage_trends hier einfügen ---
        # Nutzt self.global_shortage_history und self.critical_goods
        # Ruft ggf. self._broadcast_priority_updates auf
        self.logger.debug("Analysiere Trends bei Ressourcenknappheiten...")
        newly_critical = set()
        for resource, shortages in metrics.get("resource_shortages", {}).items():
            if shortages:
                avg_severity = sum(s for _, s in shortages) / len(shortages)
                self.global_shortage_history[resource].append(avg_severity)
                history = list(self.global_shortage_history[resource])
                if len(history) >= 10:
                    recent_avg = np.mean(history[-5:])
                    previous_avg = np.mean(history[-10:-5])
                    if recent_avg > 0.4 and recent_avg > previous_avg * 1.5:
                        if resource not in self.critical_goods:
                            newly_critical.add(resource)
        if newly_critical:
            self.critical_goods.update(newly_critical)
            self.logger.warning(f"Neue kritische Güter (Trend): {newly_critical}")
            prio_updates = {}
            for good in newly_critical:
                self.production_priorities[good] = max(
                    self.production_priorities.get(good, 1.0), 2.0
                )
                prio_updates[good] = self.production_priorities[good]
            if prio_updates:
                self._broadcast_priority_updates(prio_updates)

    def _update_coordination_effectiveness(self, metrics: Dict[str, Any]) -> None:
        """Aktualisiert die Metrik für die S3-Koordinationseffektivität."""
        # --- Logik von S3M._update_coordination_effectiveness hier einfügen ---
        # Nutzt Metriken und self.coordination_effectiveness
        plan_score, resource_score, satisfaction_score = 1.0, 1.0, 0.5  # Defaults
        if metrics["production_deviations"]:
            total_deviation = sum(
                abs(dev)
                for good_devs in metrics["production_deviations"].values()
                for _, dev in good_devs
            )
            count = sum(len(devs) for devs in metrics["production_deviations"].values())
            avg_deviation = total_deviation / max(1, count) if count > 0 else 0.0
            plan_score = max(0.0, 1.0 - avg_deviation * 2.0)
        if metrics["resource_shortages"]:
            total_severity = sum(
                sev
                for res_shorts in metrics["resource_shortages"].values()
                for _, sev in res_shorts
            )
            count = sum(
                len(shorts) for shorts in metrics["resource_shortages"].values()
            )
            avg_severity = total_severity / max(1, count) if count > 0 else 0.0
            resource_score = max(0.0, 1.0 - avg_severity)
        if metrics["producer_satisfaction"]:
            satisfaction_score = sum(
                s for _, s in metrics["producer_satisfaction"]
            ) / len(metrics["producer_satisfaction"])
        new_effectiveness = (
            0.5 * plan_score + 0.3 * resource_score + 0.2 * satisfaction_score
        )
        smoothing_factor = 0.3
        self.coordination_effectiveness = (
            1 - smoothing_factor
        ) * self.coordination_effectiveness + smoothing_factor * new_effectiveness
        self.coordination_effectiveness = np.clip(
            self.coordination_effectiveness, 0.0, 1.0
        )

    # --- Methoden für die Schnittstellen S4 / S2 ---

    def interface_with_system4(self) -> None:
        """Sendet operativen Bericht an System 4 und empfängt Direktiven."""
        # --- Logik von S3M.interface_with_system4 hier einfügen ---
        # Ruft _prepare_operational_report, S4.receive_operational_report,
        # S4.get_strategic_directives, _process_strategic_directives auf.
        # Default assumption: interface should run unless explicitly disabled
        if getattr(self.model, "vsm_on", True) is False or not self.system4planner:
            return
        self.logger.info("Schnittstelle System 3 <-> System 4...")
        try:
            report = self._prepare_operational_report()
            self.system4planner.receive_operational_report(report)
            self.logger.info("Operativer Bericht an System 4 gesendet.")
        except Exception as e:
            self.logger.error(
                f"Fehler beim Senden des Berichts an S4: {e}", exc_info=True
            )
        try:
            directives = self.system4planner.get_strategic_directives()
            if directives:
                self._process_strategic_directives(directives)
                self.logger.info("Strategische Direktiven von S4 verarbeitet.")
            else:
                self.logger.debug("Keine neuen Direktiven von S4.")
        except Exception as e:
            self.logger.error(
                f"Fehler beim Empfangen/Verarbeiten von S4 Direktiven: {e}",
                exc_info=True,
            )

    def _prepare_operational_report(self) -> "OperationalReportS3":
        """Stellt einen strukturierten operativen Bericht für System 4 zusammen."""
        # --- Logik von S3M._prepare_operational_report hier einfügen ---
        # Nutzt _collect_global_production_statistics, _calculate_global_plan_fulfillment etc.
        self.logger.debug("Erstelle operativen Bericht für System 4...")
        current_step = getattr(self.model, "current_step", -1)
        production_stats = self._collect_global_production_statistics()
        regional_statuses = {}
        for region_id, rm in self.regional_managers.items():
            if hasattr(rm, "get_summary_status"):
                regional_statuses[region_id] = rm.get_summary_status()
        current_bottlenecks = self._identify_current_bottlenecks()
        plan_fulfillment = self._calculate_global_plan_fulfillment()

        # Annahme: OperationalReportS3 ist eine Datenklasse/Pydantic-Modell
        from .communication_contracts import OperationalReportS3  # Import annehmen

        report = OperationalReportS3(
            step=current_step,
            source_system="System3",
            target_system="System4",  # Wichtig für Basisklasse
            coordination_effectiveness=self.coordination_effectiveness,
            critical_resources=list(self.critical_goods),
            resource_levels=self.get_global_resource_snapshot(),  # Momentaufnahme
            production_statistics=production_stats,
            regional_status=regional_statuses,
            bottlenecks=current_bottlenecks,
            plan_fulfillment_summary=plan_fulfillment,
        )
        return report

    def _process_strategic_directives(self, directives: "StrategicDirectiveS4") -> None:
        """Verarbeitet strategische Direktiven von System 4."""
        # --- Logik von S3M._process_strategic_directives hier einfügen ---
        # Passt self.strategic_targets, self.critical_goods, self.autonomy_level, self.fairness_weight an.
        # Ruft rm.receive_production_targets und self._broadcast_priority_updates auf.
        self.logger.info(
            f"Verarbeite strategische Direktiven von S4 (Step {directives.step})..."
        )
        self.strategic_targets = directives.dict()  # Speichere als Dict

        if directives.production_targets is not None:
            # ... Logik zum Mischen mit Autonomiegrad ...
            merged_targets = self._merge_targets_with_autonomy(
                directives.production_targets
            )
            self.strategic_targets["production_targets"] = (
                merged_targets  # Aktualisiere im State
            )
            for rm in self.regional_managers.values():
                if hasattr(rm, "receive_production_targets"):
                    rm.receive_production_targets(merged_targets)

        if directives.critical_goods_update is not None:
            # ... Logik zum Aktualisieren von self.critical_goods ...
            updated_critical = set(directives.critical_goods_update)
            # ... (Logik für newly_critical, no_longer_critical und Prio-Updates wie im Original) ...
            pass  # Platzhalter für ausführliche Logik

        if directives.system3_autonomy_level is not None:
            self.autonomy_level = np.clip(directives.system3_autonomy_level, 0.1, 0.9)
            self.logger.info(
                f"System 3 Autonomiegrad angepasst auf {self.autonomy_level:.2f}"
            )

        if directives.resource_allocation_guidelines:
            new_fairness = directives.resource_allocation_guidelines.get(
                "target_fairness_weight"
            )
            if new_fairness is not None:
                self.fairness_weight = np.clip(new_fairness, 0.1, 0.9)
                self.logger.info(
                    f"Fairness-Gewichtung von S4 angepasst auf {self.fairness_weight:.2f}"
                )

    def apply_resolution_directive(
        self, directive: "ConflictResolutionDirectiveS2"
    ) -> bool:
        """Wendet eine Konfliktlösungs-Direktive von System 2 an."""
        # --- Logik von S3M.apply_resolution_directive hier einfügen ---
        # Führt Aktionen aus, typischerweise Ressourcentransfers über RMs.
        if directive.target_system != "System3":
            self.logger.warning(
                f"Erhalte S2-Direktive für falsches System: {directive.target_system}"
            )
            return False
        action = directive.action_type
        payload = directive.payload
        self.logger.info(f"Wende Konfliktlösungs-Direktive von S2 an: Typ='{action}'")
        try:
            if action == "resource_transfer":
                # Nutze execute_inter_regional_transfer für Konsistenz
                success = self.execute_inter_regional_transfer(
                    from_region=payload["from_region"],
                    to_region=payload["to_region"],
                    resource=payload["resource"],
                    amount=payload["amount"],
                )
                if success:
                    self.logger.info(
                        f"Konfliktlösung: Transfer via S2-Direktive erfolgreich."
                    )
                else:
                    self.logger.error(
                        f"Konfliktlösung: Transfer via S2-Direktive fehlgeschlagen."
                    )
                return success
            else:
                self.logger.warning(
                    f"Unbekannter action_type '{action}' in S2-Direktive."
                )
                return False
        except KeyError as e:
            self.logger.error(f"Fehlender Schlüssel in S2-Direktive Payload: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Fehler beim Anwenden der S2-Direktive: {e}")
            return False

    # --- Hilfsmethoden für S3 Manager selbst ---

    def _merge_targets_with_autonomy(
        self, new_targets: Dict[str, float]
    ) -> Dict[str, float]:
        """Mischt neue Ziele von S4 mit alten Zielen basierend auf Autonomiegrad."""
        merged_targets = {}
        old_targets = self.strategic_targets.get("production_targets", {})
        all_goods = set(old_targets.keys()) | set(new_targets.keys())
        for good in all_goods:
            old_val = old_targets.get(good, 0.0)
            new_val = new_targets.get(
                good, old_val
            )  # Behalte altes, wenn kein neues Ziel
            merged_targets[good] = (
                self.autonomy_level * old_val + (1.0 - self.autonomy_level) * new_val
            )
        return merged_targets

    def _collect_global_production_statistics(self) -> Dict[str, Any]:
        """Sammelt aggregierte Produktionsstatistiken von allen Regionen."""
        # --- Logik von S3M._collect_global_production_statistics hier einfügen ---
        stats: Dict[str, Any] = {
            "total_output_by_good": defaultdict(float),
            "total_capacity_utilization": 0.0,
            "regional_output": defaultdict(lambda: defaultdict(float)),
        }
        # ... (Iteration über RMs, Aufruf von rm.collect_production_statistics) ...
        num_regions = len(self.regional_managers)
        total_util = 0.0
        if num_regions > 0:
            for region_id, rm in self.regional_managers.items():
                if hasattr(rm, "collect_production_statistics"):
                    regional_stats = rm.collect_production_statistics()
                    for good, amount in regional_stats.get(
                        "output_by_good", {}
                    ).items():
                        stats["total_output_by_good"][good] += amount
                        stats["regional_output"][region_id][good] = amount
                    total_util += regional_stats.get("capacity_utilization", 0.0)
            stats["total_capacity_utilization"] = total_util / num_regions
        stats["total_output_by_good"] = dict(stats["total_output_by_good"])
        stats["regional_output"] = {
            k: dict(v) for k, v in stats["regional_output"].items()
        }
        return stats

    def _calculate_global_plan_fulfillment(self) -> Dict[str, float]:
        """Berechnet die durchschnittliche Planerfüllung pro Gut über alle Regionen."""
        # --- Logik von S3M._calculate_global_plan_fulfillment hier einfügen ---
        fulfillment_summary = defaultdict(list)
        # ... (Iteration über RMs, Aufruf von rm.calculate_production_fulfillment) ...
        for rm in self.regional_managers.values():
            if hasattr(rm, "calculate_production_fulfillment"):
                fulfill_data = rm.calculate_production_fulfillment()
                for good, fulfillment in fulfill_data.items():
                    fulfillment_summary[good].append(fulfillment)
        avg_fulfillment = {
            good: sum(values) / len(values)
            for good, values in fulfillment_summary.items()
            if values
        }
        return avg_fulfillment

    def _identify_current_bottlenecks(self) -> Dict[str, Any]:
        """Identifiziert aktuelle Bottlenecks."""
        # --- Logik von S3M._identify_current_bottlenecks hier einfügen ---
        # (Kann NetworkX nutzen oder heuristisch sein)
        if (
            self.dependency_graph
            and NETWORKX_AVAILABLE
            and self.bottleneck_identification_active
        ):
            # Verwende die Ergebnisse von _identify_bottlenecks, die bereits Attribute aktualisiert haben
            # Hier nur eine einfache Rückgabe als Beispiel
            top_b = {
                n: d["score"]
                for n, d in self.dependency_graph.nodes(data=True)
                if d.get("score", 0) > 0.1
            }
            return {
                "identified_by_graph": sorted(
                    top_b.items(), key=lambda item: item[1], reverse=True
                )[:5]
            }
        return {}  # Placeholder

    def get_global_resource_snapshot(self) -> Dict[str, float]:
        """Erstellt eine Momentaufnahme der globalen Ressourcenbestände (aggregiert von RMs)."""
        snapshot = defaultdict(float)
        for rm in self.regional_managers.values():
            if hasattr(rm, "collect_regional_resources"):
                regional_resources = rm.collect_regional_resources()
                for resource, amount in regional_resources.items():
                    snapshot[resource] += amount
        return dict(snapshot)

    def _find_regional_manager_for_producer(
        self, producer_id: str
    ) -> Optional["RegionalSystem3Manager"]:
        """Findet den zuständigen RegionalSystem3Manager für einen Producer."""
        # Diese Methode wird von AuditController benötigt und bleibt hier
        agent = self.model.agents_by_id.get(producer_id)
        if agent and hasattr(agent, "region_id"):
            region_id = agent.region_id
            return self.regional_managers.get(region_id)
        self.logger.debug(
            f"Konnte RM für Producer {producer_id} nicht finden (Agent oder Region unbekannt)."
        )
        return None

    def get_all_regional_statuses(self) -> Dict[Union[int, str], Dict[str, Any]]:
        """Sammelt zusammenfassende Status von allen RMs (für S2)."""
        statuses = {}
        for region_id, rm in self.regional_managers.items():
            if hasattr(rm, "get_summary_status"):
                region_data = rm.get_summary_status()
                region_data["agent_object"] = rm
                statuses[region_id] = region_data
            else:
                self.logger.warning(
                    f"RM {region_id} fehlt Methode 'get_summary_status'."
                )
                statuses[region_id] = {}  # Leerer Status
        return statuses

    # --- Hilfsmethoden für delegierte Komponenten ---
    # Diese Methoden werden von den Komponenten aufgerufen, um Aktionen auszuführen

    def apply_priority_updates(self, priority_updates: Dict[str, float]) -> int:
        """
        Wendet Prioritätsänderungen an und broadcastet sie.
        Wird vom OperationalFeedbackLoop aufgerufen.
        """
        applied_count = 0
        updated_priorities_for_broadcast = {}
        for good, new_priority in priority_updates.items():
            old_priority = self.production_priorities.get(good, 1.0)
            # Wende Änderung an (hier wird keine separate Prüfung auf Signifikanz mehr gemacht, da dies im FeedbackLoop geschah)
            self.production_priorities[good] = new_priority
            updated_priorities_for_broadcast[good] = new_priority
            self.logger.info(
                f"Priorität für {good} auf {new_priority:.2f} gesetzt (via Feedback Loop)."
            )
            applied_count += 1

        if updated_priorities_for_broadcast:
            self._broadcast_priority_updates(updated_priorities_for_broadcast)
        return applied_count

    def send_target_factor_feedback(self, target_factors: Dict[str, float]) -> bool:
        """
        Sendet Zielanpassungsfaktoren an S4.
        Wird vom OperationalFeedbackLoop aufgerufen.
        """
        if not self.system4planner or not hasattr(
            self.system4planner, "receive_directive_or_feedback"
        ):
            self.logger.warning(
                "Kann Ziel-Feedback nicht an S4 senden (S4 Planner oder Methode fehlt)."
            )
            return False
        try:
            # Erzeuge eine generische Feedback-Nachricht (Annahme: SystemFeedback existiert)
            from .communication_contracts import SystemFeedback

            feedback_payload = {
                "type": "target_adjustment_factors",
                "factors": target_factors,
                "rationale": "Operational feedback based on plan fulfillment deviation (S3 internal loop)",
            }
            feedback_msg = SystemFeedback(
                source_system="System3",
                target_system="System4",
                step=self.model.current_step,
                feedback_type="operational_adjustment_proposal",
                payload=feedback_payload,
            )

            # Sende über die Routing-Funktion des Models (konsistenter)
            if hasattr(self.model, "route_directive"):
                self.model.route_directive(feedback_msg)
                self.logger.info(
                    f"Zielanpassungsfaktoren an System 4 gesendet: {target_factors}"
                )
                return True
            else:
                self.logger.error(
                    "Model hat keine 'route_directive' Methode zum Senden von Feedback."
                )
                return False

        except ImportError:
            self.logger.error(
                "Konnte 'SystemFeedback' nicht importieren. Senden der Ziel-Faktoren fehlgeschlagen."
            )
            return False
        except Exception as e:
            self.logger.error(f"Fehler beim Senden des Ziel-Feedbacks an S4: {e}")
            return False

    # ------------------------------------------------------------------
    # Kompatibilitäts-Helper für Tests
    # ------------------------------------------------------------------
    def _apply_resource_redistribution(self, redistribution: Dict[str, Dict[str, Any]]) -> None:
        """Wendet berechnete Ressourcenumverteilungen an."""
        for resource, data in redistribution.items():
            self.execute_inter_regional_transfer(
                from_region=data.get("from_region"),
                to_region=data.get("to_region"),
                resource=resource,
                amount=data.get("amount", 0.0),
            )

    def _apply_priority_adjustments(self, adjustments: Dict[str, float]) -> None:
        """Aktualisiert Produktionsprioritäten gemäss Feedback."""
        if not adjustments:
            return
        self.production_priorities.update(adjustments)
        self._broadcast_priority_updates(adjustments)

    def _send_target_feedback_to_s4(self, factors: Dict[str, float]) -> None:
        """Leitet Zielanpassungsfaktoren an System 4 weiter."""
        if factors:
            self.send_target_factor_feedback(factors)

    def _run_internal_feedback_cycle(self) -> None:
        """Ausführung eines Feedback-Loops (für Unit-Tests)."""
        summary = self.feedback_loop_handler.run_cycles()
        self.last_calculated_adjustments = self.feedback_loop_handler.last_calculated_adjustments
        self._apply_resource_redistribution(self.last_calculated_adjustments.get("resource_redistribution", {}))
        self._apply_priority_adjustments(self.last_calculated_adjustments.get("priority_adjustments", {}))
        self._send_target_feedback_to_s4(self.last_calculated_adjustments.get("target_adjustments", {}))

    def execute_inter_regional_transfer(
        self, from_region, to_region, resource, amount
    ) -> bool:
        """
        Führt einen einzelnen Transfer zwischen Regionen aus.
        Wird vom InterRegionalResourceBalancer oder OperationalFeedbackLoop aufgerufen.
        """
        from_rm = self.regional_managers.get(from_region)
        to_rm = self.regional_managers.get(to_region)
        if (
            from_rm
            and to_rm
            and hasattr(from_rm, "transfer_resource")
            and hasattr(to_rm, "receive_resource")
        ):
            success = from_rm.transfer_resource(resource, amount, to_region)
            if success:
                to_rm.receive_resource(resource, amount)
                # Optional: Update resource_flows tracking
                self.resource_flows[resource][f"{from_region}->{to_region}"] = (
                    self.resource_flows[resource].get(
                        f"{from_region}->{to_region}", 0.0
                    )
                    + amount
                )
                self.logger.info(
                    f"Inter-regionaler Transfer ausgeführt: {amount:.2f} {resource} von R{from_region} nach R{to_region}"
                )
                return True
            else:
                self.logger.warning(
                    f"Inter-regionaler Transfer von {amount:.2f} {resource} (R{from_region} -> R{to_region}) fehlgeschlagen (RM hat abgelehnt)."
                )
                return False
        else:
            self.logger.error(
                f"Inter-regionaler Transfer fehlgeschlagen: RM für R{from_region} oder R{to_region} nicht gefunden oder Methoden fehlen."
            )
            return False

    # Wrapper-Methoden für Tests (Kompatibilität mit älterer API)

    def _calculate_inter_regional_transfers(
        self,
        resource: str,
        needy: Dict[Union[int, str], float],
        suppliers: Dict[Union[int, str], float],
        priorities: Dict[Union[int, str], float],
        total_transfer: float,
    ) -> List[Tuple[Union[int, str], Union[int, str], float]]:
        """Berechnet Transfers je nach aktiviertem Verfahren."""
        if self.enable_advanced_optimization:
            return self._optimize_inter_regional_transfers_scipy(
                resource, needy, suppliers, priorities, total_transfer
            )
        return self._proportional_inter_regional_transfers(
            resource, needy, suppliers, priorities, total_transfer
        )

    def _optimize_inter_regional_transfers_scipy(
        self,
        resource: str,
        needy: Dict[Union[int, str], float],
        suppliers: Dict[Union[int, str], float],
        priorities: Dict[Union[int, str], float],
        total_transfer: float,
    ) -> List[Tuple[Union[int, str], Union[int, str], float]]:
        """Proxy for the optimizer used in tests."""
        return self.inter_regional_balancer._optimize_transfers_scipy(
            resource, needy, suppliers, priorities, total_transfer
        )

    def _proportional_inter_regional_transfers(
        self,
        resource: str,
        needy: Dict[Union[int, str], float],
        suppliers: Dict[Union[int, str], float],
        priorities: Dict[Union[int, str], float],
        total_transfer: float,
    ) -> List[Tuple[Union[int, str], Union[int, str], float]]:
        """Proxy for the proportional fallback used in tests."""
        return self.inter_regional_balancer._proportional_transfers(
            resource, needy, suppliers, priorities, total_transfer
        )


# -------------------------------------------------------------------------
# RegionalSystem3Manager: Verwaltet Producers einer Region
# (Diese Klasse bleibt weitgehend unverändert, da ihre Methoden von S3M und
#  den neuen Komponenten aufgerufen werden. Sie führt die Aktionen lokal aus.)
# -------------------------------------------------------------------------


class RegionalSystem3Manager:
    """
    Verwaltet Producers und Ressourcen innerhalb einer bestimmten Region.
    Agiert als lokaler Koordinator für das übergeordnete System 3.

    Verantwortlich für:
    - Registrierung und Verwaltung der Producer in der Region.
    - Empfang globaler Ziele und Prioritäten von System 3.
    - Berechnung des regionalen Ressourcenbedarfs.
    - Allokation der *regional verfügbaren* Ressourcen an die Producer.
    - Zuweisung finaler, realisierbarer Produktionsziele an Producer.
    - Berechnung und Bereitstellung regionaler Performance-Metriken und Feedbacks.
    - Durchführung von Audits auf Anweisung des AuditControllers.
    - Anwendung von Strafen auf Producer.
    - Ausführung von Ressourcentransfers zu/von anderen Regionen auf Anweisung von S3M.

    Attributes:
        parent (System3Manager): Referenz auf das übergeordnete System 3.
        region_id (Union[int, str]): ID dieser Region.
        logger (logging.Logger): Logger Instanz.
        producers (Dict[str, 'ProducerAgent']): Producer in dieser Region {producer_id: instance}.
        producer_stats (Dict[str, Dict[str, Any]]): Basis-Statistiken pro Producer.
        resource_pool (DefaultDict[str, float]): Aktuell verfügbare Ressourcen in der Region (wird von S3M gefüllt/geleert).
        resource_needs (DefaultDict[str, float]): Berechneter Ressourcenbedarf der Region für den nächsten Zyklus.
        production_targets (DefaultDict[str, float]): Produktionsziele für diese Region {gut: zielwert}.
        good_priorities (DefaultDict[str, float]): Prioritäten für Güter in dieser Region.
        production_fulfillment (DefaultDict[str, float]): Erfüllungsgrad der Ziele {gut: ratio}.
        resource_utilization (DefaultDict[str, float]): Auslastungsgrad der zugewiesenen Ressourcen {resource: ratio}.
        stress_level (float): Maß für die Anspannung der Ressourcensituation (0-1).
        producer_penalties (DefaultDict[str, Dict[str, float]]): Temporäre Strafen für Producer {producer_id: {penalty_type: factor}}.
        shortage_history (DefaultDict[str, Deque[float]]): Historie von Ressourcenknappheiten.
        production_history (DefaultDict[str, Deque[float]]): Historie der Produktionserfüllung.
    """

    parent: "System3Manager"
    region_id: Union[int, str]
    logger: logging.Logger
    producers: Dict[str, "ProducerAgent"]
    local_consumers: List["ConsumerAgent"]
    producer_stats: Dict[str, Dict[str, Any]]
    resource_pool: DefaultDict[str, float]
    resource_needs: DefaultDict[str, float]
    production_targets: DefaultDict[str, float]
    good_priorities: DefaultDict[str, float]
    production_fulfillment: DefaultDict[str, float]
    resource_utilization: DefaultDict[str, float]
    stress_level: float
    producer_penalties: DefaultDict[str, Dict[str, float]]
    shortage_history: DefaultDict[str, Deque[float]]
    production_history: DefaultDict[str, Deque[float]]

    def __init__(self, parent_system3: "System3Manager", region_id: Union[int, str]):
        """
        Initialisiert den RegionalManager.

        Args:
            parent_system3: Referenz auf das übergeordnete System 3.
            region_id: ID dieser Region.
        """
        self.parent = parent_system3
        self.region_id = region_id
        self.logger = parent_system3.logger.getChild(
            f"RM.{region_id}"
        )  # Eigener Sub-Logger

        self.producers = {}
        self.local_consumers = []
        self.producer_stats = {}
        self.resource_pool = defaultdict(float)
        self.resource_needs = defaultdict(float)
        self.production_targets = defaultdict(float)
        self.good_priorities = defaultdict(
            lambda: 1.0
        )  # Kopiere globale Prios initial?
        if hasattr(self.parent, "production_priorities"):
            self.good_priorities.update(self.parent.production_priorities)

        self.production_fulfillment = defaultdict(float)
        self.resource_utilization = defaultdict(float)
        self.stress_level = 0.0
        self.producer_penalties = defaultdict(dict)

        # Lese History Length aus Config (über Parent)
        history_len = getattr(self.parent.config, "regional_history_length", 10)
        self.shortage_history = defaultdict(lambda: deque(maxlen=history_len))
        self.production_history = defaultdict(lambda: deque(maxlen=history_len))

        self.logger.debug(
            f"RegionalSystem3Manager für Region {self.region_id} initialisiert."
        )

    # --- Methoden für Verwaltung ---
    def register_producer(self, producer_id: str, producer: "ProducerAgent") -> None:
        """Registriert einen Producer in dieser Region."""
        self.producers[producer_id] = producer
        # Sammle initiale Statistiken (können sich ändern!)
        self.producer_stats[producer_id] = {
            "initial_capacity": getattr(producer, "productive_capacity", 0.0),
            "can_produce": list(getattr(producer, "can_produce", set())),
            "tech_level": getattr(producer, "tech_level", 1.0),
        }
        self.logger.info(
            f"Region {self.region_id}: Producer {producer_id} registriert."
        )

    # Kompatibilitätsmethoden
    def add_producer(self, producer: "ProducerAgent") -> None:
        if producer.unique_id not in self.producers:
            self.register_producer(producer.unique_id, producer)

    def add_consumer(self, consumer: "ConsumerAgent") -> None:
        if consumer not in self.local_consumers:
            self.local_consumers.append(consumer)

    def coordinate_producers(self) -> None:
        """Wrapper zur Koordination innerhalb der Region."""
        self.coordinate_regional_producers(
            self.parent.strategic_targets.get("production_targets", {}),
            self.parent.production_priorities,
        )

    def receive_priority_updates(self, priorities: Dict[str, float]) -> None:
        """Empfängt und verarbeitet globale Prioritätsupdates vom S3 Manager."""
        self.good_priorities.update(priorities)
        self.logger.debug(
            f"Region {self.region_id}: Güterprioritäten aktualisiert ({len(priorities)} Güter)."
        )

    def receive_production_targets(self, global_targets: Dict[str, float]) -> None:
        """
        Empfängt globale Produktionsziele vom übergeordneten System und
        berechnet/aktualisiert die regionalen Ziele.
        """
        # --- Logik von RM.receive_production_targets hier einfügen ---
        num_regions = len(self.parent.regions) if self.parent.regions else 1
        regional_share = 1.0 / max(1, num_regions)
        producible_here = set().union(
            *(getattr(p, "can_produce", set()) for p in self.producers.values())
        )
        updated_targets = {}
        for good, global_target in global_targets.items():
            if good in producible_here:
                regional_target = (
                    global_target * regional_share
                )  # TODO: intelligentere Verteilung?
                current_target = self.production_targets.get(good, regional_target)
                adapt_rate = 1.0 - self.parent.stability_factor
                final_target = (
                    current_target * (1.0 - adapt_rate) + regional_target * adapt_rate
                )
                self.production_targets[good] = max(0.0, final_target)
                updated_targets[good] = self.production_targets[good]
        if updated_targets:
            self.logger.info(
                f"Region {self.region_id}: Produktionsziele aktualisiert für {len(updated_targets)} Güter."
            )

    # --- Kern-Koordinationsmethode (wird von S3M aufgerufen) ---
    def coordinate_regional_producers(
        self, global_targets: Dict[str, float], global_priorities: Dict[str, float]
    ) -> None:
        """
        Koordiniert die Producer innerhalb dieser Region.
        Wird typischerweise von System3Manager aufgerufen.
        """
        self.logger.info(
            f"Region {self.region_id}: Koordiniere {len(self.producers)} Producer..."
        )

        # 0. Aktualisiere lokale Prioritäten mit globalen Vorgaben
        self.good_priorities.update(global_priorities)

        # 1. Regionale Ziele ableiten/aktualisieren (falls S4 -> S3 -> RM geht)
        #    Wenn S4 die Ziele direkt setzt, ist dieser Schritt ggf. nicht nötig.
        #    Hier gehen wir davon aus, dass receive_production_targets die Ziele setzt.

        # 2. Ressourcenbedarf für regionale Ziele berechnen
        self._calculate_resource_requirements()

        # 2.5 Arbeit allokieren
        self._allocate_labor()

        # 3. Verfügbare Ressourcen (aus self.resource_pool) auf Producer verteilen
        self._allocate_resources_to_producers()

        # 4. Konkrete Produktionsziele an Producer übermitteln (basierend auf Allokation)
        self._assign_producer_targets()

        # 5. Metriken für diese Region berechnen/aktualisieren
        self._update_regional_metrics()
        self.logger.info(
            f"Region {self.region_id}: Koordination abgeschlossen. Stress: {self.stress_level:.2f}"
        )

    # --- Private Hilfsmethoden für Koordination (wie im Original) ---
    def _calculate_resource_requirements(self) -> None:
        """Berechnet den Ressourcenbedarf der Region basierend auf den regionalen Zielen."""
        # --- Logik von RM._calculate_resource_requirements hier einfügen ---
        self.logger.debug(f"Region {self.region_id}: Berechne Ressourcenbedarf...")
        self.resource_needs.clear()
        total_regional_needs = defaultdict(float)
        for producer in self.producers.values():
            if hasattr(producer, "calculate_resource_needs"):
                producer_needs = producer.calculate_resource_needs()
                for resource, amount in producer_needs.items():
                    total_regional_needs[resource] += amount
            # ... (Fallback Logik falls Methode fehlt, ggf. entfernen) ...
        self.resource_needs = total_regional_needs
        if self.resource_needs:
            top_needs = sorted(
                self.resource_needs.items(), key=lambda x: x[1], reverse=True
            )[:3]
            self.logger.debug(
                f"Region {self.region_id}: Top Bedarfe: {[(r, f'{a:.1f}') for r,a in top_needs]}"
            )

    def _allocate_labor(self) -> None:
        """Weist Arbeitskräfte (Consumer) den Produzenten zu."""
        self.logger.debug(f"Region {self.region_id}: Alloziiere Arbeitskräfte...")

        available_labor_pool = []
        for consumer in self.local_consumers:
            offering = consumer.get_labor_offering()
            offering['consumer_id'] = consumer.unique_id
            available_labor_pool.append(offering)

        total_labor_demand = 0.0
        producer_labor_needs = {}
        for pid, producer in self.producers.items():
            labor_needed = 0.0
            for line in producer.production_lines:
                target = producer.production_target.get(line.output_good, 0)
                labor_needed += (target / line.effective_efficiency) * line.labor_requirement
            producer_labor_needs[pid] = labor_needed
            total_labor_demand += labor_needed

        total_available_hours = sum(l['hours'] for l in available_labor_pool)

        if total_labor_demand > 0:
            allocation_factor = min(1.0, total_available_hours / total_labor_demand)
            for pid, demand_hours in producer_labor_needs.items():
                allocated_hours = demand_hours * allocation_factor
                self.producers[pid].assigned_labor_hours = allocated_hours

    def _allocate_resources_to_producers(self) -> None:
        """Verteilt die in `self.resource_pool` verfügbaren Ressourcen auf die Producer."""
        # --- Logik von RM._allocate_resources_to_producers hier einfügen ---
        # Nutzt _calculate_producer_priority und _calculate_single_resource_allocation
        self.logger.debug(
            f"Region {self.region_id}: Alloziiere Ressourcen an Producer..."
        )
        for resource, total_available in self.resource_pool.items():
            if total_available <= 1e-6:
                continue
            producer_resource_needs = {}
            producer_priorities = {}
            total_need_for_resource = 0.0
            for pid, p in self.producers.items():
                need = getattr(p, "calculate_resource_needs", lambda: {})().get(
                    resource, 0.0
                )
                if need > 0:
                    producer_resource_needs[pid] = need
                    total_need_for_resource += need
                    producer_priorities[pid] = self._calculate_producer_priority(
                        p, resource
                    )
            if total_need_for_resource <= 1e-6:
                continue

            for (
                pid
            ) in (
                producer_resource_needs
            ):  # Reset Zuweisung im Producer für diese Ressource
                if hasattr(self.producers[pid], "resource_stock"):
                    self.producers[pid].resource_stock[resource] = 0.0

            if total_need_for_resource <= total_available:
                for pid, need in producer_resource_needs.items():
                    self.producers[pid].resource_stock[resource] = need
                self.resource_utilization[resource] = total_need_for_resource / max(
                    1e-9, total_available
                )
                self.shortage_history[resource].append(0.0)
            else:
                allocations = self._calculate_single_resource_allocation(
                    resource,
                    total_available,
                    producer_resource_needs,
                    producer_priorities,
                )
                for pid, amount in allocations.items():
                    self.producers[pid].resource_stock[resource] = amount
                self.resource_utilization[resource] = 1.0
                self.shortage_history[resource].append(
                    1.0 - total_available / total_need_for_resource
                )

    def _calculate_producer_priority(
        self, producer: "ProducerAgent", resource: str
    ) -> float:
        """Berechnet die Priorität eines Producers für eine bestimmte Ressource."""
        # --- Logik von RM._calculate_producer_priority hier einfügen ---
        max_prio = 0.0
        has_req = False
        for line in getattr(producer, "production_lines", []):
            if any(
                req.resource_type == resource
                for req in getattr(line, "input_requirements", [])
            ):
                has_req = True
                output_good = getattr(line, "output_good", None)
                line_prio = getattr(line, "priority", 1.0)
                good_prio = self.good_priorities.get(output_good, 1.0)
                max_prio = max(max_prio, line_prio * good_prio)
        if not has_req:
            return 0.0
        penalty_factor = self.producer_penalties.get(producer.unique_id, {}).get(
            "resource_allocation", 1.0
        )
        return max(0.1, max_prio * penalty_factor)

    def _calculate_single_resource_allocation(
        self,
        resource: str,
        total_available: float,
        needs: Dict[str, float],
        priorities: Dict[str, float],
    ) -> Dict[str, float]:
        """Berechnet die Allokation für EINE knappe Ressource."""
        # --- Logik von RM._calculate_single_resource_allocation hier einfügen ---
        # Nutzt parent.current_conflict_strategy und parent.fairness_weight
        allocations = {}
        total_need = sum(needs.values())
        strategy = self.parent.current_conflict_strategy
        fairness = self.parent.fairness_weight
        # ... (Implementierung der Strategien wie im Original RM) ...
        # Beispiel für proportional:
        if strategy == "proportional":
            if total_need > 0:
                for pid, need in needs.items():
                    allocations[pid] = (need / total_need) * total_available
        # ... (Implementierung für priority_based und bargaining) ...
        else:  # Default = bargaining
            total_priority_sum = sum(
                priorities.get(pid, 0.0) * needs[pid] for pid in needs
            )
            if total_need > 0 and total_priority_sum > 0:
                for pid, need in needs.items():
                    prop_share = need / total_need
                    prio_share = (priorities.get(pid, 0.0) * need) / max(
                        1e-9, total_priority_sum
                    )
                    combined_share = (
                        fairness * prop_share + (1.0 - fairness) * prio_share
                    )
                    allocations[pid] = combined_share * total_available
            elif total_need > 0:  # Nur proportional möglich
                for pid, need in needs.items():
                    allocations[pid] = (need / total_need) * total_available

        # Capping
        final_allocations = {
            pid: min(amount, needs[pid]) for pid, amount in allocations.items()
        }
        # TODO: Ggf. Überschuss durch Capping neu verteilen
        return final_allocations

    def _assign_producer_targets(self) -> None:
        """Überträgt die finalen, realisierbaren Produktionsziele an die Producer."""
        # --- Logik von RM._assign_producer_targets hier einfügen ---
        self.logger.debug(
            f"Region {self.region_id}: Weise finale Produktionsziele an Producer zu..."
        )
        for pid, p in self.producers.items():
            max_possible_prod = defaultdict(float)
            # ... (Berechnung max_possible_production basierend auf p.resource_stock) ...
            if hasattr(p, "production_lines") and p.production_lines:
                for line in p.production_lines:
                    output_good = getattr(line, "output_good", None)
                    line_limit = float('inf')
                    if not output_good or not getattr(line, "is_active", True):
                        continue
                    input_reqs = line.calculate_inputs_for_output(1.0)
                    for res, req_unit in input_reqs.items():
                        if req_unit > 1e-9:
                            line_limit = min(
                                line_limit, p.resource_stock.get(res, 0.0) / req_unit
                            )
                        elif p.resource_stock.get(res, 0.0) < 0:
                            line_limit = 0.0  # Negativer Stock = Problem
                    max_possible_prod[output_good] += line_limit
            else:
                # Fallback: nutze calculate_inputs_for_output direkt pro Gut
                for good in getattr(p, "can_produce", set()):
                    inputs = p.calculate_inputs_for_output(1.0)
                    limit = float('inf')
                    for res, req in inputs.items():
                        if req > 1e-9:
                            limit = min(limit, p.resource_stock.get(res, 0.0) / req)
                        elif p.resource_stock.get(res, 0.0) < 0:
                            limit = 0.0
                    max_possible_prod[good] = limit

            # Setze finale Ziele
            initial_targets = getattr(p, "production_target", {})
            final_targets = {}
            labor_limit = float('inf')
            if hasattr(p, 'assigned_labor_hours'):
                total_labor_needed_for_target = 0
                for good, target in initial_targets.items():
                    line = next((l for l in p.production_lines if l.output_good == good), None)
                    if line:
                        total_labor_needed_for_target += (target / line.effective_efficiency) * line.labor_requirement
                if total_labor_needed_for_target > 1e-6:
                    labor_fulfillment_ratio = p.assigned_labor_hours / total_labor_needed_for_target
                    labor_limit = labor_fulfillment_ratio
            for good in getattr(p, "can_produce", set()):
                initial_target = initial_targets.get(good, 0.0)
                resource_limit = max_possible_prod.get(good, 0.0)
                final_target = min(initial_target, resource_limit, initial_target * labor_limit)
                penalty_factor = self.producer_penalties.get(pid, {}).get(
                    "production_target", 1.0
                )
                final_targets[good] = max(0.0, final_target * penalty_factor)
            # Update producer's target
            if hasattr(p, "production_target"):
                p.production_target = defaultdict(float, final_targets)

    def _update_regional_metrics(self) -> None:
        """Aktualisiert Metriken wie Stresslevel und Produktionserfüllung."""
        # --- Logik von RM._update_regional_metrics hier einfügen ---
        self._update_stress_level()
        self.production_fulfillment.clear()
        for good, initial_regional_target in self.production_targets.items():
            if initial_regional_target > 1e-6:
                total_planned = sum(
                    getattr(p, "production_target", {}).get(good, 0.0)
                    for p in self.producers.values()
                )
                fulfillment = total_planned / initial_regional_target
                self.production_fulfillment[good] = fulfillment
                self.production_history[good].append(fulfillment)

    def _update_stress_level(self) -> None:
        """Berechnet das aktuelle Stress-Level der Region."""
        # --- Logik von RM._update_stress_level hier einfügen ---
        total_sev, num_scarce, max_sev = 0.0, 0, 0.0
        for resource, history in self.shortage_history.items():
            if history and history[-1] > 0.05:
                severity = history[-1] ** 1.5
                weight = self.parent.production_priorities.get(resource, 1.0)
                total_sev += severity * weight
                max_sev = max(max_sev, severity * weight)
                num_scarce += 1
        if num_scarce > 0:
            self.stress_level = np.clip(
                0.6 * (total_sev / num_scarce) + 0.4 * max_sev, 0.0, 1.0
            )
        else:
            self.stress_level = 0.0

    # --- Methoden für Datenaustausch und Interaktion (wie im Original) ---
    # Diese werden von S3M oder den Komponenten aufgerufen

    def collect_regional_resources(self) -> Dict[str, float]:
        """Gibt die aktuell im regionalen Pool befindlichen Ressourcen zurück."""
        return dict(self.resource_pool)

    def calculate_resource_needs(self) -> Dict[str, float]:
        """Gibt den berechneten Ressourcenbedarf der Region zurück."""
        if not self.resource_needs:
            self._calculate_resource_requirements()
        return dict(self.resource_needs)

    def calculate_resource_surpluses(self) -> Dict[str, float]:
        """Berechnet potenzielle Überschüsse für den inter-regionalen Ausgleich."""
        surpluses = defaultdict(float)
        needs_calc = (
            self.calculate_resource_needs()
        )  # Stelle sicher, dass Bedarf aktuell ist
        for resource, available in self.resource_pool.items():
            needed = needs_calc.get(resource, 0.0)
            surplus = available - needed
            if surplus > 1e-3:
                surpluses[resource] = surplus
        return dict(surpluses)

    def calculate_regional_priority(self) -> float:
        """Berechnet die Gesamtpriorität der Region."""
        # --- Logik von RM.calculate_regional_priority hier einfügen ---
        priority_sum, target_sum = 0.0, 0.0
        for good, target in self.production_targets.items():
            if target > 0:
                prio = self.good_priorities.get(good, 1.0)
                priority_sum += prio * target
                target_sum += target
        base_prio = (priority_sum / target_sum) if target_sum > 0 else 1.0
        stress_factor = 1.0 + self.stress_level * 0.5
        critical_short_factor = 1.0
        for good in self.parent.critical_goods:
            if self.production_fulfillment.get(good, 1.0) < 0.8:
                critical_short_factor = max(critical_short_factor, 1.3)
        final_prio = base_prio * stress_factor * critical_short_factor
        return np.clip(final_prio, 0.5, 3.0)

    def allocate_resource(self, resource: str, amount: float) -> None:
        """Weist der Region eine Menge einer Ressource zu (von S3 global)."""
        if amount >= 0:
            self.resource_pool[resource] = (
                amount  # Überschreibt aktuellen Pool? Oder addiert? Hier: überschreibt
            )
            self.logger.debug(
                f"Region {self.region_id}: Ressource {resource} aktualisiert auf {amount:.2f}"
            )
        else:
            self.logger.warning(
                f"Region {self.region_id}: Ungültige Allokation für {resource}: {amount}"
            )

    def get_resource_amount(self, resource: str) -> float:
        """Gibt die Menge einer Ressource im regionalen Pool zurück."""
        return self.resource_pool.get(resource, 0.0)

    def receive_resource(self, resource: str, amount: float) -> None:
        """Empfängt Ressourcen (z.B. von anderer Region via S3)."""
        if amount > 0:
            self.resource_pool[resource] += amount
            self.logger.info(
                f"Region {self.region_id}: {amount:.2f} {resource} erhalten."
            )
        else:
            self.logger.warning(
                f"Region {self.region_id}: Ungültige Ressourcenmenge empfangen für {resource}: {amount}"
            )

    def transfer_resource(
        self, resource: str, amount: float, target_region_id: Union[int, str]
    ) -> bool:
        """Versucht, Ressourcen an eine andere Region zu transferieren (wird von S3M aufgerufen)."""
        if amount <= 0:
            return False
        available = self.resource_pool.get(resource, 0.0)

        if available >= amount - 1e-6:  # Toleranz für Fließkomma
            self.resource_pool[resource] -= amount
            self.logger.info(
                f"Region {self.region_id}: Transfer von {amount:.2f} {resource} an Region {target_region_id} genehmigt."
            )
            return True
        else:
            self.logger.warning(
                f"Region {self.region_id}: Transfer von {amount:.2f} {resource} an Region {target_region_id} fehlgeschlagen (nur {available:.2f} verfügbar)."
            )
            return False

    # --- Methoden für Feedback und Metriken (wie im Original) ---

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Gibt Performance-Metriken für die Feedback-Analyse durch S3 zurück."""
        return {
            "resource_utilization": dict(self.resource_utilization),
            "production_fulfillment": dict(self.production_fulfillment),
            "stress_level": self.stress_level,
            "producer_count": len(self.producers),
        }

    def get_summary_status(self) -> Dict[str, Any]:
        """Gibt einen zusammenfassenden Status für den S3->S4 Bericht zurück."""
        return {
            "resource_utilization": {
                r: round(u, 3) for r, u in self.resource_utilization.items()
            },
            "production_fulfillment": {
                g: round(f, 3) for g, f in self.production_fulfillment.items()
            },
            "stress_level": round(self.stress_level, 3),
            # Füge hier weitere für S4 relevante Infos hinzu
            "total_capacity": sum(
                getattr(p, "productive_capacity", 0) for p in self.producers.values()
            ),
            "avg_tech_level": (
                np.mean(
                    [getattr(p, "tech_level", 1.0) for p in self.producers.values()]
                )
                if self.producers
                else 1.0
            ),
        }

    def generate_feedback(self) -> Dict[str, Any]:
        """Generiert spezifisches Feedback für das übergeordnete System 3."""
        # --- Logik von RM.generate_feedback hier einfügen ---
        feedback = {
            "resource_shortages": {},
            "production_deviations": {},
            "stress_level": self.stress_level,
            "producer_satisfaction": self._calculate_average_producer_satisfaction(),
        }
        for resource, history in self.shortage_history.items():
            if history and history[-1] > 0.05:
                feedback["resource_shortages"][resource] = history[-1]
        for good, fulfillment in self.production_fulfillment.items():
            deviation = fulfillment - 1.0
            if abs(deviation) > 0.05:
                feedback["production_deviations"][good] = deviation
        return feedback

    def _calculate_average_producer_satisfaction(self) -> float:
        """Berechnet die durchschnittliche Zufriedenheit der Producer."""
        # --- Logik von RM._calculate_average_producer_satisfaction hier einfügen ---
        scores = [
            np.clip(getattr(p, "capacity_utilization", 0.5) * 1.1, 0.0, 1.0)
            for p in self.producers.values()
        ]  # Beispiel-Logik
        return np.mean(scores) if scores else 0.5

    def collect_production_statistics(self) -> Dict[str, Any]:
        """Sammelt Produktionsstatistiken von allen Producern."""
        # --- Logik von RM.collect_production_statistics hier einfügen ---
        stats: Dict[str, Any] = {
            "output_by_good": defaultdict(float),
            "total_regional_output": 0.0,
            "capacity_utilization": 0.0,
            "producer_count": len(self.producers),
        }
        total_cap, total_out = 0.0, 0.0
        for p in self.producers.values():
            total_out += getattr(p, "total_output_this_step", 0.0)
            total_cap += getattr(p, "productive_capacity", 0.0)
            prod_output = getattr(
                p, "market_production", getattr(p, "output_stock", {})
            )
            for good, amount in prod_output.items():
                stats["output_by_good"][good] += amount
        stats["total_regional_output"] = total_out
        if total_cap > 0:
            stats["capacity_utilization"] = total_out / total_cap
        stats["output_by_good"] = dict(stats["output_by_good"])
        return stats

    def gather_local_data(self) -> Dict[str, Any]:
        """Aggregiert wichtige regionale Kennzahlen."""
        stats = self.collect_production_statistics()
        summary = self.get_summary_status()
        feedback = self.generate_feedback()
        data = {
            "total_output": stats["total_regional_output"],
            "capacity_utilization": stats["capacity_utilization"],
            "producer_count": stats["producer_count"],
            "resource_shortfall": feedback.get("resource_shortages", {}),
            "production_fulfillment": summary.get("production_fulfillment", {}),
            "development_index": summary.get("avg_tech_level", 1.0),
        }
        return data

    def calculate_production_fulfillment(self) -> Dict[str, float]:
        """Berechnet die Produktionserfüllung für diese Region (wird von S3M genutzt)."""
        # Gibt die intern gespeicherten Werte zurück
        return dict(self.production_fulfillment)

    # --- Audit-Funktionen (wird von S3 AuditController aufgerufen) ---

    def audit_producer(self, producer_id: str) -> Optional[Dict[str, Any]]:
        """Führt ein Audit für einen spezifischen Producer durch."""
        # --- Logik von RM.audit_producer hier einfügen ---
        if producer_id not in self.producers:
            self.logger.error(
                f"Audit fehlgeschlagen: Producer {producer_id} nicht in Region {self.region_id}."
            )
            return None
        producer = self.producers[producer_id]
        self.logger.info(
            f"Führe Audit für Producer {producer_id} in Region {self.region_id} durch..."
        )
        audit_result: Dict[str, Any] = {
            "timestamp": getattr(self.parent.model, "current_step", -1),
            "resource_waste": 0.0,
            "capacity_underutilization": 0.0,
            "tech_level": getattr(producer, "tech_level", 1.0),
            "maintenance_status": getattr(producer, "maintenance_status", 1.0),
            "findings": [],
        }
        # ... (Logik für Waste, Underutilization, Maintenance Checks wie im Original RM) ...
        resource_needs_last = getattr(
            producer, "_last_resource_needs", producer.calculate_resource_needs()
        )  # Hole letzten Bedarf
        for resource, available in getattr(producer, "resource_stock", {}).items():
            needed = resource_needs_last.get(resource, 0.0)
            unused = max(0.0, available - needed)
            if available > 1e-6:
                waste_ratio = unused / available
                audit_result["resource_waste"] = max(
                    audit_result["resource_waste"], waste_ratio
                )
        actual_util = getattr(producer, "capacity_utilization", 0.0)
        audit_result["capacity_underutilization"] = max(0.0, 1.0 - actual_util)
        self.logger.info(f"Audit für Producer {producer_id} abgeschlossen.")
        return audit_result

    def apply_producer_penalties(
        self, producer_id: str, penalties: List[Dict[str, Any]]
    ) -> None:
        """Wendet Strafen auf einen Producer an."""
        # --- Logik von RM.apply_producer_penalties hier einfügen ---
        if producer_id not in self.producers:
            self.logger.error(
                f"Kann Penalty nicht anwenden: Producer {producer_id} nicht gefunden."
            )
            return
        if producer_id not in self.producer_penalties:
            self.producer_penalties[producer_id] = {}
        for penalty in penalties:
            p_type, p_factor = penalty.get("type"), penalty.get("factor")
            if p_type and p_factor is not None:
                current_factor = self.producer_penalties[producer_id].get(p_type, 1.0)
                new_factor = np.clip(current_factor * p_factor, 0.1, 1.0)
                self.producer_penalties[producer_id][p_type] = new_factor
                self.logger.warning(
                    f"Region {self.region_id}: Penalty '{p_type}' für Producer {producer_id} auf {new_factor:.2f} gesetzt."
                )
        # TODO: Mechanismus zum Auslaufen der Strafen implementieren


# -------------------------------------------------------------------------
# DynamicRegionalSystem3Manager: Inherits from RegionalSystem3Manager
# -------------------------------------------------------------------------


class DynamicRegionalSystem3Manager(RegionalSystem3Manager):
    """
    A dynamic version of RegionalSystem3Manager that assigns producer targets
    based on producer skills and current task queue length.
    """

    def __init__(self, parent_system3: "System3Manager", region_id: Union[int, str]):
        """
        Initialisiert den DynamicRegionalSystem3Manager.
        """
        super().__init__(parent_system3, region_id)
        self.logger.info(
            f"DynamicRegionalSystem3Manager für Region {self.region_id} initialisiert und verwendet."
        )

    def _assign_producer_targets(self) -> None:
        """
        Überträgt die finalen, realisierbaren Produktionsziele an die Producer,
        basierend auf Skills und Task-Queue-Länge.
        """
        self.logger.debug(
            f"DynamicRegion {self.region_id}: Weise finale Produktionsziele dynamisch zu..."
        )

        final_producer_targets: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        for good, regional_target_amount in self.production_targets.items():
            if regional_target_amount <= 0:
                continue

            self.logger.debug(
                f"DynamicRegion {self.region_id}: Bearbeite regionales Ziel für Gut '{good}' Menge: {regional_target_amount:.2f}"
            )

            candidates = []
            skill_needed = f"produce_{good}"  # Example specific skill

            for producer_id, producer in self.producers.items():
                if producer is None:  # Should not happen if registration is correct
                    self.logger.warning(
                        f"Producer {producer_id} in Region {self.region_id} ist None. Überspringe."
                    )
                    continue

                # Check if producer can produce the good (basic skill check)
                # producer.skills is a Dict[str, int]
                # producer.can_produce is a Set[str]
                if good in getattr(producer, "can_produce", set()):
                    # More specific skill check (optional, could be combined with can_produce)
                    # if getattr(producer, 'skills', {}).get(skill_needed, 0) > 0:
                    candidates.append((producer, producer_id))

            if not candidates:
                self.logger.warning(
                    f"DynamicRegion {self.region_id}: Keine fähigen Producer für Gut '{good}' gefunden. Regionales Ziel {regional_target_amount:.2f} kann nicht zugewiesen werden."
                )
                continue

            # Sort candidates: Lower score is better (less backlog relative to skill)
            # producer.skills.get(skill_needed, 1) -> skill level, default 1 if not found
            # producer.task_queue_length
            sorted_candidates = sorted(
                candidates,
                key=lambda p_info: (
                    getattr(p_info[0], "task_queue_length", 0)
                    / getattr(p_info[0], "skills", {}).get(skill_needed, 1.0)
                    if getattr(p_info[0], "skills", {}).get(skill_needed, 1.0) > 0
                    else float("inf")
                ),
            )

            self.logger.debug(
                f"DynamicRegion {self.region_id}: {len(sorted_candidates)} Kandidaten für '{good}' sortiert. Top: {getattr(sorted_candidates[0][0], 'unique_id', 'N/A') if sorted_candidates else 'N/A'}"
            )

            remaining_regional_target = regional_target_amount
            for producer, producer_id in sorted_candidates:
                if remaining_regional_target <= 0:
                    break

                assignable_capacity_for_good = 0.0
                if hasattr(producer, "production_lines"):
                    for line in producer.production_lines:
                        if getattr(line, "output_good", None) == good and getattr(
                            line, "is_active", True
                        ):
                            assignable_capacity_for_good += getattr(
                                line, "capacity_share", 0.0
                            )

                if assignable_capacity_for_good <= 0:
                    self.logger.debug(
                        f"  Producer {producer_id} hat keine Kapazität für '{good}'."
                    )
                    continue

                # Amount to assign cannot exceed producer's capacity for this good or remaining target
                # Also, consider already assigned targets for this good to this producer in this loop (though defaultdict handles this)
                amount_to_assign = min(
                    remaining_regional_target,
                    assignable_capacity_for_good
                    - final_producer_targets[producer_id][good],
                )

                # Ensure producer's overall capacity isn't implicitly exceeded if sum of line.capacity_share > productive_capacity
                # This check should ideally be part of assignable_capacity_for_good calculation or ProducerAgent's internal logic
                # For now, assume capacity_share respects overall productive_capacity.

                if amount_to_assign > 1e-6:  # Use a small epsilon for float comparison
                    self.logger.debug(
                        f"  Weise {amount_to_assign:.2f} von '{good}' an Producer {producer_id} zu (Kapazität: {assignable_capacity_for_good:.2f})."
                    )
                    final_producer_targets[producer_id][good] += amount_to_assign
                    remaining_regional_target -= amount_to_assign

                    # Update producer's task queue length
                    if hasattr(producer, "task_queue_length"):
                        producer.task_queue_length += int(
                            round(amount_to_assign)
                        )  # Assuming task_queue_length is int
                    else:
                        self.logger.warning(
                            f"Producer {producer_id} hat kein Attribut 'task_queue_length'."
                        )

            if remaining_regional_target > 1e-6:  # Use a small epsilon
                self.logger.warning(
                    f"DynamicRegion {self.region_id}: Konnte {remaining_regional_target:.2f} des regionalen Ziels für Gut '{good}' nicht zuweisen (Kapazität der Producer erschöpft oder keine weiteren Kandidaten)."
                )

        # Update actual production_target for all producers in this region
        for producer_id, producer_agent in self.producers.items():
            if producer_agent is None:
                continue

            if producer_id in final_producer_targets:
                # producer_agent.production_target should be a DefaultDict[str, float]
                # Ensure it is, or re-initialize if necessary
                if not isinstance(
                    getattr(producer_agent, "production_target", None), defaultdict
                ):
                    producer_agent.production_target = defaultdict(
                        float
                    )  # Initialize if not a defaultdict

                # Clear old targets for this producer for goods handled by this RM before updating
                # This ensures that if a good was previously targeted but is no longer, it's removed.
                # However, final_producer_targets only contains goods that *should* be targeted.
                # So, simply assigning is fine.

                # Create a new defaultdict for assignment to avoid modifying the one being iterated if it's the same object
                new_targets_for_producer = defaultdict(
                    float, final_producer_targets[producer_id]
                )
                producer_agent.production_target = new_targets_for_producer

                # Log the assigned targets to the producer
                assigned_goods_log = {
                    g: round(t, 2) for g, t in new_targets_for_producer.items()
                }
                self.logger.debug(
                    f"DynamicRegion {self.region_id}: Finale Ziele für Producer {producer_id} gesetzt: {assigned_goods_log}"
                )

            else:
                # This producer was not assigned any targets in this cycle for any good. Clear their targets.
                if hasattr(producer_agent, "production_target"):
                    producer_agent.production_target.clear()  # Clear existing targets
                else:
                    producer_agent.production_target = defaultdict(
                        float
                    )  # Ensure it exists and is empty
                self.logger.debug(
                    f"DynamicRegion {self.region_id}: Keine Ziele für Producer {producer_id} in diesem Zyklus. Ziele geleert."
                )

        self.logger.info(
            f"DynamicRegion {self.region_id}: Dynamische Zuweisung der Producer-Ziele abgeschlossen."
        )


# -------------------------------------------------------------------------
# DynamicSystem3Manager: Inherits from System3Manager
# -------------------------------------------------------------------------


class DynamicSystem3Manager(System3Manager):
    """
    A dynamic version of System3Manager that might incorporate more adaptive
    or learning-based mechanisms in the future. For now, it adds specific
    data gathering capabilities.
    """

    def __init__(self, model: "EconomicModel") -> None:
        """
        Initialisiert den DynamicSystem3Manager.

        Args:
            model: Referenz auf das Hauptmodell (EconomicModel).
        """
        super().__init__(model)
        self.logger.info(
            f"DynamicSystem3Manager initialisiert (erbt von System3Manager). S3 Adaptiv: {self.enable_adaptive_coordination}"
        )
        # Die spezialisierten Komponenten (InterRegionalResourceBalancer, etc.)
        # werden bereits in System3Manager.__init__ korrekt initialisiert und
        # verwenden die 'self' Instanz, die hier die DynamicSystem3Manager Instanz ist.

    def get_all_producers_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Collects specified data from all producers across all regional managers.

        Returns:
            A dictionary where keys are producer unique_ids and values are
            dictionaries containing 'skills', 'task_queue_length', 'region_id', and 'capacity_utilization'.
        """
        all_producers_data: Dict[str, Dict[str, Any]] = {}
        self.logger.debug(
            "DynamicS3M: Sammle Daten von allen Producern (inkl. Kapazitätsauslastung)..."
        )

        for region_id, regional_manager in self.regional_managers.items():
            # Ensure regional_manager has a 'producers' attribute and it's iterable
            if not hasattr(regional_manager, "producers") or not isinstance(
                regional_manager.producers, dict
            ):
                self.logger.warning(
                    f"RegionalManager für Region {region_id} hat kein 'producers' Dictionary. Überspringe."
                )
                continue

            for producer_id, producer_agent in regional_manager.producers.items():
                if producer_agent is None:
                    self.logger.warning(
                        f"Producer {producer_id} in Region {region_id} ist None. Überspringe."
                    )
                    continue

                # Safely access attributes using getattr
                skills = getattr(producer_agent, "skills", {})
                task_queue_length = getattr(producer_agent, "task_queue_length", 0)
                capacity_utilization = getattr(
                    producer_agent, "capacity_utilization", 0.0
                )

                # Ensure producer_id from the agent matches the key, for consistency
                agent_unique_id = getattr(producer_agent, "unique_id", producer_id)

                all_producers_data[agent_unique_id] = {
                    "skills": skills,
                    "task_queue_length": task_queue_length,
                    "region_id": regional_manager.region_id,
                    "capacity_utilization": capacity_utilization,
                }
                self.logger.debug(
                    f"  Daten für Producer {agent_unique_id} (Region {regional_manager.region_id}): "
                    f"Skills: {len(skills)} Typen, Queue: {task_queue_length}, Util: {capacity_utilization:.2f}"
                )

        if not all_producers_data:
            self.logger.info(
                "DynamicS3M: Keine Producer-Daten gesammelt (keine Regionen oder Producer gefunden)."
            )
        else:
            self.logger.info(
                f"DynamicS3M: Daten für {len(all_producers_data)} Producer gesammelt (inkl. Kapazitätsauslastung)."
            )

        return all_producers_data

    def evaluate_and_broadcast_governance_mode(self) -> None:
        """
        Evaluates KPIs based on producer and regional data, determines a
        governance mode (GREEN, YELLOW, RED), and broadcasts it.
        """
        self.logger.info("DynamicS3M: Starte Evaluierung des Governance Modus...")

        # Ensure necessary imports are available (ideally at top of file)
        # from .communication_contracts import GovernanceMode, SystemFeedback
        # import numpy as np

        all_producers_data = self.get_all_producers_data()

        # 1. Calculate Production Utilization KPI
        kpi_production_utilization = (
            np.mean(
                [
                    data.get("capacity_utilization", 0.0)
                    for data in all_producers_data.values()
                ]
            )
            if all_producers_data
            else 1.0
        )
        if not all_producers_data:
            self.logger.debug(
                "  Keine Producer-Daten für Kapazitätsauslastung, KPI default auf 1.0 (Grün)."
            )

        # 2. Calculate Inventory Variance KPI (Proxy via production_fulfillment)
        all_fulfillment_variances = []
        for rm_id, rm in self.regional_managers.items():
            if hasattr(rm, "production_fulfillment") and isinstance(
                rm.production_fulfillment, dict
            ):
                if not rm.production_fulfillment:
                    self.logger.debug(
                        f"  RegionalManager {rm_id} hat leeres 'production_fulfillment' Dict."
                    )
                for good, ratio in rm.production_fulfillment.items():
                    all_fulfillment_variances.append(
                        1.0 - ratio
                    )  # Variance from target (1.0 = on target)
            else:
                self.logger.warning(
                    f"  RegionalManager {rm_id} hat kein 'production_fulfillment' Dict oder es ist kein Dictionary."
                )

        kpi_inventory_variance = (
            np.mean(all_fulfillment_variances) if all_fulfillment_variances else 0.0
        )
        if not all_fulfillment_variances:
            self.logger.debug(
                "  Keine Planerfüllungsdaten für Inventarvarianz, KPI default auf 0.0 (Grün)."
            )

        # 3. Calculate On-Time Delivery Rate KPI (Proxy via coordination_effectiveness)
        kpi_on_time_delivery_rate = getattr(self, "coordination_effectiveness", 1.0)
        if not hasattr(self, "coordination_effectiveness"):
            self.logger.warning(
                "  Attribut 'coordination_effectiveness' nicht im S3M gefunden, KPI default auf 1.0 (Grün)."
            )

        self.logger.info(
            f"DynamicS3M KPIs: ProdUtil={kpi_production_utilization:.3f}, InvVar={kpi_inventory_variance:.3f}, DeliveryRate={kpi_on_time_delivery_rate:.3f}"
        )

        # 4. Determine Governance Mode
        # Assuming GovernanceMode is imported, e.g., from .communication_contracts import GovernanceMode
        thresholds = {
            "production_utilization": {
                "yellow_ge": 0.60,
                "red_lt": 0.60,
                "green_ge": 0.80,
            },
            "inventory_variance": {"yellow_le": 0.15, "red_gt": 0.15, "green_le": 0.05},
            "on_time_delivery_rate": {
                "yellow_ge": 0.85,
                "red_lt": 0.85,
                "green_ge": 0.95,
            },
        }

        current_mode = GovernanceMode.GREEN  # Requires GovernanceMode to be defined

        # Evaluate Production Utilization
        if kpi_production_utilization < thresholds["production_utilization"]["red_lt"]:
            current_mode = GovernanceMode.RED
            self.logger.debug(
                f"  Governance Modus auf ROT wegen ProdUtil: {kpi_production_utilization:.3f} < {thresholds['production_utilization']['red_lt']}"
            )
        elif (
            kpi_production_utilization
            < thresholds["production_utilization"]["green_ge"]
        ):  # Check for Yellow only if not Red
            current_mode = GovernanceMode.YELLOW
            self.logger.debug(
                f"  Governance Modus auf GELB wegen ProdUtil: {kpi_production_utilization:.3f} < {thresholds['production_utilization']['green_ge']}"
            )

        # Evaluate Inventory Variance (can escalate to RED or turn GREEN to YELLOW)
        if kpi_inventory_variance > thresholds["inventory_variance"]["red_gt"]:
            current_mode = GovernanceMode.RED
            self.logger.debug(
                f"  Governance Modus auf ROT (oder bleibt ROT) wegen InvVar: {kpi_inventory_variance:.3f} > {thresholds['inventory_variance']['red_gt']}"
            )
        elif kpi_inventory_variance > thresholds["inventory_variance"]["green_le"]:
            if current_mode == GovernanceMode.GREEN:  # Only escalate Green to Yellow
                current_mode = GovernanceMode.YELLOW
                self.logger.debug(
                    f"  Governance Modus von GRÜN auf GELB wegen InvVar: {kpi_inventory_variance:.3f} > {thresholds['inventory_variance']['green_le']}"
                )

        # Evaluate On-Time Delivery Rate (can escalate to RED or turn GREEN to YELLOW)
        if kpi_on_time_delivery_rate < thresholds["on_time_delivery_rate"]["red_lt"]:
            current_mode = GovernanceMode.RED
            self.logger.debug(
                f"  Governance Modus auf ROT (oder bleibt ROT) wegen DeliveryRate: {kpi_on_time_delivery_rate:.3f} < {thresholds['on_time_delivery_rate']['red_lt']}"
            )
        elif (
            kpi_on_time_delivery_rate < thresholds["on_time_delivery_rate"]["green_ge"]
        ):
            if current_mode == GovernanceMode.GREEN:  # Only escalate Green to Yellow
                current_mode = GovernanceMode.YELLOW
                self.logger.debug(
                    f"  Governance Modus von GRÜN auf GELB wegen DeliveryRate: {kpi_on_time_delivery_rate:.3f} < {thresholds['on_time_delivery_rate']['green_ge']}"
                )

        self.logger.info(
            f"DynamicS3M: Finaler Governance Modus bestimmt: {current_mode.value}"
        )

        # 5. Broadcast Mode
        # Assuming SystemFeedback is imported, e.g., from .communication_contracts import SystemFeedback
        feedback_payload = {
            "type": "governance_mode_update",
            "current_mode": current_mode.value,
            "kpis": {
                "production_utilization": round(kpi_production_utilization, 4),
                "inventory_variance": round(kpi_inventory_variance, 4),
                "on_time_delivery_rate": round(kpi_on_time_delivery_rate, 4),
            },
        }

        governance_msg = SystemFeedback(
            step=getattr(self.model, "current_step", -1),
            source_system="System3",
            target_system="Broadcast",
            feedback_type="governance_mode_status",
            payload=feedback_payload,
        )

        if hasattr(self.model, "route_directive"):
            self.model.route_directive(governance_msg)
            self.logger.info(
                f"DynamicS3M: Governance Modus '{current_mode.value}' und KPIs an Modell weitergeleitet (via route_directive)."
            )
        elif hasattr(self.model, "broadcast_message"):
            self.model.broadcast_message(governance_msg)
            self.logger.info(
                f"DynamicS3M: Governance Modus '{current_mode.value}' und KPIs an Modell gesendet (via broadcast_message)."
            )
        else:
            if hasattr(self.model, "global_governance_status"):
                self.model.global_governance_status = feedback_payload
                self.logger.info(
                    f"DynamicS3M: Governance Modus '{current_mode.value}' auf model.global_governance_status gespeichert."
                )
            else:
                self.logger.warning(
                    "DynamicS3M: Konnte Governance Modus nicht broadcasten (keine passende Methode oder Attribut im Modell gefunden)."
                )
