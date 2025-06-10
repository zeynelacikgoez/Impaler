# Impaler/agents/producer.py
# =====================================================================
# Intelligente, realitätsnahe Implementierung eines Produktionsagenten
# für planwirtschaftliche Simulationen (überarbeitet)
# =====================================================================

import random
import numpy as np
import math
import logging
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable, Deque, DefaultDict, TYPE_CHECKING
from dataclasses import dataclass, field

# Typ-Prüfung Imports
if TYPE_CHECKING:
    from ..core.model import EconomicModel

# Optional: SciPy für fortgeschrittene Optimierung
try:
    from scipy.optimize import minimize, LinearConstraint, OptimizeResult
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Optional: Reinforcement Learning (mit verbessertem Fallback)
try:
    # Annahme: Pfad stimmt oder ist im PYTHONPATH
    from ..ml.q_learning import QLearningAgent, DeepQLearningAgent, TORCH_AVAILABLE
    RL_AVAILABLE = True  # Grundlegende RL-Struktur ist da
except ImportError:
    TORCH_AVAILABLE = False # Sicherstellen, dass es definiert ist
    # Verbesserter Fallback-Agent (Dummy für tabulares QL)
    class QLearningAgent: # type: ignore
        """Dummy QLearningAgent, wenn die ML-Bibliothek fehlt. Implementiert eine einfache Regel."""
        def __init__(self, state_dim: int, action_dim: int, **kwargs):
            self.state_dim = state_dim
            self.action_dim = action_dim
            logging.getLogger(__name__).warning(
                "QLearningAgent-Bibliothek nicht gefunden. Verwende einfachen regelbasierten Fallback für RL-Agenten."
            )

        def choose_action(self, state: np.ndarray) -> int:
            """Einfache Regel: Priorisiere Wartung bei schlechtem Zustand, sonst Technologie."""
            maintenance_idx = 2
            efficiency_idx = 1
            if len(state) > maintenance_idx and state[maintenance_idx] < 0.5: return 2
            elif len(state) > efficiency_idx and state[efficiency_idx] < 0.8: return 1
            else: return 0

        def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool = False) -> None:
            pass

    # Dummy für DeepQLearningAgent, falls Torch fehlt, aber q_learning.py vorhanden ist
    if not TORCH_AVAILABLE:
        class DeepQLearningAgent: # type: ignore
            """Dummy DeepQLearningAgent, wenn PyTorch fehlt."""
            def __init__(self, state_dim: int, action_dim: int, **kwargs):
                self.state_dim = state_dim
                self.action_dim = action_dim
                self.device = "cpu"
                logging.getLogger(__name__).warning(
                    "PyTorch nicht gefunden. DeepQLearningAgent ist nicht verfügbar. Verwende Dummy."
                )
            def choose_action(self, state: np.ndarray, explore: bool = True) -> int: return random.randrange(self.action_dim)
            def store_transition(self, s, a, r, ns, d) -> None: pass
            def learn(self) -> Optional[float]: return None
            def decay_exploration(self) -> None: pass

    RL_AVAILABLE = False # Wenn der initiale Import fehlschlägt, ist RL generell nicht verfügbar

from ..core.config import InnovationEffect


# ---------------------------------------------------------------------
# Hilfsklassen (ausgelagert oder hier belassen, hier für Vollständigkeit)
# Mit Type Hints und Docstrings versehen
# ---------------------------------------------------------------------
@dataclass
class ResourceRequirement:
    """
    Repräsentiert die Anforderung einer Ressource für einen Produktionsprozess.

    Attributes:
        resource_type (str): Name der benötigten Ressource/des Inputs.
        amount_per_unit (float): Benötigte Menge pro produzierter Output-Einheit.
        is_essential (bool): Ist diese Ressource zwingend erforderlich?
        substitutes (List[Tuple[str, float]]): Liste möglicher Substitute
            als Tupel (Ersatzressource, Effizienzfaktor < 1.0, wie viel Ersatz pro Einheit Original).
    """
    resource_type: str
    amount_per_unit: float = field(default=0.0)
    is_essential: bool = True
    substitutes: List[Tuple[str, float]] = field(default_factory=list)

@dataclass
class EmissionProfile:
    """Repräsentiert das Emissionsprofil eines Produktionsprozesses."""
    emissions: Dict[str, float] = field(default_factory=dict) # {Emissionstyp: Menge pro Output-Einheit}

    def get_emission(self, emission_type: str) -> float:
        """Gibt die Emission eines bestimmten Typs zurück."""
        return self.emissions.get(emission_type, 0.0)

    def scale(self, factor: float) -> 'EmissionProfile':
        """Skaliert alle Emissionen um einen Faktor."""
        if factor < 0: return EmissionProfile() # Keine negativen Emissionen durch Skalierung
        return EmissionProfile(emissions={k: v * factor for k, v in self.emissions.items()})

# --- Produktionslinien-Klasse (überarbeitet) ---
class ProductionLine:
    """
    Repräsentiert eine Produktionslinie für ein spezifisches Gut.

    Verwaltet Input-Anforderungen, Emissionen, Effizienz und Kapazitätszuweisung.

    Attributes:
        name (str): Einzigartiger Name der Linie.
        output_good (str): Das produzierte Gut.
        input_requirements (List[ResourceRequirement]): Liste der benötigten Inputs.
        emission_profile (EmissionProfile): Emissionsdaten für diese Linie.
        base_efficiency (float): Die grundlegende technologische Effizienz der Linie (0-1+).
        effective_efficiency (float): Die tatsächliche Effizienz (base_efficiency * agent.tech_level).
        min_utilization (float): Mindestauslastung für den Betrieb (0-1).
        max_capacity (Optional[float]): Maximale Output-Kapazität dieser spezifischen Linie.
        priority (float): Priorität bei der internen Kapazitäts- und Ressourcenzuweisung.
        setup_cost (float): Ressourcenkosten für die Inbetriebnahme (noch nicht verwendet).
        is_active (bool): Ob die Linie derzeit betrieben wird/werden kann.
        capacity_share (float): Zugewiesener Anteil an der Gesamtkapazität des Agenten.
        target_output (float): Vom Planungssystem vorgegebenes Ziel für diese Linie.
        planned_output (float): Geplanter Output nach Ressourcenprüfung.
        actual_output (float): Tatsächlich produzierter Output im letzten Schritt.
        utilization (float): Tatsächliche Auslastung (actual_output / (capacity_share * tech_level)).
        output_history (Deque[Tuple[int, float]]): Historie der Produktion.
        bottleneck_info (Optional[Dict[str, Any]]): Informationen über den letzten Engpass.
        planned_consumed_inputs (Dict[str, float]): Detaillierte Liste der Ressourcen, die für
                                                    den geplanten Output verbraucht werden sollen
                                                    (inklusive Substitutionen).
    """
    name: str
    output_good: str
    input_requirements: List[ResourceRequirement]
    emission_profile: EmissionProfile
    base_efficiency: float
    effective_efficiency: float # Wird vom Agenten gesetzt
    min_utilization: float
    max_capacity: Optional[float]
    priority: float
    setup_cost: float
    is_active: bool
    capacity_share: float
    target_output: float
    planned_output: float
    actual_output: float
    utilization: float
    output_history: Deque[Tuple[int, float]]
    bottleneck_info: Optional[Dict[str, Any]]
    planned_consumed_inputs: Dict[str, float]

    def __init__(
        self,
        name: str,
        output_good: str,
        input_requirements: List[ResourceRequirement],
        emission_profile: Optional[EmissionProfile] = None,
        base_efficiency: float = 1.0, # Umbenannt von efficiency
        min_utilization: float = 0.1,
        max_capacity: Optional[float] = None,
        priority: float = 1.0,
        setup_cost: float = 0.0,
        is_active: bool = True
    ):
        self.name = name
        self.output_good = output_good
        self.input_requirements = input_requirements
        self.emission_profile = emission_profile or EmissionProfile()
        self.base_efficiency = max(0.1, base_efficiency) # Mindesteffizienz
        self.effective_efficiency = self.base_efficiency # Wird später vom Agenten aktualisiert
        self.min_utilization = np.clip(min_utilization, 0.0, 1.0)
        self.max_capacity = max_capacity if max_capacity is not None and max_capacity >= 0 else None
        self.priority = max(0.1, priority)
        self.setup_cost = setup_cost
        self.is_active = is_active

        # Laufzeitdaten initialisieren
        self.capacity_share = 0.0
        self.target_output = 0.0
        self.planned_output = 0.0
        self.actual_output = 0.0
        self.utilization = 0.0
        self.output_history = deque(maxlen=50)
        self.bottleneck_info = None
        self.planned_consumed_inputs = {}

    def calculate_inputs_for_output(self, output_amount: float) -> Dict[str, float]:
        """Berechnet benötigte Inputs für eine gegebene Outputmenge, unter Berücksichtigung der *effektiven* Effizienz."""
        if self.effective_efficiency <= 1e-6: return {} # Keine Produktion bei Effizienz 0
        inputs = {}
        required_output_factor = output_amount / self.effective_efficiency
        for req in self.input_requirements:
            inputs[req.resource_type] = required_output_factor * req.amount_per_unit
        return inputs

    def calculate_emissions_for_output(self, output_amount: float) -> Dict[str, float]:
        """Berechnet Emissionen für eine gegebene Outputmenge, unter Berücksichtigung der *effektiven* Effizienz."""
        if self.effective_efficiency <= 1e-6: return {}
        scaled_profile = self.emission_profile.scale(output_amount / self.effective_efficiency)
        return scaled_profile.emissions

    def update_base_efficiency(self, new_base_efficiency: float) -> None:
        """Aktualisiert die Basis-Effizienz."""
        self.base_efficiency = max(0.1, new_base_efficiency)
        # Effektive Effizienz wird vom Agenten aktualisiert

    def record_production(self, step: int, actual_output: float, effective_agent_tech_level: float) -> None:
        """Zeichnet Produktionsergebnisse auf."""
        self.actual_output = actual_output
        self.output_history.append((step, actual_output))
        line_capacity = self.capacity_share * effective_agent_tech_level
        if line_capacity > 1e-6:
            self.utilization = actual_output / line_capacity
        else:
            self.utilization = 0.0

    def __repr__(self) -> str:
        return (f"ProductionLine(name={self.name}, output={self.output_good}, "
                f"base_eff={self.base_efficiency:.2f}, active={self.is_active})")


# --- Producer Agent Klasse (stark überarbeitet) ---

class ProducerAgent:
    """
    Fortgeschrittener Produktionsagent für planwirtschaftliche Simulationen.

    Verwaltet mehrere Produktionslinien, reagiert auf Planziele und Ressourcenallokationen,
    trifft Investitionsentscheidungen, betreibt Forschung und passt sich an.
    Kann optional Reinforcement Learning nutzen.
    """
    # --- Typ-Annotationen für Attribute ---
    unique_id: str
    model: 'EconomicModel'
    logger: logging.Logger
    region_id: Union[int, str]
    production_lines: List[ProductionLine]
    can_produce: Set[str]
    productive_capacity: float # Gesamt-Kapazität des Agenten
    tech_level: float # Globaler Technologie-/Wissens-Multiplikator des Agenten
    innovation_potential: float
    depreciation_rate: float
    investment_budget: float
    investment_allocation: DefaultDict[str, float]
    investment_efficiency: DefaultDict[str, float]
    resource_stock: DefaultDict[str, float]
    output_stock: DefaultDict[str, float]
    resource_consumption_history: DefaultDict[str, Deque[float]] # Tracking pro Ressource
    production_target: DefaultDict[str, float] # Ziele von S3/S4
    production_priority: DefaultDict[str, float] # Lokale Prioritäten (können von global abweichen)
    emissions_history: DefaultDict[str, Deque[Tuple[int, float]]]
    maintenance_status: float
    maintenance_history: Deque[Tuple[int, float]]
    investment_history: Deque[Tuple[int, Dict[str, Any]]]
    research_progress: DefaultDict[str, float]
    innovation_capability: DefaultDict[str, float]
    total_output_this_step: float
    cumulative_output: DefaultDict[str, float]
    output_history: Deque[Tuple[int, float]]
    capacity_utilization: float
    productivity_trend: Deque[float]
    rl_mode: bool
    # Union Type für rl_agent, um sowohl QLearningAgent als auch DeepQLearningAgent aufzunehmen
    rl_agent: Optional[Union[QLearningAgent, DeepQLearningAgent]]
    _rl_state_cache: Optional[Tuple[np.ndarray, int, float]] # state, action, reward (reward from PREVIOUS step)
    _last_rl_state: Optional[np.ndarray] # Cache for the state before the current action
    _last_resource_needs: Dict[str, float] # Cache für letzten Bedarf
    _prev_tech_level: float
    _prev_avg_efficiency: float
    _prev_maintenance_status: float

    def __init__(
        self,
        unique_id: str,
        model: 'EconomicModel',
        region_id: Union[int, str],
        production_lines: List[Dict[str, Any]], # Erwarte Dicts aus Config
        initial_capacity: float = 100.0,
        initial_tech_level: float = 1.0,
        depreciation_rate: float = 0.02,
        enable_rl: bool = False,
        innovation_potential: float = 1.0,
        skills: Optional[Dict[str, int]] = None, # Added skills parameter
        initial_task_queue_length: int = 0, # Added task_queue_length parameter
        # Weitere Parameter über **kwargs aus Config (params)
        **kwargs
    ):
        """
        Initialisiert einen neuen ProducerAgent.

        Args:
            unique_id: Eindeutige ID.
            model: Referenz zum Hauptmodell.
            region_id: ID der Region.
            production_lines: Liste von Dictionaries, die Produktionslinien definieren.
                                Jedes Dict muss die Argumente für ProductionLine.__init__ enthalten.
            initial_capacity: Anfängliche Gesamtproduktionskapazität.
            initial_tech_level: Anfängliches globales Technologieniveau des Agenten.
            depreciation_rate: Jährliche Abschreibungsrate der Kapazität (0-1).
            enable_rl: Ob Reinforcement Learning aktiviert werden soll.
            innovation_potential: Grundfähigkeit des Agenten zu Innovation (Faktor).
            skills: Optional dictionary of skill names to levels.
            initial_task_queue_length: Initial length of the task queue.
            **kwargs: Zusätzliche Parameter aus der Konfiguration.
        """
        self.unique_id = unique_id
        self.model = model
        self.logger = model.logger # Zentraler Logger
        self.region_id = region_id

        # Produktionslinien aus Konfiguration erstellen
        self.production_lines = self._create_production_lines(production_lines)
        self.can_produce = set(line.output_good for line in self.production_lines)

        # Kapazität und Technologie
        self.productive_capacity = max(1.0, initial_capacity) # Mind. 1
        self.tech_level = max(0.1, initial_tech_level) # Mind. 0.1
        self.innovation_potential = max(0.1, innovation_potential)

        # Skills
        self.skills = skills if skills is not None else {}
        if self.skills:
            self.logger.info(f"Producer {self.unique_id}: Initialized with skills: {self.skills}")

        # Task Queue Length
        self.task_queue_length = initial_task_queue_length

        # Kapital und Investitionen
        self.depreciation_rate = np.clip(depreciation_rate, 0.0, 0.1)
        self.investment_budget = 0.0
        self.investment_allocation = defaultdict(float, kwargs.get("initial_investment_allocation", {}))
        # Effizienz der Investitionen kann auch konfiguriert werden
        base_inv_eff = kwargs.get("base_investment_efficiency", 1.0)
        self.investment_efficiency = defaultdict(lambda: random.uniform(base_inv_eff * 0.7, base_inv_eff * 1.3))

        # Ressourcen und Bestände
        self.resource_stock = defaultdict(float, kwargs.get("initial_resource_stock", {}))
        self.output_stock = defaultdict(float) # Produzierte Güter
        self.resource_consumption_history = defaultdict(lambda: deque(maxlen=50))

        # Planungsinformationen
        self.production_target = defaultdict(float) # Wird von außen gesetzt
        # Lokale Prioritäten können von globalen abweichen
        self.production_priority = defaultdict(lambda: 1.0, kwargs.get("initial_priorities", {}))

        # Emissionen
        self.emissions_history = defaultdict(lambda: deque(maxlen=100))

        # Wartung & Historien
        self.maintenance_status = kwargs.get("initial_maintenance_status", 1.0)
        self.maintenance_history = deque(maxlen=50)
        self.investment_history = deque(maxlen=50)
        self.research_progress = defaultdict(float, kwargs.get("initial_research", {}))
        self.innovation_capability = defaultdict(lambda: random.uniform(0.5, 1.5), kwargs.get("innovation_capability", {}))

        # Laufzeit-Metriken
        self.total_output_this_step = 0.0
        self.cumulative_output = defaultdict(float)
        self.output_history = deque(maxlen=100)
        self.capacity_utilization = 0.0
        self.productivity_trend = deque(maxlen=10)
        self._last_resource_needs = {}
        self._last_consumption_dict: Dict[str, float] = {}
        self._last_production_dict: Dict[str, float] = {}
        # Parameters for simple federated learning updates
        self.local_model_params = np.zeros(3, dtype=np.float32)
        self.local_solution = defaultdict(float)

        # Global scarcity signals from System 4
        self.shadow_prices: Dict[str, float] = {}

        # RL-Komponente
        self.rl_mode = enable_rl and RL_AVAILABLE # RL_AVAILABLE prüft q_learning.py
        self.rl_agent = None
        self._rl_state_cache = None
        self._last_rl_state = None # Initialisiere _last_rl_state

        # RL-Parameter aus kwargs holen, mit Standardwerten
        self.rl_config = kwargs.get("rl_params", {})
        self.num_key_products_rl = self.rl_config.get("num_key_products_for_state", 3)


        if self.rl_mode:
            # Definiere state_dim basierend auf der Logik in get_state_for_rl
            # Basis: util, avg_eff, maint, shortage, tech, budget = 6
            # Key Products: num_key_products_rl * 1 (fulfillment)
            # Trends: eff_trend, maint_trend = 2
            plan_goods_dim = len(getattr(self.model.config, "plan_relevant_goods", getattr(self.model.config, "goods", [])))
            _temp_state_dim = 6 + self.num_key_products_rl + 2 + plan_goods_dim
            _action_dim = 6 # Entspricht den Aktionen in do_rl_action

            if TORCH_AVAILABLE and self.rl_config.get("use_dqn", True): # Default zu DQN wenn möglich
                try:
                    self.rl_agent = DeepQLearningAgent(
                        state_dim=_temp_state_dim,
                        action_dim=_action_dim,
                        **self.rl_config.get("dqn_params", {}) # Übergib DQN-spezifische Parameter
                    )
                    self.logger.info(f"Producer {self.unique_id}: Deep Q-Learning Modus aktiviert (State: {_temp_state_dim}, Action: {_action_dim}).")
                except Exception as e:
                    self.logger.error(f"Fehler bei Initialisierung des DeepQLearningAgent für {self.unique_id}: {e}. Fallback zu Tabular/Dummy.", exc_info=True)
                    # Fallback zu tabularem (oder Dummy) QLearningAgent
                    self.rl_agent = QLearningAgent(state_dim=_temp_state_dim, action_dim=_action_dim, **self.rl_config.get("ql_params", {}))
                    self.logger.info(f"Producer {self.unique_id}: Fallback zu tabularem Q-Learning (State: {_temp_state_dim}, Action: {_action_dim}).")
            else: # Tabulares QL oder Dummy QL (wenn Torch nicht da ist, wird Dummy QL aus q_learning.py importiert)
                self.rl_agent = QLearningAgent(
                    state_dim=_temp_state_dim, # Auch hier korrekte state_dim
                    action_dim=_action_dim,
                    **self.rl_config.get("ql_params", {}) # Übergib QL-spezifische Parameter
                )
                status_msg = "Tabular Q-Learning Modus aktiviert." if RL_AVAILABLE else "RL (Dummy) Modus aktiviert."
                self.logger.info(f"Producer {self.unique_id}: {status_msg} (State: {_temp_state_dim}, Action: {_action_dim}).")
        
        # Initialwerte für Reward-Berechnung speichern
        self._prev_tech_level = self.tech_level
        self._prev_avg_efficiency = self.calculate_average_efficiency()
        self._prev_maintenance_status = self.maintenance_status

        # Initialisierung abschließen
        self._update_effective_efficiencies()
        self.initialize_production_lines() # Kapazität verteilen
        self.logger.info(f"ProducerAgent {self.unique_id} initialisiert in Region {self.region_id} "
                         f"mit Kapazität {self.productive_capacity:.1f} und Tech-Level {self.tech_level:.2f}.")

    def _create_production_lines(self, line_configs: List[Dict[str, Any]]) -> List[ProductionLine]:
        """Erstellt ProductionLine-Objekte aus Konfigurations-Dictionaries."""
        lines = []
        for i, cfg in enumerate(line_configs):
            try:
                # Konvertiere Input-Requirements Dicts in ResourceRequirement Objekte
                input_req_data = cfg.get("input_requirements", [])
                input_requirements = []
                for req in input_req_data:
                    if isinstance(req, ResourceRequirement):
                        input_requirements.append(req)
                    else:
                        input_requirements.append(ResourceRequirement(**req))

                # Emissionsprofil erstellen
                emission_profile = EmissionProfile(emissions=cfg.get("emission_factors", {}))

                # Erstelle Linie
                line = ProductionLine(
                    name=cfg.get("name", f"Line_{i+1}_{cfg.get('output_good', 'Unknown')}"),
                    output_good=cfg["output_good"], # Erforderlich
                    input_requirements=input_requirements,
                    emission_profile=emission_profile,
                    base_efficiency=cfg.get("base_efficiency", cfg.get("efficiency", 1.0)), # Nutze 'base_efficiency' oder altes 'efficiency'
                    min_utilization=cfg.get("min_utilization", 0.1),
                    max_capacity=cfg.get("max_capacity"),
                    priority=cfg.get("priority", 1.0),
                    setup_cost=cfg.get("setup_cost", 0.0),
                    is_active=cfg.get("is_active", True)
                )
                lines.append(line)
            except KeyError as e:
                self.logger.error(f"Fehlender Schlüssel in production_line Konfiguration {i}: {e}. Überspringe Linie.")
            except Exception as e:
                self.logger.error(f"Fehler beim Erstellen der Produktionslinie {i} aus Config {cfg}: {e}", exc_info=True)
        return lines

    def initialize_production_lines(self) -> None:
        """Initialisiert Kapazitätsanteile und effektive Effizienz der Linien."""
        self._update_effective_efficiencies()
        self.update_capacity_allocation() # Nutzt die zentrale Methode

    def _update_effective_efficiencies(self) -> None:
        """Aktualisiert die effektive Effizienz aller Linien basierend auf agent.tech_level."""
        for line in self.production_lines:
            # Effektive Effizienz = Basis * TechLevel * Wartungszustand (Beispiel)
            maintenance_factor = 0.8 + 0.2 * self.maintenance_status # 80% bei 0, 100% bei 1
            line.effective_efficiency = line.base_efficiency * self.tech_level * maintenance_factor
            line.effective_efficiency = max(0.01, line.effective_efficiency) # Minimalwert

    # --- Stage Execution ---
    def step_stage(self, stage: str, **kwargs) -> Optional[Any]:
        """Führt Aktionen für die angegebene Phase aus."""
        self.logger.debug(f"Producer {self.unique_id}: Executing stage '{stage}'")
        current_step = getattr(self.model, "current_step", 0)

        # Update effektive Effizienz zu Beginn relevanter Phasen
        if stage in ["local_production_planning", "production_execution"]:
            self._update_effective_efficiencies()

        if stage == "local_production_planning":
            self.plan_local_production()
        elif stage == "production_execution":
            self.execute_production()
        elif stage == "distribute_output": # Stage umbenannt / hinzugefügt?
            return self.get_produced_output() # Gibt Output zurück zur Verteilung
        elif stage == "investment_and_depreciation":
            self.update_depreciation()
            self.plan_investments()
            self.execute_investments()
        elif stage == "innovation_and_research":
            self.advance_research()
            self.apply_innovation()
        elif stage == "maintenance":
            self.perform_maintenance(**kwargs)
        elif stage == "receive_targets": # Neue Stage, um Ziele zu empfangen
            targets = kwargs.get("targets", {})
            resources = kwargs.get("resources", {})
            self.receive_production_targets(targets=targets, resources=resources)
        elif stage == "reporting":
            return self.generate_report() # Umbenannt für Konsistenz

        # RL Step (unabhängig von Stage?) oder in einer spezifischen Stage ausführen
        if self.rl_mode and self.rl_agent and stage == self.rl_config.get("rl_learning_stage", "investment_and_depreciation"):
            self.step_rl() # RL-Schritt nach relevanten Zustandsänderungen

        # ADMM Subproblem wird typischerweise direkt vom Planner aufgerufen, nicht über Stage
        # elif stage == "admm_coordination": ...

        return None # Standardmäßig nichts zurückgeben


    # --- Kernlogik Methoden (Refactored) ---

    def receive_production_targets(self, targets: Dict[str, float], resources: Optional[Dict[str, float]] = None) -> None:
        """Empfängt neue Produktionsziele und Ressourcenallokationen."""
        # Verarbeite Ziele (könnte komplexer sein, z.B. Glättung)
        if targets is not None:
            self.production_target = defaultdict(float, targets)
            self.logger.info(f"Producer {self.unique_id}: Neue Produktionsziele erhalten für {list(targets.keys())}.")
            # Nach Erhalt neuer Ziele sollten Linienziele neu verteilt werden
            self.distribute_targets_to_lines()

        # Verarbeite Ressourcen
        if resources is not None:
            for resource, amount in resources.items():
                if amount > 0:
                    self.resource_stock[resource] += amount
            self.logger.info(f"Producer {self.unique_id}: Ressourcen erhalten: {[(r, a) for r, a in resources.items() if a > 0]}.")

    def distribute_targets_to_lines(self) -> None:
        """Verteilt die Gesamtziele des Agenten auf die einzelnen Produktionslinien."""
        # Reset alte Linienziele
        for line in self.production_lines:
            line.target_output = 0.0

        # Gruppiere Linien nach Output
        lines_by_good: DefaultDict[str, List[ProductionLine]] = defaultdict(list)
        for line in self.production_lines:
            if line.is_active:
                lines_by_good[line.output_good].append(line)

        # Verteile Ziel für jedes Gut auf die Linien, die es produzieren können
        for good, agent_target in self.production_target.items():
            if agent_target <= 0 or good not in lines_by_good:
                continue

            relevant_lines = lines_by_good[good]
            # Gewichtung: Priorität * (Basis-)Effizienz? Oder nur Priorität? Hier: Prio * BaseEff
            total_weight = sum(line.priority * line.base_efficiency for line in relevant_lines)

            if total_weight <= 0: # Fallback: Gleichverteilung
                share = agent_target / len(relevant_lines) if relevant_lines else 0
                for line in relevant_lines:
                    line.target_output = min(share, line.max_capacity or float('inf'))
            else:
                # Gewichtete Verteilung
                remaining_target = agent_target
                # Sortieren, um Linien mit max_capacity zuerst zu füllen? Oder Prio zuerst? Hier Prio*Eff
                relevant_lines.sort(key=lambda l: l.priority * l.base_efficiency, reverse=True)

                temp_targets = {}
                for line in relevant_lines:
                    weight = (line.priority * line.base_efficiency) / total_weight
                    target = agent_target * weight
                    # Berücksichtige max_capacity der Linie
                    capped_target = min(target, line.max_capacity or float('inf'))
                    temp_targets[line.name] = capped_target
                    remaining_target -= capped_target

                # Verteile ggf. Reste auf Linien ohne Limit oder mit freier Kapazität
                if remaining_target > 0.01: # Signifikanter Rest
                    unlimited_lines = [l for l in relevant_lines if temp_targets[l.name] < (l.max_capacity or float('inf')) - 1e-6]
                    additional_weight = sum(l.priority * l.base_efficiency for l in unlimited_lines)
                    if additional_weight > 0:
                        for line in unlimited_lines:
                            weight = (line.priority * l.base_efficiency) / additional_weight
                            extra = remaining_target * weight
                            # Erneut auf max_capacity prüfen
                            current_target = temp_targets[line.name]
                            extra_capped = min(extra, (line.max_capacity or float('inf')) - current_target)
                            temp_targets[line.name] += extra_capped
                            # Ggf. immer noch ein kleiner Rest durch Rundung/Capping

                # Setze die finalen Linienziele
                for line in relevant_lines:
                    line.target_output = temp_targets.get(line.name, 0.0)

        self.logger.debug(f"Producer {self.unique_id}: Agenten-Ziele auf Linien verteilt.")


    def plan_local_production(self) -> None:
        """
        Plant die Produktion für den nächsten Schritt basierend auf Zielen und Ressourcen.
        Setzt `line.planned_output`, `line.bottleneck_info` und `line.planned_consumed_inputs`.
        """
        self.logger.debug(f"Producer {self.unique_id}: Starte lokale Produktionsplanung...")
        # 1. Kapazitätsallokation updaten (basierend auf aktuellen Zielen/Prioritäten)
        self.update_capacity_allocation()

        # 2. Ressourcenbedarf für *Linienziele* berechnen (vor Ressourcenprüfung)
        self._last_resource_needs = self.calculate_resource_needs(use_target_output=True)

        # 3. Für jede Linie: Prüfe Ressourcen und bestimme realisierbaren Output
        total_planned_output_value = 0.0
        temp_resource_stock = self.resource_stock.copy() # Kopie für Simulation der Zuteilung

        # Linien sortiert nach Priorität bearbeiten (höchste zuerst)
        sorted_lines = sorted([line for line in self.production_lines if line.is_active and line.target_output > 0],
                              key=lambda l: l.priority, reverse=True)

        for line in sorted_lines:
            target_output = line.target_output
            if target_output <= 0:
                line.planned_output = 0.0
                line.planned_consumed_inputs = {} # **FIX**: Reset planned inputs
                continue

            # Bestimme maximalen Output basierend auf Linie + Agenten-Tech-Level
            # Die Kapazität ist jetzt im 'capacity_share' enthalten
            line_capacity_limit = line.capacity_share # * self.tech_level ? Nein, tech_level ist in effective_eff
            if line.max_capacity is not None:
                line_capacity_limit = min(line_capacity_limit, line.max_capacity)

            # Ziel kann nicht höher sein als die Linienkapazität
            effective_target = min(target_output, line_capacity_limit)
            if effective_target <= 0:
                line.planned_output = 0.0
                line.planned_consumed_inputs = {} # **FIX**: Reset planned inputs
                continue

            # Prüfe Ressourcenverfügbarkeit für 'effective_target'
            feasible_output, bottleneck_info, consumed = self._determine_feasible_output(line, effective_target, temp_resource_stock)

            # Berücksichtige Mindestauslastung der Linie
            min_output_threshold = line.min_utilization * line_capacity_limit
            if 0 < feasible_output < min_output_threshold:
                self.logger.debug(
                    f"Linie {line.name}: Geplanter Output {feasible_output:.2f} unter Mindestauslastung ({min_output_threshold:.2f}). Setze auf 0."
                )
                feasible_output = 0.0
                line.planned_output = 0.0
                line.bottleneck_info = bottleneck_info
                line.planned_consumed_inputs = {}
                # Ressourcen NICHT reservieren, wenn Linie nicht läuft
                consumed.clear()  # Reset consumed
            else:
                line.planned_output = feasible_output
                line.bottleneck_info = bottleneck_info
                line.planned_consumed_inputs = consumed # **FIX**: Store the calculated inputs
                total_planned_output_value += feasible_output

                # Reserviere Ressourcen (reduziere temp stock)
                for resource, amount in consumed.items():
                    temp_resource_stock[resource] = temp_resource_stock.get(resource, 0.0) - amount

        self.logger.info(f"Producer {self.unique_id}: Lokale Planung abgeschlossen. Gesamt geplanter Output: {total_planned_output_value:.2f}")

    def _determine_feasible_output(self,
        line: ProductionLine,
        target_output: float,
        available_resources: Dict[str, float]
    ) -> Tuple[float, Optional[Dict[str, Any]], Dict[str, float]]:
        """
        Prüft Ressourcenverfügbarkeit für eine Linie und gibt den machbaren Output zurück.
        Berücksichtigt auch Substitutionen.

        Args:
            line: Die zu prüfende Produktionslinie.
            target_output: Der angestrebte Output für diese Linie.
            available_resources: Die *aktuell noch verfügbaren* Ressourcen für diese Prüfung.

        Returns:
            Tuple: (realisierbarer_output, bottleneck_info, verbrauchte_ressourcen)
                    bottleneck_info ist None oder Dict{'resource': str, 'factor': float}.
                    verbrauchte_ressourcen ist Dict{resource: amount}.
        """
        if target_output <= 0: return 0.0, None, {}

        required_inputs = line.calculate_inputs_for_output(target_output)
        limiting_factor = 1.0
        bottleneck_resource: Optional[str] = None
        # Sammle spaeter berechnete Verbraeuche
        planned_consumption = defaultdict(float)

        # --- Ressourcenprüfung & Substitution ---
        for resource, required_amount in required_inputs.items():
            if required_amount <= 0:
                continue

            available_orig = available_resources.get(resource, 0.0)
            total_possible = available_orig

            for req in line.input_requirements:
                if req.resource_type == resource:
                    for sub_res, sub_factor in req.substitutes:
                        if sub_factor <= 1e-9:
                            continue
                        total_possible += available_resources.get(sub_res, 0.0) * sub_factor

            current_limit = min(1.0, total_possible / required_amount) if required_amount > 1e-9 else 1.0
            if current_limit < limiting_factor:
                limiting_factor = current_limit
                bottleneck_resource = resource

            final_required = required_amount * current_limit
            use_orig = min(available_orig, final_required)
            planned_consumption[resource] += use_orig
            remaining = final_required - use_orig

            for req in line.input_requirements:
                if req.resource_type == resource and remaining > 1e-9:
                    for sub_res, sub_factor in req.substitutes:
                        if sub_factor <= 1e-9:
                            continue
                        available_sub = available_resources.get(sub_res, 0.0)
                        use_sub = min(available_sub, remaining / sub_factor)
                        if use_sub > 0:
                            planned_consumption[sub_res] += use_sub
                            remaining -= use_sub * sub_factor
                        if remaining <= 1e-9:
                            break

        # Skaliere Output basierend auf dem limitierenden Faktor
        feasible_output = target_output * limiting_factor

        # Berechne finalen Ressourcenverbrauch basierend auf der tatsaechlichen Produktionsmenge
        final_consumed = defaultdict(float)
        for resource, required_amount in required_inputs.items():
            final_required = required_amount * limiting_factor
            available_orig = available_resources.get(resource, 0.0)
            use_orig = min(available_orig, final_required)
            final_consumed[resource] += use_orig
            remaining = final_required - use_orig
            for req in line.input_requirements:
                if req.resource_type == resource and remaining > 1e-9:
                    for sub_res, sub_factor in req.substitutes:
                        if sub_factor <= 1e-9:
                            continue
                        available_sub = available_resources.get(sub_res, 0.0)
                        use_sub = min(available_sub, remaining / sub_factor)
                        if use_sub > 0:
                            final_consumed[sub_res] += use_sub
                            remaining -= use_sub * sub_factor
                        if remaining <= 1e-9:
                            break

        # Erstelle Bottleneck Info
        bottleneck_info = None
        if bottleneck_resource:
            bottleneck_info = {"resource": bottleneck_resource, "factor": limiting_factor}

        return feasible_output, bottleneck_info, dict(final_consumed)


    def execute_production(self) -> None:
        """Führt die geplante Produktion durch, verbraucht Ressourcen, erzeugt Output & Emissionen."""
        logger = self.logger # Cache logger
        logger.debug(f"Producer {self.unique_id}: Starte Produktionsausführung...")
        self.total_output_this_step = 0.0
        self._last_production_dict = defaultdict(float)
        total_emissions_this_step = defaultdict(float)
        current_step = getattr(self.model, 'current_step', 0)
        consumed_globally = defaultdict(float) # Für Verbrauchs-History
        
        # Cache attributes accessed in loops
        resource_stock = self.resource_stock
        output_stock = self.output_stock
        tech_level = self.tech_level
        production_lines = self.production_lines
        emissions_history = self.emissions_history

        active_planned_lines = [
            line for line in production_lines
            if line.is_active and line.planned_output > 0
        ]

        for line in active_planned_lines:
            output_amount = line.planned_output  # Nutze den *geplanten* Output

            # --- Ressourcenverbrauch ---
            # **FIX**: Bevorzuge geplante Verbräuche aus der Planungsphase
            inputs_needed = getattr(line, "planned_consumed_inputs", None)
            if not inputs_needed:
                # Fallback, falls der Plan aus irgendeinem Grund fehlt
                logger.warning(
                    f"Linie {line.name}: 'planned_consumed_inputs' nicht gefunden. "
                    "Berechne Inputs neu für Ausführung (ohne Substitutionen)."
                )
                inputs_needed = line.calculate_inputs_for_output(output_amount)
            
            possible = True
            actual_consumed = defaultdict(float)
            for resource, amount in inputs_needed.items():
                if resource_stock[resource] < amount - 1e-6: # Kleine Toleranz
                    logger.error(f"Producer {self.unique_id}, Linie {line.name}: "
                                 f"Inknsistenz! Nicht genug {resource} für geplanten Output {output_amount:.2f} "
                                 f"(Bedarf: {amount:.2f}, Vorrat: {resource_stock[resource]:.2f}). "
                                 f"Produktion wird abgebrochen/reduziert.")
                    output_amount = 0.0
                    possible = False
                    break 
                else:
                    actual_consumed[resource] = amount

            if not possible:
                line.record_production(current_step, 0.0, tech_level)
                continue

            for resource, amount in actual_consumed.items():
                resource_stock[resource] -= amount
                consumed_globally[resource] += amount

            # --- Output und Emissionen ---
            output_stock[line.output_good] += output_amount
            self.cumulative_output[line.output_good] += output_amount # self.cumulative_output is specific, keep self
            self.total_output_this_step += output_amount
            self._last_production_dict[line.output_good] += output_amount

            line_emissions = line.calculate_emissions_for_output(output_amount)
            for emission_type, amount in line_emissions.items():
                emissions_history[emission_type].append((current_step, amount))
                total_emissions_this_step[emission_type] += amount 

            line.record_production(current_step, output_amount, tech_level)
        
        for line in production_lines: # Iterate over all lines for zero production recording
            if not (line.is_active and line.planned_output > 0): # If not already processed
                line.record_production(current_step, 0.0, tech_level)

        self.output_history.append((current_step, self.total_output_this_step))
        if self.productive_capacity > 0:
            self.capacity_utilization = self.total_output_this_step / self.productive_capacity
            self.productivity_trend.append(self.capacity_utilization)
        else:
            self.capacity_utilization = 0.0

        for resource, amount in consumed_globally.items():
            self.resource_consumption_history[resource].append(amount)

        self._last_consumption_dict = dict(consumed_globally)

        logger.info(f"Producer {self.unique_id}: Produktion ausgeführt. Gesamt-Output: {self.total_output_this_step:.2f}. "
                            f"Kap-Auslastung: {self.capacity_utilization:.1%}. "
                            f"Emissionen (Top 1): {list(total_emissions_this_step.items())[0] if total_emissions_this_step else 'Keine'}.")


    def get_produced_output(self) -> Dict[str, float]:
        """Gibt den produzierten Output zurück und leert das interne Lager."""
        output_stock = self.output_stock # Cache attribute
        output = output_stock.copy()
        output_stock.clear()
        self.logger.debug(f"Producer {self.unique_id}: Gibt Output {output} zur Verteilung frei.")
        return output

    def calculate_resource_needs(self, use_target_output: bool = False) -> Dict[str, float]:
        """
        Berechnet den Ressourcenbedarf für die nächste Runde.

        Args:
            use_target_output: Wenn True, basiert die Berechnung auf line.target_output,
                                sonst auf line.planned_output (dem realistischeren Wert).

        Returns:
            Dict {ressource: gesamtbedarf}.
        """
        needs = defaultdict(float)
        # Cache self.production_lines to avoid repeated attribute lookup in loop
        agent_production_lines = self.production_lines
        for line in agent_production_lines:
            if not line.is_active: continue
            output_basis = line.target_output if use_target_output else line.planned_output
            if output_basis <= 0: continue

            line_needs = line.calculate_inputs_for_output(output_basis)
            for resource, amount in line_needs.items():
                needs[resource] += amount
        self._last_resource_needs = dict(needs) 
        return self._last_resource_needs

    # --- Investition, Wartung, Innovation (Refactored) ---

    def update_depreciation(self) -> None:
        """Wendet Abschreibung auf Kapazität und Effizienz an."""
        # Cache attributes
        productive_capacity_val = self.productive_capacity
        depreciation_rate_val = self.depreciation_rate
        maintenance_status_val = self.maintenance_status
        current_step_val = self.model.current_step # Assuming model is stable enough
        
        old_capacity = productive_capacity_val
        effective_dep_rate = depreciation_rate_val * (1.2 - maintenance_status_val) 
        productive_capacity_val *= (1.0 - effective_dep_rate)
        self.productive_capacity = max(1.0, productive_capacity_val)

        agent_production_lines = self.production_lines # Cache
        for line in agent_production_lines:
            line.update_base_efficiency(line.base_efficiency * (1.0 - effective_dep_rate * 0.5))

        old_maintenance = maintenance_status_val
        self.maintenance_status = max(0.1, maintenance_status_val * (1.0 - 0.015 * random.uniform(0.8, 1.2))) 
        self.maintenance_history.append((current_step_val, self.maintenance_status))

        self._update_effective_efficiencies()
        self.update_capacity_allocation()

        self.logger.debug(f"Producer {self.unique_id}: Abschreibung angewendet. Kap: {old_capacity:.1f}->{self.productive_capacity:.1f}, Maint: {old_maintenance:.2f}->{self.maintenance_status:.2f}")

    def plan_investments(self) -> None:
        """Plant Investitionsbudget und -allokation für die nächste Runde."""
        # 1. Budget bestimmen (komplexere Logik)
        self.investment_budget = self._calculate_investment_budget()

        # 2. Allokation bestimmen
        if self.investment_budget > 0:
            self.investment_allocation = self._determine_investment_allocation()
            self.logger.info(f"Producer {self.unique_id}: Investitionsbudget geplant: {self.investment_budget:.2f}. Allokation: {dict(self.investment_allocation)}")
        else:
            self.investment_allocation.clear()
            self.logger.info(f"Producer {self.unique_id}: Kein Investitionsbudget geplant.")


    def _calculate_investment_budget(self) -> float:
        """Hilfsmethode zur Berechnung des Investitionsbudgets."""
        # Beispiel: Anteil der Kapazität + Bonus für hohe Auslastung/schlechte Wartung
        # Annahme: self.config existiert und hat .get
        base_rate = getattr(self, 'config', {}).get("base_investment_rate", 0.03) # 3% der Kapazität als Basis
        budget = base_rate * self.productive_capacity

        # Auslastungsbonus
        avg_util = np.mean(list(self.productivity_trend)) if self.productivity_trend else 0.5
        if avg_util > 0.85: budget *= 1.3
        elif avg_util > 0.7: budget *= 1.1

        # Wartungsbonus
        if self.maintenance_status < 0.5: budget *= 1.4
        elif self.maintenance_status < 0.75: budget *= 1.15

        # Technologie-Gap Bonus (wenn hinter Durchschnitt zurück)
        # avg_model_tech = self.model.get_average_tech_level() # Modell muss Methode haben
        # if self.tech_level < avg_model_tech * 0.9: budget *= 1.2

        return max(0.0, budget)

    def _determine_investment_allocation(self) -> DefaultDict[str, float]:
        """Hilfsmethode zur Bestimmung der Budget-Allokation."""
        allocation = defaultdict(float)
        needs = {} # Gewichtung der Bedürfnisse

        # Bedarf für Wartung
        needs["maintenance"] = max(0.0, (1.0 - self.maintenance_status))**2 * 2.0 # Starker Bedarf bei <1.0

        # Bedarf für Kapazitätserweiterung
        avg_util = np.mean(list(self.productivity_trend)) if self.productivity_trend else 0.5
        needs["capacity_expansion"] = max(0.0, avg_util - 0.7) * 1.5 # Bedarf bei >70% Auslastung

        # Bedarf für Technologie/Effizienz
        avg_eff = self.calculate_average_efficiency()
        needs["technology_improvement"] = max(0.0, 1.0 - avg_eff) * 1.0 # Bedarf wenn Effizienz < 1.0

        # Bedarf für Forschung
        needs["research"] = self.innovation_potential * 0.5 # Grundrauschen für F&E

        # Normalisiere Bedarfsgewichte
        total_need = sum(needs.values())
        if total_need > 0:
            for category, weight in needs.items():
                allocation[category] = weight / total_need
        else: # Fallback: Gleichverteilung
            num_categories = 4
            for cat in ["maintenance", "capacity_expansion", "technology_improvement", "research"]:
                allocation[cat] = 1.0 / num_categories

        return allocation


    def execute_investments(self) -> None:
        """Führt geplante Investitionen durch."""
        if self.investment_budget <= 0: return

        self.logger.info(f"Producer {self.unique_id}: Führe Investitionen aus (Budget: {self.investment_budget:.2f}).")
        investments_summary = {}
        budget_spent = 0.0

        # Wende Investitionen pro Kategorie an
        for category, fraction in self.investment_allocation.items():
            budget = self.investment_budget * fraction
            if budget <= 0: continue

            # Annahme: self.config existiert und hat .get
            config = getattr(self, 'config', {})

            if category == "capacity_expansion":
                change = self._apply_capacity_investment(budget, config)
                investments_summary["capacity_change"] = change
                budget_spent += budget
            elif category == "technology_improvement":
                change = self._apply_tech_investment(budget, config)
                investments_summary["avg_eff_change"] = change
                budget_spent += budget
            elif category == "maintenance":
                change = self._apply_maintenance_investment(budget)
                investments_summary["maint_change"] = change
                budget_spent += budget
            elif category == "research":
                self._apply_research_investment(budget)
                investments_summary["research_budget"] = budget
                budget_spent += budget

        # Nach Investitionen: Kapazität und Effizienz neu berechnen/verteilen
        self._update_effective_efficiencies()
        self.update_capacity_allocation()

        self.investment_history.append((self.model.current_step, investments_summary))
        self.investment_budget = 0.0 # Budget ausgegeben

    def _apply_capacity_investment(self, budget: float, config: Dict[str, Any]) -> float:
        """Hilfsfunktion: Wendet Kapazitätsinvestition an."""
        efficiency = self.investment_efficiency["capacity"]
        # Kosten pro Kapazitätseinheit (könnte konfiguriert werden)
        cost_per_unit = config.get("capacity_cost_per_unit", 10.0)
        increase = (budget * efficiency) / cost_per_unit
        old_capacity = self.productive_capacity
        self.productive_capacity += increase
        self.logger.debug(f"  Investition Kapazität: +{increase:.2f} (Budget: {budget:.2f})")
        return self.productive_capacity - old_capacity

    def _apply_tech_investment(self, budget: float, config: Dict[str, Any]) -> float:
        """Hilfsfunktion: Wendet Technologie-Investition an (auf Agenten-Level)."""
        efficiency = self.investment_efficiency["technology"]
        # Steigerung des globalen Tech-Levels (abnehmender Grenznutzen)
        increase_factor = config.get("tech_investment_increase_factor", 0.05)
        max_tech_level = config.get("max_tech_level", 2.0)
        increase = increase_factor * math.log(1 + budget * efficiency) * (max_tech_level - self.tech_level) # Mehr Effekt bei niedrigem Level
        old_tech = self.tech_level
        self.tech_level = min(max_tech_level, self.tech_level + increase) 
        self.logger.debug(f"  Investition Technologie: +{increase:.3f} Tech-Level (Budget: {budget:.2f})")
        return self.tech_level - old_tech

    def _apply_maintenance_investment(self, budget: float) -> float:
        """Hilfsfunktion: Wendet Wartungsinvestition an."""
        efficiency = self.investment_efficiency["maintenance"]
        # Verbesserung des Wartungsstatus (abnehmender Grenznutzen)
        increase = 0.1 * math.log(1 + budget * efficiency) * (1.1 - self.maintenance_status)
        old_maint = self.maintenance_status
        self.maintenance_status = min(1.0, self.maintenance_status + increase)
        self.logger.debug(f"  Investition Wartung: +{increase:.3f} Status (Budget: {budget:.2f})")
        return self.maintenance_status - old_maint

    def _apply_research_investment(self, budget: float) -> None:
        """Hilfsfunktion: Verteilt Forschungsbudget auf Bereiche."""
        efficiency = self.investment_efficiency["research"]
        effective_budget = budget * efficiency
        # Annahme: self.model.config.io_config.research_areas existiert
        research_areas = getattr(getattr(getattr(self.model, 'config', {}), 'io_config', {}), 'research_areas', ["process_optimization", "material_science", "automation"]) # Konfigurierbar machen
        
        # Verteile Budget proportional zur Fähigkeit und aktuellem Fortschritt
        weights = {area: self.innovation_capability.get(area, 1.0) * max(0.1, 1.0 - self.research_progress.get(area, 0.0))
                   for area in research_areas}
        total_weight = sum(weights.values())

        if total_weight > 0:
            for area in research_areas:
                area_budget_share = effective_budget * (weights[area] / total_weight)
                # Fortschritt basierend auf Budget (vereinfacht)
                progress_increase = 0.01 * math.log(1 + area_budget_share) # Logarithmisch
                self.research_progress[area] = min(1.0, self.research_progress.get(area, 0.0) + progress_increase)
                self.logger.debug(f"  Investition F&E ({area}): +{progress_increase:.4f} Fortschritt (Budget: {area_budget_share:.2f})")


    def perform_maintenance(self, intensity: float = 0.1) -> None:
        """Führt Wartung durch (kann Kapazität temporär reduzieren)."""
        # Verbessere Status basierend auf Intensität
        self.maintenance_status = min(1.0, self.maintenance_status + 0.05 * intensity)
        # Temporäre Kapazitätsreduktion während Wartung (optional)
        # temp_capacity_reduction = 0.1 * intensity
        # self.productive_capacity *= (1.0 - temp_capacity_reduction)
        self.logger.debug(f"Producer {self.unique_id}: Wartung durchgeführt (Intensität {intensity:.2f}). Neuer Status: {self.maintenance_status:.2f}")


    # --- Forschung und Innovation (Überarbeitet) ---

    def advance_research(self) -> None:
        """Simuliert passiven Forschungsfortschritt über Zeit."""
        # Nur wenn Forschung aktiv und Budget zugewiesen wurde (passiert in execute_investments)
        # Hier nur kleiner, zufälliger Basis-Fortschritt
        for area, progress in list(self.research_progress.items()):
            if 0 < progress < 1.0:
                base_rate = 0.002 # Sehr langsam ohne Budget
                capability = self.innovation_capability.get(area, 1.0)
                self.research_progress[area] = min(1.0, progress + base_rate * capability * random.uniform(0.5, 1.5))

    def apply_innovation(self) -> None:
        """Wendet abgeschlossene Forschungsprojekte (Fortschritt >= 1.0) an."""
        innovations_applied = False
        io_cfg = getattr(getattr(self.model, 'config', None), 'io_config', None)
        if isinstance(io_cfg, dict):
            innovation_cfg = io_cfg.get('innovation_effects', {})
        elif hasattr(io_cfg, 'innovation_effects'):
            innovation_cfg = io_cfg.innovation_effects
        else:
            innovation_cfg = {}

        for area, progress in list(self.research_progress.items()):
            if progress >= 1.0:
                self.logger.info(f"Producer {self.unique_id}: Wendet Innovation aus Bereich '{area}' an!")
                effect = innovation_cfg.get(area)

                if isinstance(effect, InnovationEffect):
                    effect = effect.model_dump()

                if effect:
                    target = effect.get("target")
                    type = effect.get("type")
                    value = effect.get("value")
                    scope = effect.get("scope", "agent") # agent, all_lines, specific_line, specific_resource

                    try:
                        if target == "line_efficiency":
                            for line in self.production_lines:
                                # TODO: Scope berücksichtigen
                                old_eff = line.base_efficiency
                                if type == "multiplicative": line.update_base_efficiency(old_eff * value)
                                elif type == "additive": line.update_base_efficiency(old_eff + value)
                        elif target == "input_requirement":
                            resource_to_affect = effect.get("resource") # Optional: nur spezifische Ressource
                            for line in self.production_lines:
                                for req in line.input_requirements:
                                    if resource_to_affect is None or req.resource_type == resource_to_affect:
                                        old_amount = req.amount_per_unit
                                        if type == "multiplicative": req.amount_per_unit *= value
                                        elif type == "additive": req.amount_per_unit = max(0, old_amount + value)
                        elif target == "agent_capacity":
                            old_cap = self.productive_capacity
                            if type == "multiplicative": self.productive_capacity *= value
                            elif type == "additive": self.productive_capacity += value
                            self.productive_capacity = max(1.0, self.productive_capacity)
                            if self.productive_capacity != old_cap:
                                self.update_capacity_allocation() # Wichtig: Kapazität neu verteilen!
                        # TODO: Weitere Targets implementieren (z.B. emission_factor, new_line)

                        innovations_applied = True
                        # Reset Fortschritt und erhöhe Fähigkeit leicht
                        self.research_progress[area] = 0.0
                        self.innovation_capability[area] *= random.uniform(1.0, 1.05)
                        self.logger.info(f"  Effekt '{target}' ({type}) mit Wert {value} angewendet.")

                    except Exception as e:
                        self.logger.error(f"Fehler beim Anwenden von Innovationseffekt '{area}': {e}", exc_info=True)
                        self.research_progress[area] = 0.9 # Setze leicht zurück, um Fehler zu vermeiden

                else:
                    self.logger.warning(f"Kein Innovationseffekt für abgeschlossene Forschung '{area}' in Config gefunden.")
                    self.research_progress[area] = 0.0 # Reset trotzdem


    # --- ADMM Subproblem (Überarbeitet mit Fallback-Handling) ---

    # Standardkosten, falls keine Lambdas bereitgestellt werden
    DEFAULT_RESOURCE_COST = 1.0
    DEFAULT_EMISSION_COST = 1.0
    EXCESS_PRODUCTION_PENALTY = 1000.0 # Strafe pro Einheit über Kapazität
    NO_LINE_PENALTY_FACTOR = 1000000.0 # Strafe, wenn keine Linie für ein Gut existiert

    def local_subproblem_admm(self,
                              goods: List[str],
                              lambdas: Dict[str, float],
                              z_vals: Dict[str, float],
                              u_vals: Dict[str, float],
                              rho: float,
                              alpha_params: Optional[Dict[str, float]] = None, # Optional gemacht
                              solver_method: str = 'SLSQP' # Neuer Parameter für Solver-Methode
                              ) -> Dict[str, float]:
        """Löst das lokale ADMM-Subproblem (x-Update)."""
        self.logger.debug(f"Producer {self.unique_id}: Starte ADMM Subproblem-Lösung...")
        if alpha_params is None: alpha_params = {}

        # Stelle sicher, dass die effektiven Effizienzen aktuell sind
        self._update_effective_efficiencies()

        local_goods = [g for g in goods if g in self.can_produce]
        if not local_goods: return {}

        # Initialisierung (z.B. aktuelles Ziel oder Konsens)
        x_init = {g: self.production_target.get(g, z_vals.get(g, 0.0)) for g in local_goods}

        # Zielfunktion (lokale Kosten + ADMM Term)
        def objective_function(x_arr: np.ndarray) -> float:
            x_dict = {g: x_arr[i] for i, g in enumerate(local_goods)}
            cost = self._calculate_local_cost(x_dict, lambdas, alpha_params)
            penalty = self._calculate_admm_penalty(x_dict, z_vals, u_vals, rho)
            return cost + penalty

        # Constraints (Kapazität)
        def capacity_constraint_scipy(x_arr: np.ndarray) -> float:
            return self.productive_capacity - sum(x_arr)

        constraints = [{'type': 'ineq', 'fun': capacity_constraint_scipy}]
        bounds = [(0, None)] * len(local_goods) # Nicht-negativ
        x0_arr = np.array([x_init.get(g, 0.0) for g in local_goods])

        # --- Optimierung ---
        solution: Dict[str, float] = {}
        optimization_success = False

        if SCIPY_AVAILABLE:
            self.logger.debug(f"Versuche Optimierung mit SciPy (Solver: {solver_method})...")
            try:
                result = minimize(
                    fun=objective_function,
                    x0=x0_arr,
                    method=solver_method, # Guter Allrounder für non-linear mit Constraints
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 50} # Begrenze Iterationen
                )
                if result.success:
                    solution = {g: max(0.0, result.x[i]) for i, g in enumerate(local_goods)} # Sicherstellen >= 0
                    # Prüfe Constraint nochmal explizit wegen numerischer Ungenauigkeiten
                    if sum(solution.values()) > self.productive_capacity * 1.001:
                        self.logger.warning(f"SciPy Lösung {sum(solution.values()):.2f} verletzt Kapazität {self.productive_capacity:.2f} leicht. Skaliere...")
                        scale = self.productive_capacity / sum(solution.values())
                        solution = {g: v * scale for g, v in solution.items()}
                    optimization_success = True
                    self.logger.debug(f"SciPy Optimierung erfolgreich (Wert: {result.fun:.3f}).")
                else:
                    self.logger.warning(f"SciPy Optimierung für {self.unique_id} fehlgeschlagen: {result.message}. Nutze Fallback.")
            except Exception as e:
                self.logger.error(f"Exception während SciPy Optimierung für {self.unique_id}: {e}. Nutze Fallback.", exc_info=False)
        else:
            self.logger.warning(f"SciPy nicht verfügbar für {self.unique_id}. Nutze heuristischen Fallback für ADMM.")


        # --- Fallback ---
        if not optimization_success:
            # Pass z_vals and u_vals to the heuristic fallback
            solution = self._admm_heuristic_fallback(local_goods, z_vals, u_vals, objective_function)
            # Stelle sicher, dass Fallback-Lösung Constraints einhält
            # This check might be redundant if the fallback correctly handles capacity, but good for safety.
            if sum(solution.values()) > self.productive_capacity * 1.001: # Allow small tolerance
                self.logger.warning(f"ADMM Fallback solution {sum(solution.values()):.2f} slightly exceeds capacity {self.productive_capacity:.2f}. Scaling down.")
                scale = self.productive_capacity / max(1e-9, sum(solution.values()))
                solution = {g: v * scale for g, v in solution.items()}

        # Speichere lokale Lösung intern für ADMM Residualberechnung
        self.local_solution = solution.copy()
        return solution

    def receive_shadow_prices(self, prices: Dict[str, float]):
        """Store global scarcity signals provided by System 4."""
        self.shadow_prices = prices
        self.logger.debug(f"Producer {self.unique_id} received shadow prices: {prices}")

    def _calculate_local_cost(self, x_dict: Dict[str, float], lambdas: Dict[str, float], alpha_params: Dict[str, float]) -> float:
        """Berechnet die detaillierten lokalen Produktionskosten für eine gegebene Produktionsmenge."""
        self.logger.debug(f"Producer {self.unique_id}: Berechne lokale Kosten für Plan {x_dict}...")
        total_cost = 0.0

        plan_targets = self.production_target

        production_per_line = self._distribute_production_to_lines(x_dict)

        for line, line_production_amount in production_per_line.items():
            if line_production_amount < 1e-6:
                continue

            inputs_needed = line.calculate_inputs_for_output(line_production_amount)
            for resource, amount in inputs_needed.items():
                resource_cost = lambdas.get(resource, 1.0)
                total_cost += amount * resource_cost

            emissions = line.calculate_emissions_for_output(line_production_amount)
            for emission_type, amount in emissions.items():
                emission_penalty = lambdas.get(f"{emission_type}_penalty_{line.output_good}", lambdas.get(f"{emission_type}_penalty", 0.0))
                total_cost += amount * emission_penalty

        for good, planned_amount in x_dict.items():
            plan_target_for_good = plan_targets.get(good, 0.0)

            underproduction = max(0.0, plan_target_for_good - planned_amount)
            if underproduction > 0:
                under_penalty = lambdas.get(f"underproduction_{good}", 1.0)
                total_cost += underproduction * under_penalty

            overproduction = max(0.0, planned_amount - plan_target_for_good)
            if overproduction > 0:
                over_penalty = lambdas.get(f"overproduction_{good}", 0.1)
                inventory_cost = lambdas.get(f"inventory_{good}", 0.05)
                total_cost += overproduction * (over_penalty + inventory_cost)

            fulfilled_production = min(planned_amount, plan_target_for_good)
            production_bonus = lambdas.get(f"production_bonus_{good}", 0.0)
            total_cost -= fulfilled_production * production_bonus

            societal_bonus = lambdas.get(f"societal_bonus_{good}", 0.0)
            total_cost += planned_amount * societal_bonus

        self.logger.debug(f"Producer {self.unique_id}: Berechnete Gesamtkosten: {total_cost:.4f}")
        return total_cost

    def _distribute_production_to_lines(self, x_dict: Dict[str, float]) -> Dict['ProductionLine', float]:
        """Verteilt eine Produktionsmenge für Güter auf die fähigen Produktionslinien."""
        production_per_line = defaultdict(float)

        lines_by_good = defaultdict(list)
        for line in self.production_lines:
            lines_by_good[line.output_good].append(line)

        for good, total_amount in x_dict.items():
            relevant_lines = lines_by_good.get(good, [])
            if not relevant_lines:
                continue

            total_weight = sum(line.effective_efficiency for line in relevant_lines)
            if total_weight > 1e-9:
                for line in relevant_lines:
                    share = line.effective_efficiency / total_weight
                    production_per_line[line] += total_amount * share
            else:
                share = total_amount / len(relevant_lines)
                for line in relevant_lines:
                    production_per_line[line] += share

        return production_per_line

    def _get_best_line_for_good(self, good: str) -> Optional['ProductionLine']:
        """Helper to find a production line for a given good."""
        for line in self.production_lines:
            if line.output_good == good:
                return line
        return None

    def _calculate_admm_penalty(self, x_dict: Dict[str, float], z_vals: Dict[str, float], u_vals: Dict[str, float], rho: float) -> float:
        """Berechnet den ADMM Penalty Term."""
        penalty = 0.0
        for good, x_val in x_dict.items():
            z_val = z_vals.get(good, 0.0)
            u_val = u_vals.get(good, 0.0)
            diff = x_val - z_val + u_val
            penalty += 0.5 * rho * (diff**2)
        return penalty

    def _admm_heuristic_fallback(
        self,
        local_goods: List[str],
        z_vals: Dict[str, float],
        u_vals: Dict[str, float],
        objective_func: Callable # Beibehalten, falls für komplexere Heuristiken benötigt
    ) -> Dict[str, float]:
        """
        Heuristische Alternative zur SciPy-basierten Optimierung für das ADMM x-Update.

        Die Heuristik funktioniert wie folgt:
        1. Berechnet eine anfänglich gewünschte Produktion `x_desired_g = z_g - u_g` für jedes Gut `g`.
           `z_g` sind die Konsenswerte und `u_g` die skalierten dualen Variablen aus ADMM.
        2. Beschneidet diese gewünschten Werte bei Null: `x_clipped_g = max(0, x_desired_g)`.
           Negative Produktionsmengen sind nicht sinnvoll.
        3. Wenn die Summe aller `x_clipped_g` die `self.productive_capacity` des Agenten nicht überschreitet,
           wird diese Menge als Lösung verwendet.
        4. Andernfalls, wenn die Summe die Kapazität überschreitet, werden alle `x_clipped_g` proportional
           herunterskaliert, sodass ihre Summe genau der `self.productive_capacity` entspricht.
        """
        self.logger.debug("Führe ADMM Heuristik-Fallback aus.")
        solution = {}
        x_desired_sum = 0.0

        # 1. & 2. Calculate initial desired production and clip at zero
        for good in local_goods:
            z_g = z_vals.get(good, 0.0)
            u_g = u_vals.get(good, 0.0)
            x_desired_g = z_g - u_g
            x_clipped_g = max(0.0, x_desired_g)
            solution[good] = x_clipped_g
            x_desired_sum += x_clipped_g

        # 3. & 4. Check capacity and scale if necessary
        if x_desired_sum > self.productive_capacity:
            self.logger.debug(f"  Heuristik: Gewünschte Produktion ({x_desired_sum:.2f}) > Kapazität ({self.productive_capacity:.2f}). Skaliere.")
            if self.productive_capacity <= 1e-9: # Praktisch keine Kapazität
                # Setze alles auf 0, wenn keine Kapazität da ist, um Division durch Null zu vermeiden
                for good in local_goods:
                    solution[good] = 0.0
            else:
                scale_factor = self.productive_capacity / x_desired_sum
                for good in local_goods:
                    solution[good] *= scale_factor
        else:
            self.logger.debug(f"  Heuristik: Gewünschte Produktion ({x_desired_sum:.2f}) <= Kapazität ({self.productive_capacity:.2f}). Keine Skalierung nötig.")

        return solution


    # --- RL Methoden (Überarbeitet mit Fallback-Nutzung) ---

    def step_rl(self) -> None:
        """Führt einen Schritt des Reinforcement Learning durch (falls aktiviert)."""
        if not self.rl_mode or not self.rl_agent:
            return

        current_state = self.get_state_for_rl()
        reward = self.calculate_rl_reward() # Reward basiert auf dem Ergebnis der *letzten* Aktion

        if self._last_rl_state is not None and self._rl_state_cache is not None:
            _ , last_action, _ = self._rl_state_cache # Hol die letzte Aktion
            
            # Für DQN: Transition speichern
            if isinstance(self.rl_agent, DeepQLearningAgent):
                self.rl_agent.store_transition(self._last_rl_state, last_action, reward, current_state, False) # done=False
                # DQN lernt aus Batches, nicht pro Schritt unbedingt
                if len(self.rl_agent.memory) >= self.rl_agent.batch_size : # Nur lernen wenn genug Samples
                    loss = self.rl_agent.learn()
                    if loss is not None:
                        self.logger.debug(f"Producer {self.unique_id} DQN learn. Loss: {loss:.4f}")
            # Für tabulares QL: Direkt lernen
            elif isinstance(self.rl_agent, QLearningAgent) and (hasattr(self.rl_agent, 'q_table') or hasattr(self.rl_agent, 'policy_net')): # Sicherstellen, dass es nicht der Dummy ist
                self.rl_agent.learn(self._last_rl_state, last_action, reward, current_state, False) # done=False

        # Wähle neue Aktion basierend auf aktuellem Zustand
        action = self.rl_agent.choose_action(current_state)
        self.do_rl_action(action) # Führe die gewählte Aktion aus

        # Update Caches für den nächsten Schritt
        self._last_rl_state = current_state # Der aktuelle Zustand wird zum letzten Zustand für den nächsten Schritt
        self._rl_state_cache = (current_state, action, reward) # Speichere Zustand, Aktion, und den gerade berechneten Reward

        # Exploration decay (für DQN, bei Tabular passiert es in learn())
        if isinstance(self.rl_agent, DeepQLearningAgent):
            self.rl_agent.decay_exploration()


    def get_state_for_rl(self) -> np.ndarray:
        """Erstellt den normalisierten Zustandsvektor für RL (DQN)."""
        state_parts = []

        # 1. Kapazitätsauslastung (0-1)
        state_parts.append(np.clip(self.capacity_utilization, 0.0, 1.0))

        # 2. Durchschnittliche Effizienz (0-1+, normalisieren)
        # Annahme: max_avg_efficiency könnte 2.0 sein, oder aus config
        max_avg_eff = self.rl_config.get("max_avg_efficiency_norm", 2.0)
        state_parts.append(np.clip(self.calculate_average_efficiency() / max_avg_eff, 0.0, 1.0))

        # 3. Wartungsstatus (0-1)
        state_parts.append(np.clip(self.maintenance_status, 0.0, 1.0))

        # 4. Ressourcenknappheit (0-1)
        needs = self._last_resource_needs # Verwende die zuletzt berechneten Bedürfnisse
        shortage = 0.0
        if needs:
            total_short = sum(max(0, n - self.resource_stock.get(r, 0)) for r, n in needs.items())
            total_need = sum(needs.values())
            shortage = total_short / max(1e-6, total_need) if total_need > 0 else 0.0
        state_parts.append(np.clip(shortage, 0.0, 1.0))

        # 5. Normalisierter Technologiestand (0-1)
        max_tech_level = self.rl_config.get("max_tech_level_norm", 2.0) # aus config oder Standard
        state_parts.append(np.clip(self.tech_level / max_tech_level, 0.0, 1.0))

        # 6. Normalisiertes Investitionsbudget (0-1, relativ zur Kapazität)
        # Annahme: Budget wird selten > 20% der Kapazität sein
        norm_factor = max(1.0, self.productive_capacity * self.rl_config.get("investment_budget_norm_factor", 0.2))
        state_parts.append(np.clip(self.investment_budget / norm_factor, 0.0, 1.0))

        # 7. Erfüllung der Produktionsziele für Top N Produkte (je 0-1)
        # Identifiziere wichtige Produkte (z.B. nach höchstem Ziel)
        sorted_targets = sorted(self.production_target.items(), key=lambda item: item[1], reverse=True)
        for i in range(self.num_key_products_rl):
            fulfillment = 0.0
            if i < len(sorted_targets):
                good, target_qty = sorted_targets[i]
                if target_qty > 1e-6:
                    actual_qty = self.output_stock.get(good, 0.0) + \
                                 sum(line.actual_output for line in self.production_lines if line.output_good == good) # Berücksichtige Output_Stock + was gerade produziert wurde
                    fulfillment = np.clip(actual_qty / target_qty, 0.0, 1.5) # Erlaube leichte Übererfüllung im State
            state_parts.append(fulfillment)

        # 8. Trends für Schlüsselmetriken (Veränderung über X Schritte)
        # Trend Effizienz (normalisiert)
        eff_trend = 0.0
        if self.output_history and len(self.output_history) > 1: # Benötigt Historie für Trend
            # Einfacher Trend: (aktuell - vorher) / (Skalierungsfaktor)
            # Hier verwenden wir _prev_avg_efficiency für direkten Vergleich zum letzten RL-Schritt
            eff_change = self.calculate_average_efficiency() - self._prev_avg_efficiency
            eff_trend = np.clip(eff_change / self.rl_config.get("eff_trend_norm_factor", 0.1), -1.0, 1.0)
        state_parts.append(eff_trend)

        # Trend Wartungsstatus (normalisiert)
        maint_trend = 0.0
        if self.maintenance_history and len(self.maintenance_history) > 1:
            maint_change = self.maintenance_status - self._prev_maintenance_status
            maint_trend = np.clip(maint_change / self.rl_config.get("maint_trend_norm_factor", 0.05), -1.0, 1.0)
        state_parts.append(maint_trend)

        # 9. Wichtigste Schattenpreise
        plan_goods = getattr(self.model.config, "plan_relevant_goods", getattr(self.model.config, "goods", []))
        price_norm_factor = self.rl_config.get("price_norm_factor", 10.0)
        for good in plan_goods:
            price = self.shadow_prices.get(good, 0.0)
            normalized_price = np.clip(price / price_norm_factor, -1.0, 1.0)
            state_parts.append(normalized_price)
        
        final_state = np.array(state_parts, dtype=np.float32)
        
        # Dynamische Anpassung der state_dim im RL-Agenten, falls nötig (beim ersten Mal)
        # Dies ist ein Workaround, idealerweise ist state_dim von Anfang an korrekt.
        if self.rl_agent and hasattr(self.rl_agent, 'state_dim') and self.rl_agent.state_dim != len(final_state):
            self.logger.warning(f"Producer {self.unique_id}: State dimension mismatch. Expected {self.rl_agent.state_dim}, got {len(final_state)}. Re-initializing RL agent if possible or use dummy.")
            # Hier könnte eine Neuinitalisierung versucht werden, ist aber komplex und fehleranfällig.
            # Besser: Sicherstellen, dass _temp_state_dim in __init__ korrekt ist.
            # Für den Moment: Loggen und hoffen, dass es nur beim ersten Mal passiert oder der Dummy greift.
            # Wenn ein Dummy-Agent verwendet wird, hat dieser evtl. keine state_dim oder eine flexible.
            if not (isinstance(self.rl_agent, QLearningAgent) and not hasattr(self.rl_agent, 'q_table')): # Nicht der Dummy QLAgent
                 if not (isinstance(self.rl_agent, DeepQLearningAgent) and not hasattr(self.rl_agent, 'policy_net')): # Nicht der Dummy DQLAgent
                    # Versuche, die state_dim anzupassen, wenn es ein "echter" Agent ist.
                    # Dies ist riskant und sollte vermieden werden.
                    try:
                        self.rl_agent.state_dim = len(final_state)
                        # Bei DQN müsste das Netzwerk neu gebaut werden - sehr problematisch.
                        # Bei Tabular QL könnten die Q-Table Keys ungültig werden.
                        self.logger.error(f"Producer {self.unique_id}: RL Agent state_dim was dynamically adjusted. This is risky!")
                    except:
                        self.logger.error(f"Producer {self.unique_id}: Could not dynamically adjust RL agent state_dim.")


        return final_state


    def do_rl_action(self, action_index: int) -> None:
        """Führt eine vom RL-Agenten gewählte diskrete Aktion aus."""
        self.logger.debug(f"Producer {self.unique_id}: Führt RL-Aktion {action_index} aus.")
        # Annahme: Aktionen sind diskret und haben feste Auswirkungen
        # Diese Aktionen sollten mit den Investitions- und Wartungsphasen koordiniert werden.
        # Hier vereinfachte direkte Anwendung oder Setzen von Flags für die nächste Phase.

        # Standard-Investitionsbetrag (z.B. 10% des aktuellen Budgets oder ein Bruchteil der Kapazität)
        # Dieser Betrag sollte über rl_config konfigurierbar sein.
        base_investment_amount = self.investment_budget * self.rl_config.get("rl_action_investment_fraction", 0.25)
        if base_investment_amount < 1e-3 and self.investment_budget < 1e-3: # Wenn kein Budget, evtl. Fixbetrag basierend auf Kapazität
             base_investment_amount = self.productive_capacity * self.rl_config.get("rl_action_min_cap_fraction_inv", 0.01)


        if action_index == 0: # Investiere in Kapazitätserweiterung
            self.investment_allocation["capacity_expansion"] += base_investment_amount
            self.logger.info(f"RL Action: Allocate {base_investment_amount} to capacity_expansion.")
        elif action_index == 1: # Investiere in Technologieverbesserung
            self.investment_allocation["technology_improvement"] += base_investment_amount
            self.logger.info(f"RL Action: Allocate {base_investment_amount} to technology_improvement.")
        elif action_index == 2: # Führe Wartung durch / Investiere in Wartung
            # Dies könnte direkt Wartung auslösen oder Budget für Wartungsinvestition zuweisen
            self.investment_allocation["maintenance"] += base_investment_amount
            # self.perform_maintenance(intensity=0.2) # Alternative: direkte Aktion
            self.logger.info(f"RL Action: Allocate {base_investment_amount} to maintenance.")
        elif action_index == 3: # Investiere in Forschung
            self.investment_allocation["research"] += base_investment_amount
            self.logger.info(f"RL Action: Allocate {base_investment_amount} to research.")
        elif action_index == 4: # Fokus auf Ressourceneffizienz (könnte spezifische Forschung oder Prozessanpassung sein)
            # Hier könnte man z.B. die Priorität für Forschungsprojekte setzen, die Input-Requirements senken
            # Oder eine kleine Ad-hoc Investition in Linien mit hohem Ressourcenverbrauch tätigen
            # Fürs Erste: Zusätzliches Budget für Technologieforschung mit Fokus Effizienz
            self.investment_allocation["technology_improvement"] += base_investment_amount * 0.5 # Kleinerer Bonus
            self.research_progress["process_optimization"] = min(1.0, self.research_progress.get("process_optimization", 0.0) + 0.05) # Kleiner direkter Boost
            self.logger.info(f"RL Action: Focus on resource efficiency (boost tech & process research).")
        elif action_index == 5: # "Hold" / Konservative Strategie (wenig Investitionen)
            # Reduziere die Allokationen leicht oder mache nichts extra
            for k in self.investment_allocation: self.investment_allocation[k] *= 0.8
            self.logger.info(f"RL Action: Hold / Conservative strategy.")
        else:
            self.logger.warning(f"Unbekannte RL-Aktion: {action_index}")
        
        # Sicherstellen, dass Gesamtallokation nicht Budget übersteigt (falls direkt ausgegeben)
        # Da wir hier nur Allokationen für die execute_investments Phase setzen, ist das weniger kritisch.


    def calculate_rl_reward(self) -> float:
        """Berechnet den Reward für die letzte RL-Aktion basierend auf Zustandsänderungen und Zielerreichung."""
        reward = 0.0
        cfg = self.rl_config.get("reward_weights", {}) # Hole Reward-Gewichte aus Config

        # 1. Planerfüllung (für Top N Produkte)
        plan_fulfillment_reward = 0.0
        sorted_targets = sorted(self.production_target.items(), key=lambda item: item[1], reverse=True)
        for i in range(self.num_key_products_rl):
            if i < len(sorted_targets):
                good, target_qty = sorted_targets[i]
                if target_qty > 1e-6:
                    # actual_qty = self.output_stock.get(good, 0.0) # Output-Stock wurde evtl. schon geleert
                    # Besser: tatsächliche Produktion der letzten Runde
                    actual_qty = sum(line.actual_output for line in self.production_lines if line.output_good == good and line.output_history and line.output_history[-1][0] == self.model.current_step -1)

                    fulfillment_ratio = actual_qty / target_qty
                    # Reward-Struktur: Positiv bei Erfüllung, stark negativ bei starker Untererfüllung
                    if fulfillment_ratio >= 0.95:
                        plan_fulfillment_reward += cfg.get("target_met_bonus", 1.0) * fulfillment_ratio
                    elif fulfillment_ratio < 0.5:
                        plan_fulfillment_reward -= cfg.get("target_miss_penalty_severe", 2.0) * (1.0 - fulfillment_ratio)
                    else:
                        plan_fulfillment_reward -= cfg.get("target_miss_penalty_mild", 0.5) * (1.0 - fulfillment_ratio)
        reward += cfg.get("plan_fulfillment_weight", 1.0) * plan_fulfillment_reward

        # 2. Langfristige Effizienzgewinne
        # Tech-Level Veränderung
        tech_change = self.tech_level - self._prev_tech_level
        if tech_change > 0:
            reward += cfg.get("tech_increase_weight", 0.5) * (tech_change * 10) # Skalieren für größeren Impact
        elif tech_change < 0: # bestrafe leichten Rückgang weniger
            reward -= cfg.get("tech_decrease_penalty", 0.2) * abs(tech_change * 10)

        # Avg. Effizienz Veränderung
        current_avg_eff = self.calculate_average_efficiency()
        avg_eff_change = current_avg_eff - self._prev_avg_efficiency
        if avg_eff_change > 0:
            reward += cfg.get("avg_eff_increase_weight", 0.5) * (avg_eff_change * 10)
        elif avg_eff_change < 0:
            reward -= cfg.get("avg_eff_decrease_penalty", 0.2) * abs(avg_eff_change * 10)

        # 3. Ökonomische Rationalität basierend auf Schattenpreisen
        economic_rationality_reward = 0.0
        resource_cost = 0.0
        last_consumption = self._get_last_step_consumption()
        for resource, amount in last_consumption.items():
            price = self.shadow_prices.get(resource, 0.0)
            resource_cost += amount * price

        revenue = 0.0
        last_production = self._get_last_step_production()
        for good, amount in last_production.items():
            price = self.shadow_prices.get(good, 0.0)
            revenue += amount * price

        economic_rationality_reward = revenue - resource_cost
        reward += cfg.get("economic_rationality_weight", 0.8) * economic_rationality_reward

        # 4. Resilienz (Wartungsstatus)
        maint_change = self.maintenance_status - self._prev_maintenance_status
        if maint_change > 0:
            reward += cfg.get("maint_increase_weight", 0.4) * (maint_change * 10)
        elif maint_change < 0:
            reward -= cfg.get("maint_decrease_penalty", 0.6) * abs(maint_change * 10) # Stärkere Strafe für Wartungsabnahme

        if self.maintenance_status < cfg.get("low_maint_threshold", 0.3):
            reward -= cfg.get("low_maint_penalty", 1.0) # Feste Strafe für sehr niedrige Wartung

        # Update "Previous" Werte für den nächsten Schritt
        self._prev_tech_level = self.tech_level
        self._prev_avg_efficiency = current_avg_eff
        self._prev_maintenance_status = self.maintenance_status

        # Reward Clipping zur Stabilisierung (optional, aber oft nützlich)
        reward_clip_min = cfg.get("reward_clip_min", -5.0)
        reward_clip_max = cfg.get("reward_clip_max", 5.0)
        reward = np.clip(reward, reward_clip_min, reward_clip_max)

        self.logger.debug(
            f"Producer {self.unique_id} RL Reward: {reward:.3f} (Economic: {economic_rationality_reward:.3f})"
        )
        return float(reward)


    # --- Berichterstattung ---

    def generate_report(self) -> Dict[str, Any]:
        """Erstellt einen umfassenden Bericht über den Zustand des Producers."""
        report = {
            "step": self.model.current_step,
            "producer_id": self.unique_id,
            "region_id": self.region_id,
            "total_output": self.total_output_this_step,
            "capacity_utilization": self.capacity_utilization,
            "productive_capacity": self.productive_capacity,
            "tech_level": self.tech_level,
            "maintenance_status": self.maintenance_status,
            "resource_stock": dict(self.resource_stock),
            "output_stock": dict(self.output_stock),
            "production_targets": dict(self.production_target),
            "research_progress": dict(self.research_progress),
            "average_efficiency": self.calculate_average_efficiency(),
            "line_details": [vars(line).copy() for line in self.production_lines]  # Gibt Details pro Linie ohne Side-Effects
        }
        # Entferne komplexe Objekte wie Deques aus den Liniendetails für einfachere Serialisierung
        for line_detail in report["line_details"]:
            line_detail.pop('output_history', None)
            # line_detail.pop('efficiency_history', None) # Kann drin bleiben, wenn benötigt
        return report

    def update_and_get_task_queue_length(self) -> int:
        """
        Calculates the task queue length based on outstanding production targets.
        This is the sum of differences between target_output and actual_output for all lines
        where target > actual.
        """
        queue_length = 0.0 # Use float for summation
        for line in self.production_lines:
            if line.target_output > line.actual_output:
                # Consider queue length as the number of "units of work" based on some base unit,
                # or simply the sum of target shortfalls.
                # For simplicity, let's sum the shortfalls.
                queue_length += (line.target_output - line.actual_output)
        self.task_queue_length = int(round(queue_length)) # Ensure it's an integer
        return self.task_queue_length

    def _get_last_step_consumption(self) -> Dict[str, float]:
        """Return resources consumed in the previous step."""
        if hasattr(self, "_last_consumption_dict") and self._last_consumption_dict:
            return self._last_consumption_dict
        consumption = defaultdict(float)
        for resource, hist in self.resource_consumption_history.items():
            if hist:
                consumption[resource] = hist[-1]
        return dict(consumption)

    def _get_last_step_production(self) -> Dict[str, float]:
        """Return production amounts from the last step per good."""
        return getattr(self, "_last_production_dict", {})

    # --- Hilfsmethoden ---

    def update_capacity_allocation(self) -> None:
        """Aktualisiert die Kapazitätsanteile der Produktionslinien."""
        # Diese Methode wird jetzt zentraler genutzt
        active_lines = [line for line in self.production_lines if line.is_active]
        if not active_lines: return

        # Gewichtung (Prio * BasisEffizienz?)
        total_weight = sum(line.priority * line.base_efficiency for line in active_lines)

        if total_weight <= 0:
            # Gleichverteilung
            share = self.productive_capacity / len(active_lines)
            for line in active_lines:
                line.capacity_share = min(share, line.max_capacity or float('inf'))
        else:
            # Gewichtete Verteilung
            # Diese Logik könnte komplexer sein, um max_capacity besser zu handhaben
            # Hier einfache proportionale Zuweisung mit Capping
            temp_shares = {}
            for line in active_lines:
                weight = (line.priority * line.base_efficiency) / total_weight
                share = weight * self.productive_capacity
                temp_shares[line.name] = min(share, line.max_capacity or float('inf'))

            # Normalisiere ggf. neu, falls Capping Summe < total Capacity ergibt
            current_sum = sum(temp_shares.values())
            if current_sum < self.productive_capacity * 0.999:  # Kleine Toleranz
                remaining = self.productive_capacity - current_sum
                # Betrachte nur Linien, die noch Kapazität aufnehmen können
                adjustable_lines = [
                    line
                    for line in active_lines
                    if line.max_capacity is None
                    or temp_shares[line.name] < line.max_capacity - 1e-9
                ]

                if adjustable_lines:
                    total_adj_weight = sum(
                        line.priority * line.base_efficiency for line in adjustable_lines
                    )
                    for line in adjustable_lines:
                        weight = line.priority * line.base_efficiency
                        additional = remaining * (weight / total_adj_weight)
                        if line.max_capacity is not None:
                            available = line.max_capacity - temp_shares[line.name]
                            additional = min(additional, available)
                        temp_shares[line.name] += additional

            for line in active_lines:
                line.capacity_share = temp_shares.get(line.name, 0.0)

        self.logger.debug(f"Producer {self.unique_id}: Kapazitätsanteile aktualisiert.")


    def calculate_average_efficiency(self) -> float:
        """Berechnet die gewichtete durchschnittliche *effektive* Effizienz."""
        active_lines = [line for line in self.production_lines if line.is_active and line.capacity_share > 0]
        if not active_lines: return 0.0

        total_weighted_eff = sum(line.effective_efficiency * line.capacity_share for line in active_lines)
        total_capacity_share = sum(line.capacity_share for line in active_lines)

        return total_weighted_eff / total_capacity_share if total_capacity_share > 0 else 0.0