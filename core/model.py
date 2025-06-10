# Impaler/core/model.py
"""
Haupt-Simulationsmodell für das Impaler Framework.

Orchestriert Agenten (Producer, Consumer, etc.), VSM-Systeme (1-5)
und den Simulationsablauf über einen StageManager. Nutzt eine Pydantic-basierte
Konfiguration für Flexibilität und Validierung.

**Änderung:** Distribution-Stage und zugehörige Methoden entfernt, da
             diese Logik durch die VSM-Systeme (S2/S3) abgedeckt wird/werden sollte.
"""

import logging
import inspect
import os
import random
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Tuple,
    Set,
    Union,
    Callable,
    Type,
    DefaultDict,
)

# Parallelisierung (optional aber empfohlen)
try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    # Fallback zu sequenzieller Ausführung, wenn joblib fehlt

# Pydantic für Konfiguration und Validierung
from pydantic import ValidationError

# Kernkomponenten
# Annahme: Pfade sind relativ zum Projekt-Root oder PYTHONPATH ist korrekt gesetzt
from .stagemanager import StageManager
from .crisismanager import CrisisManager
from .datacollector import DataCollector
from .state_estimator import StateEstimator
from .config import (
    SimulationConfig,
    create_default_config,
    AgentPopulationConfig,
    SpecificAgentConfig,
)  # Import Pydantic Config

# Agenten
# Beispielhafte Importpfade - ggf. anpassen!
from ..agents import ProducerAgent, ConsumerAgent, ResourceAgent

# from ..agents.infrastructure import InfrastructureAgent # Falls vorhanden

# VSM Subsysteme
# HINWEIS: Die tatsächlichen Importpfade können abweichen!
try:
    from ..vsm import (
        System2Coordinator,
        System3Manager,
        System4Planner,
    )
    from ..vsm.system3 import RegionalSystem3Manager
    from ..governance import GovernmentAgent

    # System1 und System5 haben derzeit keine vollwertigen Implementierungen.
    class System1Operator:
        def __init__(self, *args, **kwargs):
            pass

        def step_stage(self, *args, **kwargs):
            pass

    System5Policy = GovernmentAgent

except ImportError as e:
    logging.warning(
        f"VSM-Module konnten nicht vollständig importiert werden: {e}. VSM-Funktionalität ist möglicherweise eingeschränkt."
    )

    # Definiere Dummy-Klassen, um Laufzeitfehler zu vermeiden, wenn VSM deaktiviert ist
    class DummyVSMSystem:
        def __init__(self, *args, **kwargs):
            pass

        def step_stage(self, *args, **kwargs):
            pass

    System1Operator = System2Coordinator = System3Manager = System4Planner = (
        DummyVSMSystem
    )
    System5Policy = DummyVSMSystem

# Datenstrukturen für Kommunikation
# from ..vsm.communication_contracts import OperationalReportS3, StrategicDirectiveS4, ConflictResolutionDirectiveS2, SystemFeedback # Falls ausgelagert

# Utils
from ..utils.math_utils import gini_coefficient

# from ..utils.geo_utils import region_distance_matrix # Falls benötigt und vorhanden

# --- Globale Konstanten / Mappings ---

# Sichereres Mapping von Klassen-Namen (String) zu tatsächlichen Klassen
# Fügen Sie hier alle Agententypen hinzu, die über die Konfiguration erstellt werden können
AGENT_CLASS_MAP: Dict[str, Type] = {
    "ProducerAgent": ProducerAgent,
    "ConsumerAgent": ConsumerAgent,
    "ResourceAgent": ResourceAgent,
    # "InfrastructureAgent": InfrastructureAgent, # Falls vorhanden
    # Füge ggf. CommunityConsumerAgent hinzu, falls es direkt erstellt werden soll
    # "CommunityConsumerAgent": CommunityConsumerAgent
}

# --- Hauptmodellklasse ---


class EconomicModel:
    """
    Zentrale Orchestrierungsklasse für die Impaler-Simulation.

    Verwaltet Agenten, VSM-Systeme, den Simulationsablauf (über Stages),
    Krisen und Datensammlung basierend auf einer validierten Konfiguration.
    """

    # --- Typ-Annotationen ---
    config: SimulationConfig
    logger: logging.Logger
    current_step: int
    start_time: str
    parallel_execution: bool
    data_collector: DataCollector
    stage_manager: StageManager
    crisis_manager: CrisisManager
    state_estimator: Optional[StateEstimator]
    mu_t: Optional[np.ndarray]
    P_t: Optional[np.ndarray]
    u_t: Optional[np.ndarray]
    producers: List[ProducerAgent]
    consumers: List[ConsumerAgent]
    resource_agent: Optional[ResourceAgent]
    infrastructure_agents: List[Any]
    agents_by_id: Dict[str, Any]
    producers_by_region: Dict[Union[int, str], List[ProducerAgent]]
    consumers_by_region: Dict[Union[int, str], List[ConsumerAgent]]
    infra_by_region: Dict[Union[int, str], List[Any]]
    system5policy: Optional[System5Policy]
    system4planner: Optional[System4Planner]
    system3manager: Optional[System3Manager]
    system2coordinator: Optional[System2Coordinator]
    system1operator: Optional[System1Operator]
    regional_managers: Dict[Union[int, str], "RegionalSystem3Manager"]
    transport_network: Dict[str, Dict[str, Dict[str, float]]]
    aggregated_consumer_demand: Dict[str, float]
    regional_consumer_demand: Dict[Union[int, str], Dict[str, float]]
    plan_execution_gap: Dict[str, float]
    historical_plan_accuracy: deque
    environmental_impacts: Dict[str, Any]
    welfare_metrics: Dict[str, Any]
    detailed_metrics: Dict[str, Any]
    total_production_this_step: DefaultDict[
        str, float
    ]  # Hinzugefügt für stage_production_execution

    def __init__(self, config: Union[dict, SimulationConfig, str]):
        """
        Initialisiert das Wirtschaftsmodell.

        Args:
            config: Kann ein Dictionary, ein SimulationConfig-Objekt oder
                    ein Pfad zu einer JSON-Konfigurationsdatei sein.
        """
        self.current_step: int = 0
        self.start_time: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Konfiguration laden und validieren
        self.config: SimulationConfig
        try:
            if isinstance(config, SimulationConfig):
                self.config = config
            elif isinstance(config, str):  # Pfad zur JSON-Datei
                self.config = SimulationConfig.from_json_file(config)
            elif isinstance(config, dict):
                self.config = SimulationConfig.from_dict(config)
            else:
                # Logge hier, bevor Logging konfiguriert wird
                logging.warning(
                    "Keine valide Konfiguration übergeben, verwende Standardkonfiguration."
                )
                self.config = create_default_config()
        except (ValidationError, FileNotFoundError, ValueError) as e:
            # Logge den Fehler und brich ab oder fahre mit Defaults fort
            logging.basicConfig(
                level=logging.ERROR
            )  # Stelle sicher, dass Logging initialisiert ist
            logger_init = logging.getLogger("EconomicModel.init")
            logger_init.error(
                f"!!! Kritischer Konfigurationsfehler: {e}. Simulation kann nicht starten.",
                exc_info=True,
            )
            raise ValueError("Konfigurationsfehler beim Modellstart.") from e

        # 2. Logging einrichten (basierend auf validierter Config)
        self._setup_logging()
        self.logger.info(
            f"Initialisiere EconomicModel mit Konfiguration: {self.config.simulation_name}"
        )

        # 3. Zufallsgenerator initialisieren
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        self.logger.info(
            f"Zufallsgenerator initialisiert mit Seed: {self.config.random_seed}"
        )

        # 4. Grundparameter setzen
        self.parallel_execution: bool = (
            self.config.parallel_execution and JOBLIB_AVAILABLE
        )
        if self.config.parallel_execution and not JOBLIB_AVAILABLE:
            self.logger.warning("Joblib nicht gefunden. Parallelisierung deaktiviert.")

        # 5. Kernkomponenten initialisieren
        self.data_collector: DataCollector = DataCollector(
            self.config.logging_config.dict()
        )  # Nutze Logging-Config für DC
        self.stage_manager: StageManager = StageManager(self)
        self.crisis_manager: CrisisManager = CrisisManager(
            self
        )  # Nimmt das Model als Argument
        self.state_estimator = None
        if self.config.state_estimator_config.enabled:
            self.state_estimator = StateEstimator(self)

        self.mu_t = None
        self.P_t = None
        self.u_t = np.zeros(1)

        # 6. Agentenlisten und Lookups initialisieren
        self.producers: List[ProducerAgent] = []
        self.consumers: List[ConsumerAgent] = []
        self.resource_agent: Optional[ResourceAgent] = (
            None  # Wird in _create_agents erstellt
        )
        self.infrastructure_agents: List[Any] = (
            []
        )  # Typ spezifischer machen, wenn Klasse bekannt
        # Lookups
        self.agents_by_id: Dict[str, Any] = {}
        self.producers_by_region: Dict[Union[int, str], List[ProducerAgent]] = (
            defaultdict(list)
        )
        self.consumers_by_region: Dict[Union[int, str], List[ConsumerAgent]] = (
            defaultdict(list)
        )
        self.infra_by_region: Dict[Union[int, str], List[Any]] = defaultdict(list)

        # 7. VSM-Systeme initialisieren (brauchen Referenz auf das Modell)
        self._init_vsm_systems()

        # 8. Agenten erstellen (nutzt die neue Config-Struktur)
        self._create_agents()

        # 9. Netzwerke initialisieren (Transport etc.)
        self._init_networks()

        # 10. VSM-Systeme verbinden (Agenten zuordnen etc.)
        self._connect_vsm_systems()

        # 11. Standard-Simulationsphasen registrieren
        self._register_default_stages()

        # 12. Zusätzliche Zustandsvariablen initialisieren
        self.aggregated_consumer_demand: Dict[str, float] = defaultdict(float)
        self.regional_consumer_demand: Dict[Union[int, str], Dict[str, float]] = (
            defaultdict(lambda: defaultdict(float))
        )
        self.plan_execution_gap: Dict[str, float] = {}  # Diskrepanz pro Gut
        self.historical_plan_accuracy: deque = deque(maxlen=20)  # Letzte 20 Schritte
        self.environmental_impacts: Dict[str, Any] = {}  # Wird in Stage berechnet
        self.welfare_metrics: Dict[str, Any] = {}  # Wird in Stage berechnet
        self.detailed_metrics: Dict[str, Any] = {}  # Letzter Satz an Metriken
        self.total_production_this_step = defaultdict(
            float
        )  # Hinzugefügt für stage_production_execution

        self.logger.info("EconomicModel-Initialisierung abgeschlossen.")

    def _setup_logging(self) -> None:
        """Konfiguriert das Logging-System basierend auf der Config."""
        log_cfg = self.config.logging_config
        log_level = getattr(logging, log_cfg.log_level.upper(), logging.INFO)

        # Root-Logger konfigurieren
        logging.basicConfig(
            level=log_level, format=log_cfg.log_format, force=True
        )  # Überschreibt existierende Konfiguration

        root_logger = logging.getLogger()

        # Stelle sicher, dass Handler nicht dupliziert werden, wenn __init__ mehrfach aufgerufen wird
        # (sollte nicht passieren, aber sicherheitshalber)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)  # Entferne alle vorherigen Handler

        # Console Handler
        if log_cfg.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_cfg.log_format))
            console_handler.setLevel(log_level)
            root_logger.addHandler(console_handler)

        # File Handler
        if log_cfg.log_to_file:
            log_dir = os.path.dirname(os.path.abspath(log_cfg.log_file_path))
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            try:
                file_handler = logging.FileHandler(
                    log_cfg.log_file_path, mode="a"
                )  # Append-Modus
                file_handler.setFormatter(logging.Formatter(log_cfg.log_format))
                file_handler.setLevel(log_level)
                root_logger.addHandler(file_handler)
            except Exception as e:
                root_logger.error(
                    f"Fehler beim Einrichten des File-Loggings nach {log_cfg.log_file_path}: {e}"
                )

        # Spezifischen Logger für diese Klasse holen
        self.logger = logging.getLogger("EconomicModel")

    def _init_vsm_systems(self) -> None:
        """Initialisiert die VSM-Subsysteme."""
        self.logger.info("Initialisiere VSM-Subsysteme...")
        # ResourceAgent wird in _create_agents als spezifischer Agent behandelt
        # Reihenfolge wichtig? Erstmal alle erstellen.
        self.system5policy: Optional[System5Policy] = (
            System5Policy("government", self) if self.config.vsm_on else None
        )
        self.system4planner: Optional[System4Planner] = (
            System4Planner("system4planner", self) if self.config.vsm_on else None
        )
        self.system3manager: Optional[System3Manager] = (
            System3Manager(self) if self.config.vsm_on else None
        )
        self.system2coordinator: Optional[System2Coordinator] = (
            System2Coordinator(self) if self.config.vsm_on else None
        )
        self.system1operator: Optional[System1Operator] = (
            System1Operator(self) if self.config.vsm_on else None
        )
        # RegionalSystem3Manager werden in _connect_vsm_systems erstellt
        self.regional_managers: Dict[Union[int, str], RegionalSystem3Manager] = {}

    def _resolve_param(self, param_value: Any) -> Any:
        """Hilfsfunktion: Löst Parameter auf (zufällig aus Range/Liste oder direkt)."""
        if isinstance(param_value, (tuple, list)) and len(param_value) == 2:
            # Annahme: (min, max) -> random.uniform oder randint
            min_val, max_val = param_value
            if isinstance(min_val, int) and isinstance(max_val, int):
                return random.randint(min_val, max_val)
            elif isinstance(min_val, (float, int)) and isinstance(
                max_val, (float, int)
            ):
                # Konvertiere zu float für uniform
                return random.uniform(float(min_val), float(max_val))
        elif (
            isinstance(param_value, list)
            and len(param_value) > 1
            and not isinstance(param_value[0], dict)
        ):
            # Wenn es eine Liste von Optionen ist (und keine Liste von Dicts, z.B. für production_lines)
            return random.choice(param_value)
        # Ansonsten den Wert direkt verwenden (kann auch ein Dict oder eine Liste sein!)
        return param_value

    def _create_agents(self) -> None:
        """Erstellt alle Agenten basierend auf der Konfiguration."""
        self.logger.info("Erstelle Agenten...")
        # Verwende die definierten Regionen-Namen/-IDs aus der Config
        region_keys = self.config.regional_config.regions  # Dies sind die Namen/IDs
        num_regions = len(region_keys)
        if num_regions == 0:
            self.logger.warning(
                "Keine Regionen in der Konfiguration definiert. Agentenerstellung nicht möglich."
            )
            return

        created_ids = set()
        total_agents_created = 0

        # Support simple 'agent_counts' override used in tests
        if not self.config.agent_populations and getattr(
            self.config, "agent_counts", None
        ):
            ac = getattr(self.config, "agent_counts")
            prod_count = ac.get("producers", 0)
            cons_count = ac.get("consumers", 0)
            regions = ac.get("regions", num_regions)
            if regions > num_regions:
                self.logger.warning(
                    "agent_counts definiert mehr Regionen als in regional_config vorhanden."
                )
                regions = num_regions
            if prod_count:
                self.config.agent_populations["default_producers"] = (
                    AgentPopulationConfig(
                        agent_class="ProducerAgent",
                        count=prod_count,
                        params={"production_lines": []},
                        id_prefix="prod",
                    )
                )
            if cons_count:
                self.config.agent_populations["default_consumers"] = (
                    AgentPopulationConfig(
                        agent_class="ConsumerAgent",
                        count=cons_count,
                        params={"base_needs": {self.config.goods[0]: 1.0}},
                        id_prefix="cons",
                    )
                )

        # ResourceAgent immer erstellen (kann als specific_agent konfiguriert werden, falls nötig)
        if "ResourceAgent" in AGENT_CLASS_MAP and self.resource_agent is None:
            # Suche nach spezifischer Konfiguration
            resource_config = next(
                (
                    cfg
                    for cfg in self.config.specific_agents
                    if cfg.agent_class == "ResourceAgent"
                ),
                None,
            )
            if resource_config:
                if resource_config.unique_id in created_ids:
                    self.logger.warning(
                        f"ResourceAgent ID '{resource_config.unique_id}' bereits vergeben."
                    )
                else:
                    try:
                        self.resource_agent = AGENT_CLASS_MAP["ResourceAgent"](
                            unique_id=resource_config.unique_id,
                            model=self,
                            **resource_config.params,  # Erwartet, dass Parameter hier definiert sind
                        )
                        self._register_agent(
                            self.resource_agent, AGENT_CLASS_MAP["ResourceAgent"]
                        )
                        created_ids.add(resource_config.unique_id)
                        total_agents_created += 1
                    except Exception as e:
                        self.logger.error(
                            f"Fehler beim Erstellen des spezifischen ResourceAgent '{resource_config.unique_id}': {e}",
                            exc_info=True,
                        )
            else:
                # Erstelle Standard ResourceAgent
                try:
                    res_id = "resource_agent_0"
                    if res_id in created_ids:
                        res_id = f"resource_agent_{random.randint(1000, 9999)}"  # Fallback ID
                    self.resource_agent = AGENT_CLASS_MAP["ResourceAgent"](
                        unique_id=res_id, model=self
                    )
                    self._register_agent(
                        self.resource_agent, AGENT_CLASS_MAP["ResourceAgent"]
                    )
                    created_ids.add(res_id)
                    total_agents_created += 1
                except Exception as e:
                    self.logger.error(
                        f"Fehler beim Erstellen des Standard ResourceAgent: {e}",
                        exc_info=True,
                    )

        # 1. Agenten aus Populationen erstellen
        for pop_name, pop_config in self.config.agent_populations.items():
            agent_cls_name = pop_config.agent_class
            if agent_cls_name not in AGENT_CLASS_MAP:
                self.logger.error(
                    f"Unbekannte Agentenklasse '{agent_cls_name}' in Population '{pop_name}'. Überspringe."
                )
                continue
            AgentClass = AGENT_CLASS_MAP[agent_cls_name]

            # Bestimme Regionenverteilung für diese Population
            region_assignments = self._get_region_assignments(
                pop_config.count, region_keys, pop_config.region_distribution
            )

            # Erstelle Agenten der Population
            created_in_pop = 0
            for i in range(pop_config.count):
                unique_id = f"{pop_config.id_prefix or pop_name}_{i}"
                if unique_id in created_ids:
                    # Versuche alternative ID
                    alt_id_counter = 1
                    while f"{unique_id}_{alt_id_counter}" in created_ids:
                        alt_id_counter += 1
                    unique_id = f"{unique_id}_{alt_id_counter}"
                    self.logger.warning(
                        f"Generierte ID für Population '{pop_name}' war Duplikat. Verwende stattdessen '{unique_id}'."
                    )

                region_id_assigned = region_assignments[
                    i
                ]  # Bekomme die zugewiesene Regions-ID/-Name

                # Parameter auflösen (Defaults + Randomisierung)
                agent_params = {}
                try:
                    for key, value in pop_config.params.items():
                        agent_params[key] = self._resolve_param(value)
                except Exception as e:
                    self.logger.error(
                        f"Fehler beim Auflösen der Parameter für Population '{pop_name}': {e}",
                        exc_info=True,
                    )
                    continue  # Nächsten Agenten versuchen

                # Agent instanziieren
                try:
                    # Stelle sicher, dass region_id übergeben wird, falls nicht in params
                    # und die __init__ des Agenten dieses Argument unterstützt
                    if "region_id" not in agent_params:
                        sig = inspect.signature(AgentClass.__init__)
                        if "region_id" in sig.parameters:
                            agent_params["region_id"] = region_id_assigned

                    # Provide a simple default production line if none was given
                    if AgentClass == ProducerAgent:
                        if (
                            "production_lines" not in agent_params
                            or not agent_params.get("production_lines")
                        ):
                            default_good = (
                                self.config.goods[0] if self.config.goods else "GoodA"
                            )
                            agent_params["production_lines"] = [
                                {"output_good": default_good, "input_requirements": []}
                            ]

                    agent = AgentClass(unique_id=unique_id, model=self, **agent_params)
                    self._register_agent(agent, AgentClass)
                    created_ids.add(unique_id)
                    created_in_pop += 1
                    total_agents_created += 1
                except Exception as e:
                    self.logger.error(
                        f"Fehler beim Instanziieren von Agent '{unique_id}' ({AgentClass.__name__}) aus Population '{pop_name}': {e}",
                        exc_info=True,
                    )
            self.logger.info(
                f"{created_in_pop}/{pop_config.count} Agenten aus Population '{pop_name}' ({AgentClass.__name__}) erstellt."
            )

        # 2. Spezifische Agenten erstellen
        for agent_config in self.config.specific_agents:
            agent_cls_name = agent_config.agent_class
            unique_id = agent_config.unique_id

            if unique_id in created_ids:
                # Spezifische Agenten sollen nicht überschrieben werden, wenn ID schon existiert
                if (
                    agent_cls_name == "ResourceAgent"
                    and self.resource_agent
                    and self.resource_agent.unique_id == unique_id
                ):
                    self.logger.warning(
                        f"Spezifischer ResourceAgent '{unique_id}' überschreibt den Standard-Agenten."
                    )
                    # Führe trotzdem Erstellung durch, um ggf. Parameter zu aktualisieren
                else:
                    self.logger.warning(
                        f"Spezifische Agent ID '{unique_id}' ist bereits vergeben. Überspringe."
                    )
                    continue

            if agent_cls_name not in AGENT_CLASS_MAP:
                self.logger.error(
                    f"Unbekannte Agentenklasse '{agent_cls_name}' für spezifischen Agenten '{unique_id}'. Überspringe."
                )
                continue
            AgentClass = AGENT_CLASS_MAP[agent_cls_name]

            # Parameter auflösen (erlaubt weiterhin Ranges auch für spezifische Agenten)
            agent_params = {}
            try:
                for key, value in agent_config.params.items():
                    agent_params[key] = self._resolve_param(value)
            except Exception as e:
                self.logger.error(
                    f"Fehler beim Auflösen der Parameter für spezifischen Agenten '{unique_id}': {e}",
                    exc_info=True,
                )
                continue

            # Region ID muss in params sein, wenn der Agent eine region_id erwartet
            if "region_id" not in agent_params and hasattr(
                AgentClass(unique_id="dummy", model=self), "region_id"
            ):
                self.logger.error(
                    f"Spezifischer Agent '{unique_id}' ({AgentClass.__name__}) hat keine 'region_id' in params. Überspringe."
                )
                continue

            # Agent instanziieren
            try:
                # Wenn es der ResourceAgent ist, überschreibe die Instanzvariable
                if AgentClass == ResourceAgent:
                    self.resource_agent = AgentClass(
                        unique_id=unique_id, model=self, **agent_params
                    )
                    agent = self.resource_agent
                else:
                    agent = AgentClass(unique_id=unique_id, model=self, **agent_params)

                self._register_agent(agent, AgentClass)
                created_ids.add(unique_id)
                total_agents_created += 1
                self.logger.info(
                    f"Spezifischen Agenten '{unique_id}' ({AgentClass.__name__}) erstellt."
                )
            except Exception as e:
                self.logger.error(
                    f"Fehler beim Erstellen des spezifischen Agenten '{unique_id}': {e}",
                    exc_info=True,
                )

        self.logger.info(
            f"Agentenerstellung abgeschlossen. Gesamt: {total_agents_created} Agenten."
        )
        self.logger.info(
            f"Verteilung: {len(self.producers)} Producer, {len(self.consumers)} Consumer, {len(self.infrastructure_agents)} Infra, {'1 ResourceAgent' if self.resource_agent else 'Kein ResourceAgent'}."
        )

    def _get_region_assignments(
        self,
        count: int,
        region_keys: List[Union[int, str]],
        distribution: Optional[Dict[Union[int, str], Union[int, float]]],
    ) -> List[Union[int, str]]:
        """Weist Agenten basierend auf der Verteilungskonfiguration Regionen zu."""
        assignments: List[Union[int, str]] = []
        num_regions = len(region_keys)

        if distribution:
            total_assigned = 0
            assigned_counts: Dict[Union[int, str], int] = {r: 0 for r in region_keys}
            region_map = {
                str(k): k for k in region_keys
            }  # Mapping von String-Key zu Original-Key

            # Verarbeite Verteilung (kann int oder str als Key haben)
            valid_distribution = {}
            for key, value in distribution.items():
                mapped_key = region_map.get(str(key))  # Versuche Key zu mappen
                if mapped_key is not None:
                    valid_distribution[mapped_key] = value
                else:
                    try:  # Versuche als int, falls regions als int Indizes verwendet werden
                        int_key = int(key)
                        if int_key < num_regions:
                            valid_distribution[region_keys[int_key]] = value
                        else:
                            self.logger.warning(
                                f"Ungültiger Region Key '{key}' (als Index interpretiert) in region_distribution ignoriert."
                            )
                    except (ValueError, IndexError):
                        self.logger.warning(
                            f"Ungültiger Region Key '{key}' in region_distribution ignoriert."
                        )

            # Absolute Zahlen zuerst
            abs_dist = {
                k: v for k, v in valid_distribution.items() if isinstance(v, int)
            }
            for region_key, num in abs_dist.items():
                num_to_assign = min(
                    num, count - total_assigned
                )  # Nicht mehr zuweisen als übrig
                if num_to_assign > 0:
                    assignments.extend([region_key] * num_to_assign)
                    assigned_counts[region_key] = num_to_assign
                    total_assigned += num_to_assign

            # Relative Anteile für den Rest
            rel_dist = {
                k: v for k, v in valid_distribution.items() if isinstance(v, float)
            }
            remaining_count = count - total_assigned
            if remaining_count > 0 and rel_dist:
                total_fraction = sum(rel_dist.values())
                if total_fraction > 0:
                    cumulative_assigned_frac = 0
                    # Sortiere, um Rundung deterministischer zu machen
                    sorted_rel_items = sorted(
                        rel_dist.items(), key=lambda item: item[0]
                    )
                    for i, (region_key, fraction) in enumerate(sorted_rel_items):
                        # Berechne Anteil, runde kaufmännisch, aber stelle sicher, dass Summe stimmt
                        share = fraction / total_fraction
                        num_this_region = round(share * remaining_count)

                        # Korrigiere Rundung am letzten Element, um Summe zu erzwingen
                        if i == len(rel_dist) - 1:
                            num_this_region = remaining_count - cumulative_assigned_frac

                        num_to_assign = min(
                            num_this_region, remaining_count - cumulative_assigned_frac
                        )
                        if num_to_assign > 0:
                            assignments.extend([region_key] * num_to_assign)
                            assigned_counts[region_key] += num_to_assign
                            total_assigned += num_to_assign
                            cumulative_assigned_frac += num_to_assign

            # Fülle den Rest ggf. per Round-Robin auf oder weise zufällig zu
            while total_assigned < count:
                region_idx = total_assigned % num_regions
                region_key = region_keys[region_idx]
                assignments.append(region_key)
                assigned_counts[region_key] += 1
                total_assigned += 1

            # Mische die Zuweisungen, um Clusterbildung zu vermeiden
            random.shuffle(assignments)
            # Stelle sicher, dass die Länge stimmt (durch Rundung etc.)
            assignments = assignments[:count]
            if len(assignments) < count:  # Falls durch Rundung zu wenige
                assignments.extend(
                    [
                        random.choice(region_keys)
                        for _ in range(count - len(assignments))
                    ]
                )

            self.logger.debug(
                f"Regionale Verteilung für {count} Agenten: {assigned_counts}"
            )

        else:
            # Default: Round Robin über die region_keys
            assignments = [region_keys[i % num_regions] for i in range(count)]
            self.logger.debug(f"Regionale Verteilung für {count} Agenten: Round Robin.")

        return assignments

    def _register_agent(self, agent: Any, agent_class: Type) -> None:
        """Registriert einen Agenten in den Modell-Listen und Lookups."""
        if agent.unique_id in self.agents_by_id:
            self.logger.warning(
                f"Agent mit ID '{agent.unique_id}' existiert bereits. Wird überschrieben."
            )

        self.agents_by_id[agent.unique_id] = agent
        region_id = getattr(
            agent, "region_id", None
        )  # Nicht jeder Agent hat eine Region (z.B. ResourceAgent)

        if agent_class == ProducerAgent:
            self.producers.append(agent)
            if region_id is not None:
                self.producers_by_region[region_id].append(agent)
        elif agent_class == ConsumerAgent:
            self.consumers.append(agent)
            if region_id is not None:
                self.consumers_by_region[region_id].append(agent)
        # elif agent_class == InfrastructureAgent: # Falls vorhanden
        #     self.infrastructure_agents.append(agent)
        #     if region_id is not None:
        #         self.infra_by_region[region_id].append(agent)
        elif agent_class == ResourceAgent:
            # Prüfe, ob schon ein ResourceAgent registriert wurde
            current_resource_agent = self.agents_by_id.get(
                getattr(self.resource_agent, "unique_id", None)
            )
            if (
                current_resource_agent is not None
                and current_resource_agent is not agent
            ):
                self.logger.warning(
                    f"Überschreibe existierenden ResourceAgent '{current_resource_agent.unique_id}' mit '{agent.unique_id}'."
                )
                # Entferne alten Agenten aus Lookup
                if current_resource_agent.unique_id in self.agents_by_id:
                    del self.agents_by_id[current_resource_agent.unique_id]
            self.resource_agent = agent
        else:
            self.logger.debug(
                f"Agent '{agent.unique_id}' vom Typ {agent_class.__name__} keiner Hauptliste zugeordnet."
            )

    def _init_networks(self) -> None:
        """Initialisiert Netzwerke (Transport, etc.) basierend auf der Konfiguration."""
        self.logger.info("Initialisiere Netzwerke...")
        # Transportnetzwerk aus regionaler Konfiguration erstellen
        self.transport_network: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        regions = self.config.regional_config.regions
        capacity_config = self.config.regional_config.transport_capacity
        costs_config = self.config.regional_config.transport_costs
        default_capacity = 100.0  # Fallback-Kapazität

        for r1_name in regions:
            for r2_name in regions:
                if r1_name != r2_name:
                    # Kapazität
                    capacity = capacity_config.get(r1_name, {}).get(
                        r2_name, default_capacity
                    )
                    # Kosten (pro Gut/Ressource)
                    costs = costs_config.get(r1_name, {}).get(r2_name, {})

                    self.transport_network[r1_name][r2_name] = {
                        "capacity": capacity,
                        "costs": costs,  # Dict {good/resource: cost_factor}
                        "current_flow": 0.0,  # Wird in jedem Step zurückgesetzt
                        "congestion": 0.0,  # Wird berechnet
                    }
        self.logger.debug("Transportnetzwerk initialisiert.")
        # TODO: Andere Netzwerke (Innovation, Ressourcenabhängigkeit) hier initialisieren,
        # falls sie nicht dynamisch aus Agenten abgeleitet werden.

    def _connect_vsm_systems(self) -> None:
        """Verbindet VSM-Systeme und initialisiert regionale Manager."""
        if not self.config.vsm_on:
            self.logger.info("VSM ist deaktiviert. Überspringe Systemverbindung.")
            return

        self.logger.info("Verbinde VSM-Systeme und initialisiere Regionalmanager...")

        if not self.system3manager:
            self.logger.warning(
                "System3Manager nicht initialisiert, VSM-Verbindung nicht möglich."
            )
            return

        region_keys = self.config.regional_config.regions
        if not region_keys:
            self.logger.warning(
                "Keine Regionen definiert, VSM-Verbindung nicht möglich."
            )
            return

        # Delegiere die Initialisierung der RegionalSystem3Manager an System3
        self.system3manager.initialize_regions(
            producers_by_region=self.producers_by_region,
            consumers_by_region=self.consumers_by_region,
        )

        # Spiegle das Mapping im Model (vereinfachter Zugriff)
        self.regional_managers = self.system3manager.regional_managers

        if self.system4planner:
            setattr(self.system3manager, "system4planner", self.system4planner)

        self.logger.info("VSM-Systeme verbunden.")

    def _register_default_stages(self) -> None:
        """Registriert die Standard-Simulationsphasen mit Abhängigkeiten."""
        self.logger.info("Registriere Simulationsphasen...")
        stages_config = self.config.stages  # Nutze die Reihenfolge/Liste aus der Config

        # Definiere Stages mit Namen, Funktion, Parallelisierbarkeit und Standard-Reihenfolge/Abhängigkeiten
        # Die 'order' und 'depends_on' könnten auch aus der Config kommen, falls gewünscht.
        # Hier verwenden wir eine vordefinierte Logik.
        stage_definitions = [
            ("resource_regen", self.stage_resource_regen, True, 10, []),
            (
                "state_estimation",
                self.stage_state_estimation,
                False,
                15,
                ["resource_regen"],
            ),
            ("need_estimation", self.stage_need_estimation, True, 20, []),
            (
                "infrastructure_development",
                self.stage_infrastructure_development,
                True,
                30,
                [],
            ),
            (
                "system5_policy",
                self.stage_system5_policy,
                False,
                40,
                ["need_estimation", "infrastructure_development"],
            ),
            (
                "system4_strategic_planning",
                self.stage_system4_strategic_planning,
                False,
                50,
                ["system5_policy", "need_estimation", "state_estimation"],
            ),
            (
                "system4_tactical_planning",
                self.stage_system4_tactical_planning,
                False,
                60,
                ["system4_strategic_planning"],
            ),
            (
                "system3_aggregation",
                self.stage_system3_aggregation,
                False,
                70,
                ["system4_tactical_planning"],
            ),
            (
                "system2_coordination",
                self.stage_system2_coordination,
                False,
                80,
                ["system3_aggregation"],
            ),
            (
                "system3_feedback",
                self.stage_system3_feedback,
                False,
                90,
                ["system2_coordination"],
            ),
            (
                "system1_operations",
                self.stage_system1_operations,
                False,
                100,
                ["system3_feedback"],
            ),
            (
                "local_production_planning",
                self.stage_local_production_planning,
                True,
                110,
                ["system1_operations"],
            ),
            (
                "admm_update",
                self.stage_admm_update,
                False,
                120,
                ["local_production_planning"],
            ),
            (
                "production_execution",
                self.stage_production_execution,
                True,
                130,
                ["admm_update", "local_production_planning"],
            ),
            # ("distribution", self.stage_distribution, False, 140, ["production_execution"]), # ENTFERNT
            (
                "consumption",
                self.stage_consumption,
                True,
                150,
                ["production_execution"],
            ),  # Hängt jetzt von Produktion ab
            (
                "environmental_impact",
                self.stage_environmental_impact,
                False,
                160,
                ["production_execution"],
            ),
            (
                "technology_progress",
                self.stage_technology_progress,
                True,
                170,
                ["production_execution"],
            ),
            ("crisis_management", self.stage_crisis_management, False, 180, []),
            (
                "welfare_assessment",
                self.stage_welfare_assessment,
                False,
                190,
                ["consumption"],
            ),
            (
                "learning_adaptation",
                self.stage_learning_adaptation,
                False,
                200,
                ["welfare_assessment"],
            ),
            (
                "vsm_reconfiguration",
                self.stage_vsm_reconfiguration,
                False,
                210,
                ["learning_adaptation"],
            ),
            (
                "bookkeeping",
                self.stage_bookkeeping,
                False,
                220,
                ["vsm_reconfiguration", "welfare_assessment", "environmental_impact"],
            ),
            (
                "evaluate_governance_mode",
                self.stage_evaluate_governance_mode,
                False,
                230,
                ["bookkeeping"],
            ),
        ]

        # Registriere nur die Stages, die in der Konfiguration aufgeführt sind
        enabled_stage_names = set(stages_config)
        for name, func, parallel, order, deps in stage_definitions:
            if name in enabled_stage_names:
                try:
                    # Filtere Abhängigkeiten, die nicht aktiviert sind
                    enabled_deps = [d for d in deps if d in enabled_stage_names]
                    self.stage_manager.add_stage(
                        name=name,
                        func=func,
                        parallelizable=parallel,
                        order=order,
                        depends_on=enabled_deps,
                    )
                except ValueError as e:
                    self.logger.error(
                        f"Fehler beim Registrieren der Stage '{name}': {e}"
                    )
                    raise

        self.logger.info(
            f"Simulationsphasen registriert: {', '.join(s['name'] for s in self.stage_manager.list_stages())}"
        )

    def step(self) -> bool:
        """Führt einen kompletten Simulationsschritt aus."""
        if self.current_step >= self.config.simulation_steps:
            self.logger.info("Maximale Anzahl an Simulationsschritten erreicht.")
            return False  # Simulation beendet

        self.logger.info(
            f"--- Beginne Step {self.current_step + 1}/{self.config.simulation_steps} ---"
        )

        # Führe Stages aus
        try:
            self.stage_manager.run_stages(self.parallel_execution)
        except Exception as e:
            self.logger.critical(
                f"!!! Kritischer Fehler in Step {self.current_step + 1} während Stage Execution: {e}",
                exc_info=True,
            )
            # Hier könnte man entscheiden, die Simulation abzubrechen
            return False  # Simulation beendet aufgrund eines Fehlers

        self.logger.info(f"--- Beende Step {self.current_step + 1} ---")

        # Schrittzähler erhöhen
        self.current_step += 1
        return True  # Simulation läuft weiter

    # ---------------------------------------------------------------
    # Implementierung der Stage-Funktionen
    # ---------------------------------------------------------------
    # Diese Methoden enthalten die Logik für jede Phase. Sie rufen typischerweise
    # Methoden der VSM-Systeme oder Agenten auf.

    def stage_resource_regen(self):
        if self.resource_agent:
            if hasattr(self.resource_agent, "step_stage"):
                self.resource_agent.step_stage("resource_regen")
        else:
            self.logger.warning(
                "Kein ResourceAgent vorhanden für 'resource_regen' Stage."
            )

    def stage_state_estimation(self):
        """Stage: Führt die Zustandsschätzung für den aktuellen Schritt durch."""
        if not self.state_estimator:
            self.logger.debug("Zustandsschätzung übersprungen (deaktiviert).")
            return

        u_t_minus_1 = self.u_t if self.u_t is not None else np.zeros(1)

        if hasattr(self, "_ingest_raw_data_streams"):
            y_t = self._ingest_raw_data_streams()
        else:
            meas_dim = self.config.state_estimator_config.measurement_dimension
            y_t = np.random.rand(meas_dim)
            self.logger.warning(
                "Keine Methode '_ingest_raw_data_streams' gefunden. Verwende zufällige Messdaten für StateEstimator."
            )

        try:
            mu_t, P_t = self.state_estimator.estimate_state(u_t_minus_1, y_t)
            self.mu_t = mu_t
            self.P_t = P_t
        except Exception as e:
            self.logger.error(f"Fehler in der Stage 'state_estimation': {e}", exc_info=True)

    def stage_need_estimation(self):
        """Stage: Konsumenten aktualisieren Bedarfe, dann Aggregation."""

        # Funktion für einen einzelnen Konsumenten
        def _update_single_consumer_needs(consumer):
            if hasattr(consumer, "step_stage"):
                # Diese Stage sollte nur Bedarfe anpassen, nicht an Planer senden
                consumer.step_stage("need_adaptation")  # Eigene Stage im Consumer?
            # Gib Region mit zurück, damit die Aggregation regional erfolgen kann
            needs = getattr(consumer, "get_current_needs", lambda: {})()
            return consumer.region_id, needs

        # Führe parallel oder sequentiell aus
        if self.parallel_execution and len(self.consumers) > 10:
            results = Parallel(n_jobs=self.config.num_workers)(
                delayed(_update_single_consumer_needs)(c) for c in self.consumers
            )
            # results ist jetzt eine Liste von Needs-Dicts
            self._aggregate_consumer_demands_from_results(results)
        else:
            # Sequentiell Needs holen und aggregieren
            all_needs = [_update_single_consumer_needs(c) for c in self.consumers]
            self._aggregate_consumer_demands_from_results(all_needs)

        # Sende aggregierte Bedarfe an S4 (oder zuständiges System)
        if self.system4planner and hasattr(
            self.system4planner, "update_demand_forecast"
        ):
            self.system4planner.update_demand_forecast(
                global_demand=dict(self.aggregated_consumer_demand),
                regional_demand=dict(self.regional_consumer_demand),
            )

    def _aggregate_consumer_demands_from_results(
        self, all_needs: List[Tuple[str, Dict[str, float]]]
    ) -> None:
        """Aggregiere Bedarfe global und regional."""
        self.aggregated_consumer_demand = defaultdict(float)
        self.regional_consumer_demand = defaultdict(lambda: defaultdict(float))

        for region_id, needs in all_needs:
            for good, amount in needs.items():
                self.aggregated_consumer_demand[good] += amount
                self.regional_consumer_demand[region_id][good] += amount

        self.logger.debug(
            "Konsumentenbedarfe (global & regional) aus Ergebnissen aggregiert."
        )

    def stage_infrastructure_development(self):
        # Funktion für einen einzelnen Infra-Agenten
        def _develop_single_infra(infra):
            if hasattr(infra, "step_stage"):
                infra.step_stage("development")
            # Ggf. Status oder Änderungen zurückgeben

        # Führe parallel oder sequentiell aus
        if (
            self.parallel_execution and len(self.infrastructure_agents) > 5
        ):  # Beispielschwellwert
            Parallel(n_jobs=self.config.num_workers)(
                delayed(_develop_single_infra)(i) for i in self.infrastructure_agents
            )
        else:
            for i in self.infrastructure_agents:
                _develop_single_infra(i)

    def stage_system5_policy(self):
        if self.system5policy and hasattr(self.system5policy, "step_stage"):
            self.system5policy.step_stage("system5_policy")

    def stage_system4_strategic_planning(self):
        if self.system4planner and hasattr(self.system4planner, "step_stage"):
            self.system4planner.step_stage("system4_strategic_planning")

    def stage_system4_tactical_planning(self):
        if self.system4planner and hasattr(self.system4planner, "step_stage"):
            self.system4planner.step_stage(
                "system4_tactical_planning"
            )  # Hier werden Pläne erstellt

    def stage_system3_aggregation(self):
        if self.system3manager and hasattr(self.system3manager, "step_stage"):
            self.system3manager.step_stage("system3_aggregation")

    def stage_system2_coordination(self):
        if self.system2coordinator and hasattr(self.system2coordinator, "step_stage"):
            self.system2coordinator.step_stage("system2_coordination")

    def stage_system3_feedback(self):
        if self.system3manager and hasattr(self.system3manager, "step_stage"):
            self.system3manager.step_stage("system3_feedback")  # Name der Methode in S3

    def stage_system1_operations(self):
        if self.system1operator and hasattr(self.system1operator, "step_stage"):
            self.system1operator.step_stage(
                "system1_operations"
            )  # Name der Methode in S1

    def stage_local_production_planning(self):
        # Funktion für einen einzelnen Producer
        def _plan_single_producer(producer):
            if hasattr(producer, "step_stage"):
                producer.step_stage("local_production_planning")
            # Ggf. Bedarf zurückgeben für ADMM

        # Führe parallel oder sequentiell aus
        if self.parallel_execution and len(self.producers) > 10:
            Parallel(n_jobs=self.config.num_workers)(
                delayed(_plan_single_producer)(p) for p in self.producers
            )
        else:
            for p in self.producers:
                _plan_single_producer(p)

    def stage_admm_update(self):
        if (
            self.config.admm_on
            and self.system4planner
            and hasattr(self.system4planner, "step_stage")
        ):
            # ADMM benötigt möglicherweise Interaktion über mehrere Stufen
            # Hier wird eine Iterationsrunde im Planner ausgelöst
            self.system4planner.step_stage("admm_update")

    def stage_production_execution(self):
        """Führt Produktion aus und sammelt Ergebnisse sicher."""

        # Funktion für einen einzelnen Producer
        def _execute_single_producer(
            producer: ProducerAgent,
        ) -> Tuple[str, Dict[str, float], Dict[str, float]]:
            produced_output = {}
            consumed_resources = {}
            if hasattr(producer, "step_stage"):
                producer.step_stage("production_execution")
            if hasattr(producer, "get_produced_output"):  # Holt Output UND leert Lager
                produced_output = producer.get_produced_output()
            # Verbrauch muss anders getrackt werden, z.B. durch History oder Rückgabewert
            # Annahme: Wir holen den letzten Verbrauch aus einer History
            if hasattr(producer, "resource_consumption_history"):
                # Nimm den letzten Eintrag für jede Ressource
                consumed_resources = {
                    res: deque_obj[-1]
                    for res, deque_obj in producer.resource_consumption_history.items()
                    if deque_obj
                }

            return producer.unique_id, produced_output, consumed_resources

        # Führe parallel oder sequentiell aus
        if (
            self.parallel_execution and len(self.producers) > 10
        ):  # Schwellwert für Parallelisierung
            results = Parallel(n_jobs=self.config.num_workers)(
                delayed(_execute_single_producer)(p) for p in self.producers
            )
        else:
            results = [_execute_single_producer(p) for p in self.producers]

        # --- Sequenzieller Teil: Ergebnisse verarbeiten ---
        self.total_production_this_step.clear()  # Zurücksetzen für diesen Schritt
        total_consumption_this_step = defaultdict(float)

        for _, produced_output, consumed_resources in results:
            for good, amount in produced_output.items():
                self.total_production_this_step[good] += amount
            for resource, consumed in consumed_resources.items():
                total_consumption_this_step[resource] += consumed

        # Beispiel: Aktualisiere globales Ressourceninventar (falls ResourceAgent existiert)
        if self.resource_agent and hasattr(self.resource_agent, "inventory"):
            for resource, consumed in total_consumption_this_step.items():
                self.resource_agent.inventory[resource] = (
                    self.resource_agent.inventory.get(resource, 0.0) - consumed
                )
                if self.resource_agent.inventory[resource] < 0:
                    self.logger.warning(
                        f"Ressourcenbestand von '{resource}' nach Produktion negativ ({self.resource_agent.inventory[resource]:.2f}). Prüfe Logik!"
                    )
                    self.resource_agent.inventory[resource] = 0.0  # Korrigiere auf 0

        self.logger.info(
            f"Produktionsergebnisse aggregiert. Gesamtproduktion: {sum(self.total_production_this_step.values()):.2f}"
        )

    # stage_distribution wurde entfernt

    def stage_consumption(self):
        """Konsumenten verbrauchen Güter und bewerten Zufriedenheit."""
        # Hole die verteilten Güter (Annahme: S3/S2 haben sie verteilt)
        # Diese Logik muss angepasst werden, WIE die Konsumenten ihre Güter bekommen!
        # Beispiel Annahme: Die Güter liegen im RegionalManager Storage
        # und werden von hier an die Konsumenten verteilt.

        # 1. Sammle verteilbare Güter pro Region
        regional_available_goods = defaultdict(lambda: defaultdict(float))
        for region_id, rm in self.regional_managers.items():
            regional_available_goods[region_id] = getattr(rm, "storage", {}).copy()
            if hasattr(rm, "storage"):
                rm.storage.clear()  # Leere regionales Lager nach Entnahme

        # 2. Verteile Güter an Konsumenten in der Region (z.B. nach Bedarf)
        for region_id, consumers_in_region in self.consumers_by_region.items():
            if not consumers_in_region:
                continue
            available = regional_available_goods[region_id]
            if not available:
                continue

            # Berechne Gesamtbedarf in der Region für Verteilung
            regional_needs = defaultdict(float)
            for c in consumers_in_region:
                needs = getattr(c, "get_current_needs", lambda: {})()
                for good, amount in needs.items():
                    regional_needs[good] += amount

            # Verteile jedes Gut proportional zum Bedarf
            for good, total_avail in available.items():
                total_need = regional_needs.get(good, 0.0)
                if total_need > 0:
                    for c in consumers_in_region:
                        c_need = getattr(c, "get_current_needs", lambda: {})().get(
                            good, 0.0
                        )
                        share = c_need / total_need if total_need > 0 else 0.0
                        allocation = total_avail * share
                        if allocation > 0:
                            if hasattr(c, "receive_allocation"):
                                c.receive_allocation({good: allocation})

        # 3. Lasse Konsumenten konsumieren und bewerten (parallelisierbar)
        def _consume_and_evaluate(consumer: ConsumerAgent) -> Tuple[str, float]:
            if hasattr(consumer, "step_stage"):
                consumer.step_stage("consumption")  # Eigene Stage im Consumer?
            satisfaction = getattr(consumer, "satisfaction", 0.0)
            return consumer.unique_id, satisfaction

        # Führe parallel oder sequentiell aus
        if self.parallel_execution and len(self.consumers) > 10:
            results = Parallel(n_jobs=self.config.num_workers)(
                delayed(_consume_and_evaluate)(c) for c in self.consumers
            )
        else:
            results = [_consume_and_evaluate(c) for c in self.consumers]

        # 4. Aggregiere Ergebnisse
        satisfactions = [s for _, s in results]
        if satisfactions:
            avg_satisfaction = np.mean(satisfactions)
            self.logger.info(
                f"Konsum abgeschlossen. Durchschnittliche Zufriedenheit: {avg_satisfaction:.3f}"
            )
            if not hasattr(self, "welfare_metrics"):
                self.welfare_metrics = {}
            self.welfare_metrics["avg_satisfaction_current"] = avg_satisfaction

    def stage_environmental_impact(self):
        self.logger.info("Stage: Umweltauswirkungen berechnen")
        self._calculate_environmental_impact()  # Implementiere diese Methode

    def stage_technology_progress(self):
        self.logger.info("Stage: Technologie-Fortschritt aktualisieren")

        # Funktion für einen Producer
        def _progress_single_producer(producer):
            if hasattr(producer, "step_stage"):
                producer.step_stage("innovation_and_research")
                # 'apply_innovation' könnte auch hier oder separat aufgerufen werden
                producer.step_stage("apply_innovation")

        # Führe parallel oder sequentiell aus
        if self.parallel_execution and len(self.producers) > 10:
            Parallel(n_jobs=self.config.num_workers)(
                delayed(_progress_single_producer)(p) for p in self.producers
            )
        else:
            for p in self.producers:
                _progress_single_producer(p)
        self.logger.debug("Technologielevel aktualisiert.")

    def stage_crisis_management(self):
        if hasattr(self, "crisis_manager"):
            self.crisis_manager.update_crisis_states()
        else:
            self.logger.debug("Kein CrisisManager vorhanden.")

    def stage_welfare_assessment(self):
        self.logger.info("Stage: Wohlfahrtsbewertung")
        self._assess_welfare()  # Implementiere diese Methode

    def stage_learning_adaptation(self):
        self.logger.info("Stage: Lernen und Adaption")

        # Führe Lernschritte parallel aus, wenn möglich
        def _learn_single_agent(agent):
            if getattr(agent, "rl_mode", False) and hasattr(agent, "step_rl"):
                agent.step_rl()
            # Ggf. andere Adaptionsmechanismen aufrufen
            if hasattr(agent, "adapt_needs_and_preferences"):
                agent.adapt_needs_and_preferences()  # Bspw. für Consumer

        all_learners = [
            a
            for a in self.agents_by_id.values()
            if hasattr(a, "step_rl") or hasattr(a, "adapt_needs_and_preferences")
        ]
        if self.parallel_execution and len(all_learners) > 10:
            Parallel(n_jobs=self.config.num_workers)(
                delayed(_learn_single_agent)(a) for a in all_learners
            )
        else:
            for a in all_learners:
                _learn_single_agent(a)

        # System-Level Adaption
        if self.historical_plan_accuracy:
            avg_acc = np.mean(list(self.historical_plan_accuracy))
            self.logger.debug(
                f"Durchschnittliche Planungsgenauigkeit (letzte {len(self.historical_plan_accuracy)} Steps): {avg_acc:.3f}"
            )
            # TODO: Füge hier Logik zur Anpassung von Systemparametern hinzu (z.B. ADMM Rho)
        self.logger.debug("Lern- und Adaptionsschritte durchgeführt.")

    def stage_vsm_reconfiguration(self):
        self.logger.info("Stage: VSM Rekonfiguration")
        self._dynamic_vsm_reconfiguration()  # Implementiere diese Methode

    def stage_evaluate_governance_mode(self):
        self.logger.info("Stage: Evaluating Governance Mode...")
        if (
            hasattr(self, "system3manager")
            and self.system3manager is not None
            and hasattr(self.system3manager, "evaluate_and_broadcast_governance_mode")
        ):
            try:
                self.system3manager.evaluate_and_broadcast_governance_mode()
                self.logger.info("Governance mode evaluation and broadcast successful.")
            except Exception as e:
                self.logger.error(
                    f"Error during governance mode evaluation: {e}", exc_info=True
                )
        else:
            self.logger.warning(
                "system3manager or 'evaluate_and_broadcast_governance_mode' method not found. Skipping governance mode evaluation."
            )

    def stage_bookkeeping(self):
        self.logger.info("Stage: Bookkeeping / Datensammlung")
        # Analysiere Plan-Ist-Abweichungen
        plan_discrepancies = self._analyze_plan_discrepancies()
        # Berechne und speichere Metriken
        self._update_metrics(plan_discrepancies)
        # Sammle Daten mit DataCollector
        self.data_collector.collect_data(self)
        # Setze transiente Zustände zurück
        self._reset_transient_state()

    # ---------------------------------------------------------------
    # Implementierung der benötigten Hilfs- und Logik-Methoden
    # (Hier nur Stubs - müssen basierend auf spezifischen Anforderungen
    #  und den anderen Modulen implementiert werden)
    # ---------------------------------------------------------------

    # _aggregate_consumer_demands wurde entfernt/ersetzt

    # _execute_distribution wurde entfernt

    def _calculate_environmental_impact(self) -> None:
        """Berechnet globale und regionale Umweltauswirkungen."""
        impact = defaultdict(float)
        regional = defaultdict(lambda: defaultdict(float))
        # Hole Emissionen von Produzenten (werden in execute_production gesetzt)
        for p in self.producers:
            if hasattr(p, "emissions_history"):
                # Nimm letzten Eintrag pro Emissionstyp
                last_emissions = {
                    etype: deq[-1][1]
                    for etype, deq in p.emissions_history.items()
                    if deq and deq[-1][0] == self.current_step
                }
                for etype, amount in last_emissions.items():
                    impact[etype] += amount
                    if hasattr(p, "region_id"):
                        regional[p.region_id][etype] += amount

        self.environmental_impacts = {
            "global": dict(impact),
            "regional": dict(regional),
        }
        self.logger.debug(f"Umweltauswirkungen berechnet: Global={dict(impact)}")

    # _update_technology wurde entfernt (passiert jetzt in stage_technology_progress)

    def _assess_welfare(self) -> None:
        """Berechnet Wohlfahrts- und Ungleichheitsmetriken."""
        # Hole aktuelle Werte direkt von Agenten oder aus self.detailed_metrics (Bookkeeping)
        prod_outputs = [p.total_output_this_step for p in self.producers]
        consumer_satisfactions = [c.satisfaction for c in self.consumers]

        self.welfare_metrics = {
            "production_gini": gini_coefficient(prod_outputs) if prod_outputs else 0.0,
            "satisfaction_gini": (
                gini_coefficient(consumer_satisfactions)
                if consumer_satisfactions
                else 0.0
            ),
            "avg_satisfaction": (
                np.mean(consumer_satisfactions) if consumer_satisfactions else 0.0
            ),
            # Füge hier ggf. Atkinson/Theil etc. hinzu
        }
        self.logger.debug(
            f"Wohlfahrt berechnet: Avg Sat={self.welfare_metrics['avg_satisfaction']:.3f}, Prod Gini={self.welfare_metrics['production_gini']:.3f}"
        )

    # _execute_learning_adaptation wurde entfernt (passiert in stage_learning_adaptation)

    def _dynamic_vsm_reconfiguration(self) -> None:
        """Passt VSM-Verbindungen oder -Parameter dynamisch an."""
        # TODO: Implementiere Logik zur Anpassung basierend auf Krisen, Performance etc.
        # Z.B. Ändere Autonomielevel von S3, passe Konfliktstrategie in S2 an, etc.
        self.logger.debug("VSM-Rekonfiguration geprüft/durchgeführt (Placeholder).")

    def _analyze_plan_discrepancies(self) -> Dict[str, Any]:
        """Analysiert Abweichungen zwischen Plan und Ausführung."""
        planned = defaultdict(float)
        actual = self.total_production_this_step

        # Hole Plan aus S4 (oder wo immer er herkommt)
        # Annahme: Der *letzte* Plan von S4 ist relevant
        if self.system4planner and getattr(
            self.system4planner, "current_strategic_plan", None
        ):
            plan_targets = getattr(
                self.system4planner.current_strategic_plan, "production_targets", {}
            )
            planned = defaultdict(float, plan_targets)

        all_goods = set(planned.keys()) | set(actual.keys())
        discrepancies = {}
        weighted_diff_sum_sq = 0
        total_plan_value = sum(planned.values())

        for good in all_goods:
            p = planned.get(good, 0.0)
            a = actual.get(good, 0.0)
            diff = a - p
            rel_diff = diff / p if p > 1e-6 else (1.0 if diff > 0 else 0.0)
            discrepancies[good] = {
                "planned": p,
                "actual": a,
                "diff": diff,
                "rel_diff": rel_diff,
            }
            if total_plan_value > 0:
                weighted_diff_sum_sq += (diff**2) * (
                    p / total_plan_value
                )  # Gewichtete quadratische Abweichung

        accuracy = (
            max(0.0, 1.0 - np.sqrt(weighted_diff_sum_sq) / max(1e-6, total_plan_value))
            if total_plan_value > 0
            else 1.0
        )
        self.historical_plan_accuracy.append(accuracy)

        self.logger.debug("Plan-Ist-Abweichungen analysiert.")
        return {"accuracy": accuracy, "details": discrepancies}

    def _update_metrics(self, plan_discrepancies: Dict[str, Any]) -> None:
        """Sammelt alle Metriken für den DataCollector."""
        # Ruft die verschiedenen Berechnungsfunktionen auf oder greift
        # auf bereits berechnete Attribute zu.
        # Hole Metriken aus den verschiedenen Bereichen
        core_metrics = (
            self.data_collector._calculate_core_stats(self)
            if hasattr(self.data_collector, "_calculate_core_stats")
            else {}
        )
        prod_metrics = (
            self.data_collector._calculate_production_metrics(self)
            if hasattr(self.data_collector, "_calculate_production_metrics")
            else {}
        )
        res_metrics = (
            self.data_collector._calculate_resource_metrics(self)
            if hasattr(self.data_collector, "_calculate_resource_metrics")
            else {}
        )
        tech_metrics = (
            self.data_collector._calculate_technology_metrics(self)
            if hasattr(self.data_collector, "_calculate_technology_metrics")
            else {}
        )
        infra_metrics = (
            self.data_collector._calculate_infrastructure_metrics(self)
            if hasattr(self.data_collector, "_calculate_infrastructure_metrics")
            else {}
        )
        crisis_metrics = (
            self.data_collector._calculate_crisis_metrics(self)
            if hasattr(self.data_collector, "_calculate_crisis_metrics")
            else {}
        )

        self.detailed_metrics = {
            "step": self.current_step,
            **core_metrics,
            **prod_metrics,
            **res_metrics,
            **tech_metrics,
            **infra_metrics,
            **crisis_metrics,
            "environmental": self.environmental_impacts,
            "welfare": self.welfare_metrics,
            "planning": plan_discrepancies,
            # Füge VSM-spezifische Metriken hinzu?
            # "vsm_state": { ... }
        }
        self.logger.debug("Alle Metriken für den Step aktualisiert.")

    def _reset_transient_state(self) -> None:
        """Setzt Zustände zurück, die nur für einen Step gelten."""
        # Transportnetzwerk-Fluss zurücksetzen
        for r1 in self.transport_network:
            for r2 in self.transport_network.get(r1, {}):
                if "current_flow" in self.transport_network[r1][r2]:
                    self.transport_network[r1][r2]["current_flow"] = 0.0
                if "congestion" in self.transport_network[r1][r2]:
                    self.transport_network[r1][r2]["congestion"] = 0.0

        # 'market_production' / 'output_stock' wird jetzt in execute_production geleert
        # 'consumption_received' der Konsumenten
        for c in self.consumers:
            if hasattr(c, "reset_consumption_received"):
                c.reset_consumption_received()

        self.logger.debug("Transiente Zustände für nächsten Step zurückgesetzt.")

    # ---------------------------------------------------------------
    # Öffentliche Methoden
    # ---------------------------------------------------------------

    def run_simulation(self) -> None:
        """Führt die gesamte Simulation für die konfigurierte Anzahl Schritte durch."""
        self.logger.info(
            f"Starte Simulation '{self.config.simulation_name}' für {self.config.simulation_steps} Schritte."
        )
        while self.current_step < self.config.simulation_steps:
            if not self.step():
                self.logger.error(
                    f"Simulation in Schritt {self.current_step + 1} vorzeitig beendet."
                )
                break
        self.logger.info("Simulation abgeschlossen.")

    def get_results(self) -> Dict[str, Any]:
        """Gibt die gesammelten Simulationsdaten zurück."""
        return self.data_collector.export_data(format="dict")

    def save_results(self, path: str, format: str = "json") -> None:
        """Speichert die Simulationsdaten in einer Datei."""
        self.data_collector.export_data(format=format, path=path)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Gibt eine Zusammenfassung der wichtigsten Metriken zurück."""
        if hasattr(self.data_collector, "get_summary_statistics"):
            summary = self.data_collector.get_summary_statistics()
            # flatten common metrics for backwards compatibility
            if isinstance(summary.get("total_output"), dict):
                summary["last_total_output"] = summary["total_output"].get("last")
                summary["total_output"] = summary["total_output"].get("last")
            if isinstance(summary.get("production_gini"), dict):
                summary["production_gini"] = summary["production_gini"].get("last")
            return summary
        else:
            # Einfacher Fallback
            if self.detailed_metrics:
                welfare = self.detailed_metrics.get("welfare", {})
                planning = self.detailed_metrics.get("planning", {})
                prod = self.detailed_metrics.get("production_metrics", {})
                return {
                    "last_step": self.detailed_metrics.get("step"),
                    "last_total_output": prod.get("total_output"),
                    "last_avg_satisfaction": welfare.get("avg_satisfaction"),
                    "last_plan_accuracy": planning.get("accuracy"),
                }
            return {"status": "Keine Metriken verfügbar."}

    def get_agent(self, agent_id: str) -> Optional[Any]:
        """Gibt einen Agenten anhand seiner ID zurück."""
        return self.agents_by_id.get(agent_id)

    def route_directive(self, directive: Any) -> bool:
        """Leitet Direktiven (z.B. von S2) an das Zielsystem weiter."""
        target_system_name = getattr(directive, "target_system", None)
        success = False
        self.logger.debug(
            f"Versuche Direktive an System '{target_system_name}' zu routen: {directive}"
        )
        try:
            target_attr_map = {
                "System3": self.system3manager,
                "System4": self.system4planner,
                "System5": self.system5policy,
                # Füge hier ggf. Mapping zu RegionalManagern hinzu, falls Direktiven direkt an sie gehen
            }
            target_system = target_attr_map.get(target_system_name)

            if target_system:
                # Finde passende Methode im Zielsystem
                method_name = None
                if hasattr(directive, "action_type"):
                    # Versuche Methode basierend auf action_type zu finden
                    method_name_by_action = {
                        "resource_transfer": "apply_resolution_directive",  # S3
                        "request_plan_adjustment": "receive_directive_or_feedback",  # S4
                        "request_intervention": "handle_escalation",  # S5
                        # Füge hier weitere Mappings hinzu
                    }.get(directive.action_type)
                    if method_name_by_action and hasattr(
                        target_system, method_name_by_action
                    ):
                        method_name = method_name_by_action

                # Fallback: Generische Methoden
                if not method_name:
                    if hasattr(target_system, "receive_directive"):
                        method_name = "receive_directive"
                    elif hasattr(target_system, "handle_directive"):
                        method_name = "handle_directive"
                    elif hasattr(target_system, "receive_directive_or_feedback"):
                        method_name = "receive_directive_or_feedback"

                if method_name:
                    method_to_call = getattr(target_system, method_name)
                    result = method_to_call(directive)
                    # Erfolg hängt davon ab, was die Methode zurückgibt
                    # Annahme: bool oder None (None wird als Erfolg gewertet)
                    success = result is not False
                    self.logger.info(
                        f"Direktive an System '{target_system_name}' via Methode '{method_name}' geroutet (Erfolg: {success})."
                    )
                else:
                    self.logger.warning(
                        f"Keine passende Methode im Zielsystem '{target_system_name}' für Direktive gefunden."
                    )
            else:
                self.logger.warning(
                    f"Kann Direktive nicht weiterleiten: Zielsystem '{target_system_name}' unbekannt oder nicht vorhanden."
                )

        except Exception as e:
            self.logger.error(
                f"Fehler beim Verarbeiten/Routen der Direktive für System '{target_system_name}': {e}",
                exc_info=True,
            )
            success = False

        return success
