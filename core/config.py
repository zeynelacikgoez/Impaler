# Impaler/core/config.py
"""
Umfassendes Konfigurationsmodul für das Impaler Simulationsframework
unter Verwendung von Pydantic für Datenvalidierung und -struktur.

Dieses Modul definiert die Datenstrukturen für alle Konfigurationsaspekte,
einschließlich Simulationseinstellungen, Agentenparameter, VSM-Einstellungen,
Szenarien, Umweltbedingungen und mehr. Es ersetzt die vorherige Implementierung
mit Standardklassen durch Pydantic-Modelle für erhöhte Robustheit und Klarheit.
"""

import json
import os
import random
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable, Literal

from .config_models import PlanwirtschaftParams

# Pydantic für Datenvalidierung und -modellierung importieren
# Stellen Sie sicher, dass pydantic installiert ist: pip install pydantic
from pydantic import BaseModel, Field, validator, ValidationError, root_validator

# Logger für dieses Modul
logger = logging.getLogger(__name__)

# --- Hilfsmodelle für verschachtelte Konfigurationsstrukturen ---

class ProductionPathInput(BaseModel):
    """Definiert einen einzelnen Input für einen Produktionspfad."""
    resource_type: str
    amount_per_unit: float = Field(..., ge=0.0)
    is_essential: bool = True
    # Beispiel: Substitutes könnten hier auch als Pydantic-Modell definiert werden
    substitutes: List[Tuple[str, float]] = Field(default_factory=list)

class ProductionPath(BaseModel):
    """Definiert einen spezifischen Produktionspfad für ein Gut."""
    # Zuvor war 'inputs' ein Dict, jetzt eine Liste von Input-Objekten für mehr Struktur
    input_requirements: List[ProductionPathInput] = Field(default_factory=list)
    efficiency: float = Field(1.0, ge=0.0)
    emission_factors: Optional[Dict[str, float]] = Field(default_factory=dict) # z.B. {"co2": 1.5}
    name: Optional[str] = None # Optionaler Name für den Pfad

class ResourceShockModel(BaseModel):
    """Modell für einen Ressourcenschock in Szenarien."""
    step: int = Field(..., ge=0)
    resource: str
    # Relative Änderung: <0 = Verknappung, >0 = Überschuss (z.B. -0.2 für -20%)
    relative_change: float
    duration: Optional[int] = Field(None, ge=1) # Dauer in Steps, None für permanent
    affected_regions: Optional[List[str]] = None # None für global
    name: str = "Unnamed Resource Shock"
    description: str = ""
    is_random: bool = False

class TechnologyBreakthroughModel(BaseModel):
    """Modell für einen Technologie-Durchbruch in Szenarien."""
    step: int = Field(..., ge=0)
    affected_goods: List[str]
    efficiency_boost: float = Field(0.0, ge=0.0) # Faktor > 1.0, z.B. 1.1 für +10%
    # Reduziert den Bedarf an Inputs, z.B. {"energy": 0.9} für 10% weniger Energie
    input_reduction_factors: Dict[str, float] = Field(default_factory=dict)
    # Optional: Reduziert Emissionen
    emission_reduction_factors: Dict[str, float] = Field(default_factory=dict)
    adoption_rate: float = Field(1.0, ge=0.0, le=1.0) # Anteil der Producer, die sofort adaptieren
    name: str = "Unnamed Tech Breakthrough"
    description: str = ""

class PolicyChangeModel(BaseModel):
    """Modell für eine Politikänderung in Szenarien."""
    step: int = Field(..., ge=0)
    # Typ der Änderung, z.B. "priority_weight", "resource_limit", "target_adjustment"
    policy_type: str
    # Name des Parameters/Ziels, das geändert wird, z.B. "planning_priorities.fairness_weight" oder "environment_config.sustainability_targets.co2"
    target_parameter: str
    # Der neue Wert oder die relative Änderung (abhängig vom policy_type)
    value: Any
    affected_scope: Optional[Union[str, List[str]]] = None # z.B. "Region1", ["GoodA"] oder None für global
    name: str = "Unnamed Policy Change"
    description: str = ""

# --- Additional small config models used in tests ---

class ConsumerAdaptationParams(BaseModel):
    """Parameters controlling consumer adaptation behaviour."""
    enable_need_adaptation: bool = True
    adaptation_rate: float = 0.05
    preference_adaptation_rate: float = 0.01
    substitution_learning_rate: float = 0.02
    social_influence_factor: float = 0.1
    substitution: Dict[str, Any] = Field(
        default_factory=lambda: {
            "learning_rate": 0.15,
            "memory_decay": 0.97,
            "shortage_threshold": 0.25,
            "price_threshold": 0.05,
            "elasticity_rho": 0.3,
            "budget_fixed": True,
        }
    )

    def get(self, item: str, default: Any = None) -> Any:  # pragma: no cover - small helper
        return getattr(self, item, default)

class ProducerParamsConfig(BaseModel):
    """Basic producer-level economic parameters."""
    base_investment_rate: float = 0.03
    capacity_cost_per_unit: float = 10.0
    tech_investment_increase_factor: float = 0.05
    max_tech_level: float = 2.0

class InnovationEffect(BaseModel):
    """Simple representation of an innovation effect entry."""
    target: str
    type: str = "multiplicative"
    value: float = 1.0
    scope: str = "agent"


# --- Konfigurationsklassen für Hauptbereiche ---

class DemandConfig(BaseModel):
    """Konfiguration für Bedarfsmodellierung."""
    population_growth_rate: float = 0.01
    regional_growth_variation: Dict[str, float] = Field(default_factory=dict)
    population_max_capacity: Optional[float] = None
    base_consumption_per_capita: Dict[str, float] = Field(default_factory=lambda: {"A": 2.0, "B": 1.5, "C": 0.8})
    essential_goods: List[str] = Field(default_factory=lambda: ["A"])
    luxury_goods: List[str] = Field(default_factory=lambda: ["C"])
    elasticities: Dict[str, float] = Field(default_factory=lambda: {"A": 0.2, "B": 0.5, "C": 0.8})
    cross_elasticities: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    max_saturation_levels: Dict[str, float] = Field(default_factory=lambda: {"A": 5.0, "B": 3.0, "C": 2.5})
    saturation_progression_rate: Dict[str, float] = Field(default_factory=lambda: {"A": 0.05, "B": 0.08, "C": 0.12})
    preference_evolution: Dict[str, Dict[int, float]] = Field(default_factory=dict)
    seasonal_factors: Dict[str, Dict[int, float]] = Field(default_factory=dict) # {Gut: {Monat (1-12): Faktor}}
    enable_dynamic_demand: bool = True
    update_interval_steps: int = Field(5, ge=1)
    regional_demand_factors: Dict[str, Dict[str, float]] = Field(default_factory=dict) # {Region: {Gut: Faktor}}
    demographic_segments: Dict[str, Dict[str, float]] = Field(default_factory=dict) # {Segment: {Gut: Faktor}}
    consumer_adaptation_params: Dict[str, Any] = Field(default_factory=lambda: {
        "enable_need_adaptation": True,
        "adaptation_rate": 0.05,
        "shortage_adaptation": {"threshold_steps": 5, "severity_factor": 0.03, "max_reduction_factor": 0.7},
        "surplus_adaptation": {"threshold_steps": 7, "increase_factor": 1.02, "max_increase_factor": 1.5},
        "good_specific_adaptation": {},
        "substitution": {
            "learning_rate": 0.15,
            "memory_decay": 0.97,
            "shortage_threshold": 0.25,
            "price_threshold": 0.05,
            "elasticity_rho": 0.3,
            "budget_fixed": True,
        },
    })

class ScenarioConfig(BaseModel):
    """Konfiguration für externe Ereignisse und Schocks."""
    resource_shocks: List[ResourceShockModel] = Field(default_factory=list)
    technology_breakthroughs: List[TechnologyBreakthroughModel] = Field(default_factory=list)
    policy_changes: List[PolicyChangeModel] = Field(default_factory=list)
    # TODO: Weitere Event-Modelle hinzufügen (DemographicEventModel, TradeDisruptionModel etc.)
    # demographic_events: List[Any] = Field(default_factory=list)
    # trade_disruptions: List[Any] = Field(default_factory=list)
    enable_scenario_events: bool = True
    random_events_enabled: bool = False
    random_events_probability: float = Field(0.05, ge=0.0, le=1.0)
    # Typen für zufällige Events (müssen zu den definierten Modellen passen)
    random_event_types: List[str] = ["resource_shock", "policy_change"]

class IOConfig(BaseModel):
    """Konfiguration für Input-Output Beziehungen und Technologie."""
    # Beispiel: {"GoodA": [ProductionPath(...), ProductionPath(...)], "GoodB": ProductionPath(...)}
    initial_io_matrix: Dict[str, Union[ProductionPath, List[ProductionPath]]] = Field(default_factory=dict)
    enable_dynamic_io: bool = True
    # Step -> Good -> Neuer Parameter (z.B. 'efficiency', 'inputs.resource') -> Neuer Wert/Faktor
    technology_upgrade_steps: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    # Konfiguration für Innovations-Effekte (Beispiel)
    innovation_effects: Dict[str, Union[InnovationEffect, Dict[str, Any]]] = Field(
        default_factory=lambda: {
            "process_optimization": {
                "target": "line_efficiency",
                "type": "multiplicative",
                "value": 1.05,
                "scope": "all",
            },
            "material_science": {
                "target": "input_requirement",
                "type": "multiplicative",
                "value": 0.90,
                "scope": "all",
            },
            "automation": {
                "target": "agent_capacity",
                "type": "multiplicative",
                "value": 1.1,
                "scope": "agent",
            },
        }
    )

    @validator("innovation_effects", pre=True)
    def _validate_innovation_effects(cls, v):
        if isinstance(v, dict):
            return {
                key: (val if isinstance(val, InnovationEffect) else InnovationEffect(**val))
                for key, val in v.items()
            }
        return v

class EnvironmentConfig(BaseModel):
    """Konfiguration für Umweltaspekte."""
    # Gut -> Emissionstyp -> Faktor pro Output-Einheit (kann auch in IOConfig.ProductionPath sein)
    emission_factors: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {
        "A": {"co2": 2.0, "pollution": 1.5}, "B": {"co2": 3.0, "pollution": 2.0}
    })
    environmental_capacities: Dict[str, float] = Field(default_factory=lambda: {"co2": 1000.0, "pollution": 500.0})
    environmental_thresholds: Dict[str, Dict[float, str]] = Field(default_factory=dict)
    resource_capacities: Dict[str, float] = Field(default_factory=lambda: {"wood": 1500.0, "water": 3000.0})
    resource_regeneration_rates: Dict[str, float] = Field(default_factory=lambda: {"wood": 0.1, "water": 0.2})
    # {Umweltfaktor: {Step: Zielwert}}
    sustainability_targets: Dict[str, Dict[int, float]] = Field(default_factory=dict)

class PlanningPriorities(BaseModel):
    """Konfiguration für Planungsprioritäten."""
    goods_priority: Dict[str, float] = Field(default_factory=lambda: {"A": 0.8, "B": 0.5, "C": 0.3})
    resource_priority: Dict[str, float] = Field(default_factory=lambda: {"energy": 0.7, "water": 0.8})
    fairness_weight: float = Field(1.0, ge=0)
    co2_weight: float = Field(2.0, ge=0)
    efficiency_weight: float = Field(1.5, ge=0)
    resilience_weight: float = Field(1.0, ge=0)
    planwirtschaft_params: PlanwirtschaftParams = Field(
        default_factory=PlanwirtschaftParams,
        description="Detaillierte Parameter für die planwirtschaftliche Steuerung und Kostenfunktionen."
    )
    # {FaktorName: {Step: NeuerWert}}
    priority_evolution: Dict[str, Dict[int, float]] = Field(default_factory=dict)
    # {Region: {FaktorName: Multiplikator}}
    regional_priorities: Dict[str, Dict[str, float]] = Field(default_factory=dict)

class RegionalConfig(BaseModel):
    """Konfiguration für regionale Unterschiede."""
    regions: List[str] = ["Region1", "Region2"]
    # {Region: {Gut: Kapazitätsfaktor}}
    regional_production_capacity: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    # {Region: {Ressource: Verfügbarkeitsfaktor}}
    regional_resources: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    # {Region: {Gut: Effizienzfaktor}}
    regional_efficiency: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    # Transportkosten: {RegionVon: {RegionNach: {Gut/Ressource: Kostenfaktor}}}
    transport_costs: Dict[str, Dict[str, Dict[str, float]]] = Field(default_factory=dict)
    # Transportkapazität: {RegionVon: {RegionNach: Kapazität pro Step}}
    transport_capacity: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    # {Region: {Step: {Eigenschaft: Änderungsfaktor}}}
    development_paths: Dict[str, Dict[int, Dict[str, float]]] = Field(default_factory=dict)
    # {Region: [Liste spezialisierter Güter]}
    regional_specialization: Dict[str, List[str]] = Field(default_factory=dict)

class ADMMConfig(BaseModel):
    """Konfiguration für ADMM-Parameter."""
    rho: float = Field(0.1, gt=0)
    rho_min: float = Field(0.01, gt=0, description="Minimum value for adaptive rho.")
    rho_max: float = Field(50.0, gt=0, description="Maximum value for adaptive rho.")
    tolerance: float = Field(1e-3, gt=0)
    max_iterations: int = Field(20, ge=1)
    # Spezifische Überschreibungen pro Gut oder Region
    good_specific_params: Dict[str, Dict[str, float]] = Field(default_factory=dict) # {"GoodA": {"rho": 0.15}}
    regional_params: Dict[str, Dict[str, float]] = Field(default_factory=dict) # {"Region1": {"max_iterations": 25}}

class AgentParameterConfig(BaseModel):
    """Parameter für Agenten, unterstützt Ranges für Randomisierung."""
    # Beispiel: {"initial_capacity": (80.0, 120.0), "depreciation_rate": 0.02}
    # _resolve_param im Model wird diese interpretieren
    params: Dict[str, Any] = Field(default_factory=dict)

class AgentPopulationConfig(BaseModel):
    """Konfiguration für eine Population von Agenten."""
    agent_class: str # z.B. "ProducerAgent", "ConsumerAgent"
    count: int = Field(..., ge=0)
    params: Dict[str, Any] = Field(default_factory=dict) # Parameter für __init__, können Ranges enthalten
    region_distribution: Optional[Dict[Union[int, str], Union[int, float]]] = None # Region ID -> Anzahl oder Anteil
    id_prefix: Optional[str] = None

class SpecificAgentConfig(BaseModel):
    """Konfiguration für eine spezifische Agenten-Instanz."""
    agent_class: str
    unique_id: str
    params: Dict[str, Any] # Enthält region_id und andere spezifische Parameter

class LoggingConfig(BaseModel):
    """Konfiguration für Logging."""
    log_level: str = "INFO"
    log_to_console: bool = True
    log_to_file: bool = False
    log_file_path: str = "impaler_simulation.log"
    log_format: str = '%(asctime)s - [%(name)s] %(levelname)s - %(message)s'


class StateEstimatorConfig(BaseModel):
    """Konfiguration für den StateEstimator (AEKF)."""
    enabled: bool = Field(True, description="Aktiviert oder deaktiviert die Zustandsschätzung.")
    state_dimension: int = Field(20, ge=1, description="Dimension n des Systemzustandsvektors x_t.")
    measurement_dimension: int = Field(50, ge=1, description="Dimension p des Messvektors y_t.")
    ensemble_size: int = Field(100, ge=10, description="Anzahl der Partikel (Ensemble-Members) N für den AEKF.")
    initial_process_noise_q: float = Field(0.01, gt=0, description="Initialer Skalar für die Prozessrauschen-Kovarianz Q.")
    initial_measurement_noise_r: float = Field(0.1, gt=0, description="Initialer Skalar für die Messrauschen-Kovarianz R.")


class EconomicModelConfig(BaseModel):
    """Minimal container for high level model sub-configs used in tests."""
    producer_params: ProducerParamsConfig = Field(default_factory=ProducerParamsConfig)
    io_config: IOConfig = Field(default_factory=IOConfig)

# --- Haupt-Simulationskonfiguration ---

class SimulationConfig(BaseModel):
    """Hauptkonfigurationsklasse für die Impaler-Simulation."""

    # --- Subkonfigurationen ---
    demand_config: DemandConfig = Field(default_factory=DemandConfig)
    scenario_config: ScenarioConfig = Field(default_factory=ScenarioConfig)
    io_config: IOConfig = Field(default_factory=IOConfig)
    environment_config: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    planning_priorities: PlanningPriorities = Field(default_factory=PlanningPriorities)
    regional_config: RegionalConfig = Field(default_factory=RegionalConfig)
    admm_config: ADMMConfig = Field(default_factory=ADMMConfig)
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)
    state_estimator_config: StateEstimatorConfig = Field(default_factory=StateEstimatorConfig)

    # --- Grundlegende Simulation ---
    simulation_name: str = "Impaler_Default_Run"
    # Globale Listen, werden auch zur Validierung von Einträgen in anderen Sektionen genutzt
    goods: List[str] = ["A", "B", "C"]
    resources: List[str] = ["iron_ore", "wood", "energy", "water"]
    simulation_steps: int = Field(100, ge=1)
    random_seed: int = 42

    # --- VSM-Parameter & Algorithmen ---
    vsm_on: bool = True
    admm_on: bool = True # Steuert, ob ADMM-Stage aktiv ist
    # ml_adaptation_on: bool = False # Eigener Schalter für ML
    # crisis_management_on: bool = True # Eigener Schalter für Krisen

    # --- Performance Einstellungen ---
    performance_profile: Literal["fast_prototype", "balanced", "high_accuracy"] = "balanced"

    # --- Agenten-Definitionen (NEU) ---
    agent_populations: Dict[str, AgentPopulationConfig] = Field(default_factory=dict)
    specific_agents: List[SpecificAgentConfig] = Field(default_factory=list)

    # --- Simulation-Stages ---
    stages: List[str] = [
        "resource_regen",
        "state_estimation",
        "need_estimation",
        "infrastructure_development",
        "system5_policy",
        "system4_strategic_planning_admm",
        "system4_tactical_planning",
        "system3_aggregation",
        "system2_coordination",
        "system3_feedback",
        "broadcast_plan_and_prices",
        "system1_operations",
        "local_production_planning",
        "local_rl_execution",
        "admm_update",
        "production_execution",
        "consumption",
        "environmental_impact",
        "technology_progress",
        "crisis_management",
        "welfare_assessment",
        "learning_adaptation",
        "vsm_reconfiguration",
        "bookkeeping",
    ]

    # --- Parallelisierung ---
    parallel_execution: bool = False
    num_workers: int = Field(4, ge=1)

    # --- Zusatzparameter ---
    # Erlaubt das Hinzufügen beliebiger weiterer Parameter für Erweiterungen
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    # --- Pydantic Konfiguration ---
    class Config:
        validate_assignment = True # Prüft Typen auch bei nachträglicher Zuweisung
        extra = 'allow' # Erlaubt zusätzliche Felder, die nicht explizit definiert sind
                        # Alternativ 'forbid', um Tippfehler in Keys zu verhindern

    # --- Convenience Methoden ---
    def apply_performance_profile(self) -> None:
        """Passt Parameter anhand des gewählten Performance-Profils an."""
        profile = self.performance_profile
        if profile == "fast_prototype":
            self.admm_config.max_iterations = min(self.admm_config.max_iterations, 5)
            self.admm_config.tolerance = max(self.admm_config.tolerance, 1e-2)
            self.logging_config.log_level = "WARNING"
        elif profile == "high_accuracy":
            self.admm_config.max_iterations = max(self.admm_config.max_iterations, 50)
            self.admm_config.tolerance = min(self.admm_config.tolerance, 1e-4)

    # --- Validatoren (Beispiele) ---
    @validator('goods', 'resources', each_item=True)
    def name_must_be_valid(cls, v):
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("Gut/Ressourcen-Namen dürfen nicht leer sein.")
        return v

    @root_validator(skip_on_failure=True)
    def check_goods_resources_consistency(cls, values):
        """Prüft, ob in Konfigurationen verwendete Güter/Ressourcen auch global definiert sind."""
        defined_goods = set(values.get('goods', []))
        defined_resources = set(values.get('resources', []))

        configs_to_check = [
            values.get('demand_config'),
            values.get('io_config'),
            values.get('environment_config'),
            values.get('planning_priorities'),
            values.get('regional_config'),
        ]

        errors = []

        # Check goods used in various configs only if the field was explicitly provided
        goods_keys_to_check = ['base_consumption_per_capita', 'elasticities', 'max_saturation_levels', 'goods_priority']
        for cfg in filter(None, configs_to_check):
            cfg_dict = cfg.dict()
            provided = getattr(cfg, '__fields_set__', set())
            for key in goods_keys_to_check:
                if key in cfg_dict and key in provided:
                    for good in cfg_dict[key].keys():
                        if good not in defined_goods:
                            errors.append(
                                f"Gut '{good}' in '{key}' verwendet, aber nicht in globaler 'goods'-Liste definiert."
                            )

        # Check resources used in various configs
        resources_keys_to_check = ['resource_priority', 'resource_capacities', 'resource_regeneration_rates']
        for cfg in filter(None, configs_to_check):
            cfg_dict = cfg.dict()
            provided = getattr(cfg, '__fields_set__', set())
            for key in resources_keys_to_check:
                if key in cfg_dict and key in provided:
                    for resource in cfg_dict[key].keys():
                        if resource not in defined_resources:
                            errors.append(
                                f"Ressource '{resource}' in '{key}' verwendet, aber nicht in globaler 'resources'-Liste definiert."
                            )

        # Check IO Matrix
        io_cfg = values.get('io_config')
        if io_cfg and 'initial_io_matrix' in getattr(io_cfg, '__fields_set__', set()):
            io_matrix = io_cfg.initial_io_matrix
            for output_good, paths in io_matrix.items():
                if output_good not in defined_goods:
                    errors.append(
                        f"Output-Gut '{output_good}' in 'initial_io_matrix' verwendet, aber nicht in globaler 'goods'-Liste definiert."
                    )
                path_list = paths if isinstance(paths, list) else [paths]
                for path in path_list:
                    if hasattr(path, 'input_requirements'):
                        for req in path.input_requirements:
                            if req.resource_type not in defined_goods and req.resource_type not in defined_resources:
                                errors.append(
                                    f"Input '{req.resource_type}' für Gut '{output_good}' verwendet, aber weder als Gut noch als Ressource definiert."
                                )
                    elif isinstance(path, dict) and 'inputs' in path:
                        for resource in path['inputs'].keys():
                            if resource not in defined_goods and resource not in defined_resources:
                                errors.append(
                                    f"Input '{resource}' für Gut '{output_good}' verwendet, aber weder als Gut noch als Ressource definiert."
                                )


        if errors:
            for err in set(errors):
                logger.warning(err)

        return values

    # --- Methoden zum Laden/Speichern ---

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """Konvertiert die Konfiguration in ein serialisierbares Dictionary."""
        # Pydantic v2 verwendet ``model_dump`` für den Export
        return self.model_dump(exclude_unset=False, **kwargs)

    def to_json_file(self, file_path: str, **kwargs) -> None:
        """Speichert die Konfiguration als JSON-Datei."""
        # Stelle sicher, dass das Verzeichnis existiert
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # ``model_dump_json`` ist der empfohlene Weg in Pydantic v2
                f.write(self.model_dump_json(indent=2, exclude_unset=False, **kwargs))
            logger.info(f"Konfiguration erfolgreich gespeichert: {file_path}")
        except IOError as e:
            logger.error(f"Fehler beim Schreiben der Konfigurationsdatei '{file_path}': {e}")
            raise

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Erstellt eine SimulationConfig aus einem Dictionary mit Validierung."""
        try:
            cfg = cls.parse_obj(data)
            cfg.apply_performance_profile()
            return cfg
        except ValidationError as e:
            logger.error(f"Fehler beim Validieren der Konfiguration aus Dictionary:\n{e}")
            # Fehler direkt weitergeben, damit Tests auf ValidationError reagieren koennen
            raise

    @classmethod
    def from_json_file(cls, file_path: str) -> 'SimulationConfig':
        """Lädt eine Konfiguration aus einer JSON-Datei mit Validierung."""
        logger.info(f"Lade Konfiguration aus: {file_path}")
        try:
            cfg = cls.parse_file(file_path)
            cfg.apply_performance_profile()
            return cfg
        except FileNotFoundError:
            logger.error(f"Konfigurationsdatei nicht gefunden: {file_path}")
            raise
        except json.JSONDecodeError as e:
             logger.error(f"Fehler beim Parsen der JSON-Konfigurationsdatei '{file_path}': {e}")
             # Ursprünglichen JSONDecodeError weiterreichen
             raise
        except ValidationError as e:
            logger.error(f"Fehler beim Validieren der JSON-Konfiguration '{file_path}':\n{e}")
            # Auch hier das ursprüngliche ValidationError weiterreichen
            raise


# --- Hilfsfunktionen für Standardkonfigurationen ---

def create_default_config(**overrides) -> SimulationConfig:
    """
    Erstellt eine Standardkonfiguration mit sinnvollen Defaults und optionalen Überschreibungen.

    Args:
        **overrides: Keyword-Argumente, um Standardwerte zu überschreiben.

    Returns:
        SimulationConfig-Objekt.
    """
    # Erstelle Basis-Config mit Pydantic-Defaults
    config = SimulationConfig()

    # Füge Beispiel-Agenten hinzu, wenn keine definiert sind
    if not config.agent_populations and not config.specific_agents:
        logger.debug("Keine Agenten in der Konfiguration definiert, füge Standard-Populationen hinzu.")
        config.agent_populations = {
            "default_producers": AgentPopulationConfig(
                agent_class="ProducerAgent",
                count=20, # Angepasst von 10
                params={
                    "initial_capacity": (70.0, 130.0), # Etwas breitere Range
                    "initial_tech_level": (0.7, 1.3),
                     # Beispiel für Produktionslinien - spezifischer machen!
                     # Dies erfordert, dass ProductionPath als Pydantic-Modell definiert ist
                     "production_lines": [
                         ProductionPath(input_requirements=[ProductionPathInput(resource_type="energy", amount_per_unit=0.5)], efficiency=1.0, name="Line_Default_A", output_good="A"),
                         ProductionPath(input_requirements=[ProductionPathInput(resource_type="energy", amount_per_unit=0.6)], efficiency=0.9, name="Line_Default_B", output_good="B")
                     ],
                    "depreciation_rate": 0.015
                },
                id_prefix="prod_"
            ),
             "default_consumers": AgentPopulationConfig(
                agent_class="ConsumerAgent",
                count=50, # Angepasst von 20
                params={
                    "base_needs": {"A": (1.8, 2.2), "B": (1.3, 1.7), "C": (0.6, 1.0)}
                },
                id_prefix="cons_"
            )
        }

    # Wende Overrides an, falls vorhanden
    if overrides:
        try:
            config = config.copy(update=overrides)
        except ValidationError as e:
             logger.error(f"Fehler beim Anwenden der Overrides auf die Default-Konfiguration: {e}")
             raise ValueError(f"Fehler in Overrides: {e}") from e

    config.apply_performance_profile()
    return config

# TODO: Implementiere create_minimal_config, create_environmental_focus_config etc.
#       basierend auf der neuen Pydantic-Struktur, falls benötigt.