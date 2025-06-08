# Impaler/core/config_models.py
"""
Pydantic-Modelle für verschachtelte Konfigurationsstrukturen in SimulationConfig.

Dieses Modul enthält die Definitionen für Datenstrukturen, die innerhalb
der Haupt-Konfigurationsklassen (z.B. in Listen oder Dictionaries) verwendet werden.
Die Verwendung von Pydantic stellt Typ-Sicherheit und Validierung sicher.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union, Tuple

# Konstante für numerische Stabilität
EPSILON = 1e-9

# --- Modelle für IOConfig ---

class ProductionPathInput(BaseModel):
    """Definiert einen einzelnen Input für einen Produktionspfad."""
    resource_type: str = Field(..., description="Name der benötigten Ressource/des Inputs (muss in config.goods oder config.resources definiert sein)")
    amount_per_unit: float = Field(..., ge=0.0, description="Benötigte Menge pro produzierter Output-Einheit (>0)")
    is_essential: bool = Field(True, description="Ist diese Ressource zwingend erforderlich für diesen Pfad?")
    # Liste möglicher Substitute als Tupel (Ersatzressource, Substitutionsfaktor).
    # Faktor gibt an, wie viel Ersatz pro Einheit Original benötigt wird (Faktor > 1 bedeutet mehr Ersatz nötig).
    substitutes: List[Tuple[str, float]] = Field(default_factory=list, description="Liste möglicher Substitute [(Ersatz, Faktor > 0)]")

    @validator('substitutes', each_item=True)
    def check_substitute_factor(cls, v: Tuple[str, float]) -> Tuple[str, float]:
        """Stellt sicher, dass Substitutionsfaktoren positiv sind."""
        name, factor = v
        if factor <= 0:
            raise ValueError(f"Substitutionsfaktor für '{name}' muss positiv sein.")
        return v

class ProductionPath(BaseModel):
    """Definiert einen spezifischen Produktionspfad für ein Gut."""
    name: Optional[str] = Field(None, description="Optionaler Name für diesen Produktionspfad (z.B. 'Standard', 'Öko')")
    input_requirements: List[ProductionPathInput] = Field(..., description="Liste der benötigten Inputs für diesen Pfad")
    efficiency: float = Field(1.0, gt=0.0, description="Effizienz dieses Pfades (höher ist besser, >0)")
    # Emissionen pro *produzierter* Output-Einheit dieses Pfades
    emission_factors: Dict[str, float] = Field(default_factory=dict, description='{Emissionstyp: Menge pro Output-Einheit}')

# --- Modelle für ScenarioConfig ---

class ResourceShockModel(BaseModel):
    """Modell für einen Ressourcenschock."""
    step: int = Field(..., ge=0, description="Simulationsschritt, in dem der Schock eintritt")
    resource: str = Field(..., description="Betroffene Ressource (muss in config.resources existieren)")
    relative_change: float = Field(..., description="Relative Änderung der Verfügbarkeit/Kapazität (z.B. -0.2 für -20%, 0.1 für +10%)")
    duration: Optional[int] = Field(None, ge=1, description="Dauer des Schocks in Schritten (None für permanent)")
    affected_regions: Optional[List[str]] = Field(None, description="Betroffene Regionen (Liste von Namen). None für globalen Effekt.")
    name: str = Field("Unnamed Resource Shock", description="Name des Ereignisses")
    description: str = Field("", description="Beschreibung des Ereignisses")
    is_random: bool = Field(False, description="Wurde dieses Event zufällig generiert?")

class TechnologyBreakthroughModel(BaseModel):
    """Modell für einen Technologie-Durchbruch."""
    step: int = Field(..., ge=0)
    affected_goods: List[str] = Field(..., description="Güter, deren Produktion beeinflusst wird (müssen in config.goods existieren)")
    # Faktor, um den die Basis-Effizienz der betroffenen Linien multipliziert wird (z.B. 1.1 für +10%)
    efficiency_boost_factor: float = Field(1.0, ge=0.0)
    # Faktor, um den der Bedarf an spezifischen Inputs reduziert wird (z.B. {"Energy": 0.9} für 10% weniger Energie)
    input_reduction_factors: Dict[str, float] = Field(default_factory=dict)
    # Faktor, um den spezifische Emissionen reduziert werden (z.B. {"co2": 0.8} für 20% weniger CO2)
    emission_reduction_factors: Dict[str, float] = Field(default_factory=dict)
    adoption_rate: float = Field(1.0, ge=0.0, le=1.0, description="Anteil der betroffenen Produzenten, die die Technologie sofort adaptieren")
    name: str = "Unnamed Tech Breakthrough"
    description: str = ""

class PolicyChangeModel(BaseModel):
    """Modell für eine Politikänderung."""
    step: int = Field(..., ge=0)
    # Typ der Politikänderung, z.B. "priority_weight", "resource_limit", "target_adjustment", "tax_rate", "subsidy"
    policy_type: str
    # Name des Parameters/Ziels, das geändert wird (Punktnotation für verschachtelte Configs).
    # z.B. "planning_priorities.fairness_weight", "environment_config.environmental_capacities.co2"
    target_parameter: str
    # Der neue absolute Wert für den Parameter.
    new_value: Any
    affected_scope: Optional[Union[str, List[str]]] = Field(None, description="Betroffener Bereich: Regionsname, Güterliste oder None für global")
    name: str = "Unnamed Policy Change"
    description: str = ""

# TODO: Weitere Modelle für Szenarien definieren (DemographicEventModel, TradeDisruptionModel etc.)

# --- Modelle für Agenten-Konfiguration in SimulationConfig ---

class AgentParameterConfig(BaseModel):
    """
    Struktur für die Parameter eines Agenten oder einer Population.
    Unterstützt spezifische Werte oder Tupel/Listen für Randomisierung.
    """
    # Nimmt beliebige Parameter entgegen, die an den Agenten __init__ übergeben werden.
    # Beispiel: {"initial_capacity": (80.0, 120.0), "depreciation_rate": 0.02, "enable_rl": True}
    # Die Auflösung von Ranges/Listen erfolgt in EconomicModel._resolve_param
    params: Dict[str, Any] = Field(default_factory=dict)

class AgentPopulationConfig(BaseModel):
    """Konfiguration für eine Population von Agenten."""
    agent_class: str = Field(..., description='Klassenname des Agenten (z.B. "ProducerAgent")')
    count: int = Field(..., ge=0, description="Anzahl zu erstellender Agenten in dieser Population")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameter für den Agenten-__init__, können Ranges (Tuple) oder Listen (Auswahl) enthalten")
    # Region ID (int oder str, je nach Verwendung) -> Anzahl (int) oder Anteil (float)
    region_distribution: Optional[Dict[Union[int, str], Union[int, float]]] = Field(None, description="Verteilung auf Regionen {RegionID: Anzahl/Anteil}")
    id_prefix: Optional[str] = Field(None, description="Präfix für die unique_ids der Agenten dieser Population")

class SpecificAgentConfig(BaseModel):
    """Konfiguration für eine spezifische, individuelle Agenten-Instanz."""
    agent_class: str = Field(..., description='Klassenname des Agenten')
    unique_id: str = Field(..., description="Eindeutige ID für diesen spezifischen Agenten")
    # Enthält alle notwendigen Parameter für __init__, *inklusive* 'region_id' (falls relevant).
    # Kann auch Ranges/Listen enthalten, die dann aufgelöst werden.
    params: Dict[str, Any] = Field(..., description="Spezifische Parameter, inkl. region_id")

# --- Modell für Logging-Konfiguration ---

class LoggingConfig(BaseModel):
    """Konfiguration für das Logging-System."""
    # Log-Level als String (Groß-/Kleinschreibung egal)
    log_level: str = Field("INFO", description="Logging-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    log_to_console: bool = Field(True, description="Ob Logs auf der Konsole ausgegeben werden sollen")
    log_to_file: bool = Field(False, description="Ob Logs in eine Datei geschrieben werden sollen")
    log_file_path: str = Field("impaler_simulation.log", description="Pfad zur Log-Datei, wenn log_to_file=True")
    log_format: str = Field('%(asctime)s - [%(name)s] %(levelname)s - %(message)s', description="Formatierungsstring für Log-Nachrichten")

    @validator('log_level')
    def check_log_level(cls, level: str) -> str:
        """Validiert, ob der Log-Level-String gültig ist."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level.upper() not in valid_levels:
            raise ValueError(f"Ungültiger log_level '{level}'. Muss einer von {valid_levels} sein.")
        return level.upper() # Konvertiere zu Großbuchstaben für Konsistenz

# --- Neue Modelle für planwirtschaftliche Parameter ---

class PlanwirtschaftParams(BaseModel):
    """Strukturierte Parameter für eine detaillierte planwirtschaftliche Steuerung.
    Diese Klasse definiert die "Stellschrauben" für die Wirtschaftsplanung."""

    # --- GRUPPE 1: Strafen und Kosten (Penalties & Costs) ---

    underproduction_penalty: float = Field(
        1.0, ge=0.0,
        description="Globaler Straffaktor für jede Einheit Unterproduktion gegenüber dem Plan."
    )
    overproduction_penalty: float = Field(
        0.1, ge=0.0,
        description="Globaler Straffaktor für jede Einheit Überproduktion gegenüber dem Plan."
    )
    inventory_cost: float = Field(
        0.05, ge=0.0,
        description="Lagerkosten pro Einheit eines Gutes, das über den Plan hinaus produziert und nicht sofort verbraucht wird."
    )
    co2_penalties: Optional[Dict[str, float]] = Field(
        None,
        description="Gut-spezifische Strafkosten pro Einheit CO2-Äquivalent. Bsp: {'Stahl': 15.0, 'Landwirtschaft': 2.5}"
    )

    # --- GRUPPE 2: Boni und Anreize (Bonuses & Incentives) ---

    societal_bonuses: Optional[Dict[str, float]] = Field(
        None,
        description="Ein gesellschaftlicher Bonus (negative Kosten) pro produzierter Einheit eines Gutes, um dessen Herstellung zu fördern. Bsp: {'Medikamente': 0.3}"
    )

    # --- GRUPPE 3: Priorisierung und strukturelle Vorgaben ---

    priority_sectors: List[str] = Field(
        default_factory=list,
        description="Liste von Gütern/Sektoren, die bei der Ressourcenallokation und Planerfüllung priorisiert werden."
    )
    priority_sector_demand_factor: float = Field(
        1.0, ge=1.0,
        description="Faktor, um den der angemeldete Bedarf priorisierter Sektoren für die Planung 'verstärkt' wird, um eine bevorzugte Versorgung sicherzustellen."
    )
    priority_levels: Optional[Dict[str, int]] = Field(
        None,
        description="Feingranulare Prioritätsstufen pro Gut (z.B. 0=niedrig, 1=mittel, 2=hoch), die die Bedarfsanpassung beeinflussen können."
    )

    # --- GRUPPE 4: Dynamische Anpassungen ---

    seasonal_demand_weights: Optional[List[float]] = Field(
        None,
        description="Liste von globalen Nachfragemultiplikatoren für jede Saison, um saisonale Schwankungen zu modellieren. Bsp: [1.2, 1.0, 0.8, 1.1]"
    )

    # --- GRUPPE 5: Physische Limits ---

    production_limits: Optional[Dict[str, float]] = Field(
        None,
        description="Absolute, harte Produktions-Obergrenze für ein Gut pro Periode. Bsp: {'Stahl': 10000}"
    )
    
    # --- GRUPPE 6: Steuerungs-Flag ---
    
    planwirtschaft_objective: bool = Field(
        True,
        description="Wenn True, aktivieren die Agenten ihre erweiterte Kostenfunktion, die diese Parameter berücksichtigt."
    )

    class Config:
        extra = 'forbid'  # Verhindert Tippfehler in Konfigurationsdateien, da unbekannte Felder einen Fehler auslösen.
