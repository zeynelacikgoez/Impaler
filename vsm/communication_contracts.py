# Impaler/vsm/communication_contracts.py
"""
Definition von Datenstrukturen (Contracts) für die Kommunikation
zwischen den verschiedenen Ebenen des Viable System Models (VSM).

Diese Modelle verwenden Pydantic zur Sicherstellung von Typ-Sicherheit,
Validierung und klarer Definition der ausgetauschten Informationen.
Sie dienen als "Verträge" zwischen den Systemen S2, S3, S4 und S5.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Literal

import enum # Added import for Enum

# Pydantic für Datenvalidierung und -modellierung importieren
try:
    from pydantic import BaseModel, Field, validator
except ImportError:
    # Fallback, falls Pydantic nicht installiert ist - Simulation wird wahrscheinlich nicht korrekt laufen
    logging.error("Pydantic ist nicht installiert. VSM Communication Contracts können nicht definiert werden.")
    # Definiere Dummy-BaseModel, um Import-Fehler in anderen Modulen zu vermeiden
    class BaseModel: pass
    def Field(*args, **kwargs): return None
    def validator(*args, **kwargs): return lambda f: f


class GovernanceMode(enum.Enum):
    """
    Represents the governance mode, indicating overall system health or operational status.
    GREEN: Nominal operation.
    YELLOW: Caution, potential issues or degraded performance.
    RED: Critical, significant issues requiring immediate attention.
    """
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class BaseVSMMessage(BaseModel):
    """Basisklasse für alle VSM-Kommunikationsnachrichten."""
    step: int = Field(..., description="Simulationsschritt, in dem die Nachricht erstellt wurde")
    source_system: str = Field(..., description="Name des sendenden Systems (z.B. 'System3', 'System2')")
    target_system: str = Field(..., description="Name des empfangenden Systems (z.B. 'System4', 'System3')")
    message_id: Optional[str] = Field(None, description="Optionale eindeutige ID der Nachricht")

    class Config:
        extra = 'forbid' # Keine zusätzlichen Felder erlauben

# --- S3 -> S4 Kommunikation ---

class RegionalStatusInfo(BaseModel):
    """Zusammenfassender Status einer einzelnen Region."""
    stress_level: Optional[float] = Field(None, ge=0.0, le=1.0)
    resource_utilization: Optional[Dict[str, float]] = Field(default_factory=dict) # {resource: utilization_ratio}
    production_fulfillment: Optional[Dict[str, float]] = Field(default_factory=dict) # {good: fulfillment_ratio}
    # Weitere regionale KPIs hier...

class OperationalReportS3(BaseVSMMessage):
    """
    Operativer Bericht von System 3 an System 4.

    Enthält aggregierte Daten über den Zustand der operativen Ebene,
    Performance-Metriken und identifizierte Probleme.
    """
    source_system: str = "System3"
    target_system: str = "System4"
    coordination_effectiveness: float = Field(..., ge=0.0, le=1.0, description="Gesamteffektivität der S3-Koordination")
    critical_resources: List[str] = Field(default_factory=list, description="Liste aktuell als kritisch eingestufter Ressourcen")
    resource_levels: Optional[Dict[str, float]] = Field(default_factory=dict, description="Globale Ressourcenbestände (Momentaufnahme)")
    production_statistics: Dict[str, Any] = Field(default_factory=dict, description="Aggregierte Produktionsdaten (z.B. output_by_good, capacity_utilization)")
    regional_status: Dict[Union[int, str], RegionalStatusInfo] = Field(default_factory=dict, description="Zusammenfassender Status pro Region")
    bottlenecks: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Identifizierte Engpässe im Produktionsnetzwerk")
    plan_fulfillment_summary: Dict[str, float] = Field(default_factory=dict, description="Durchschnittlicher Planerfüllungsgrad pro Gut")
    # Weitere Felder nach Bedarf...

# --- S4 -> S3 Kommunikation ---

class StrategicDirectiveS4(BaseVSMMessage):
    """
    Strategische Direktiven von System 4 an System 3.

    Enthält Produktionsziele, Richtlinien für Ressourcenallokation,
    Prioritäten und andere Vorgaben für die operative Ebene.
    """
    source_system: str = "System4"
    target_system: str = "System3"
    # Globale Produktionsziele für die nächste Periode
    production_targets: Dict[str, float] = Field(default_factory=dict)
    # Richtlinien für Ressourcenverteilung (kann Prioritäten, Fairness-Ziele etc. enthalten)
    resource_allocation_guidelines: Dict[str, Any] = Field(default_factory=dict)
    # Optional: Update der Liste kritischer Güter
    critical_goods_update: Optional[List[str]] = None
    # Optional: Anpassung des Autonomiegrads von System 3
    system3_autonomy_level: Optional[float] = Field(None, ge=0.0, le=1.0)
    # Optional: Vorgaben für Investitionen oder Technologie-Fokus
    investment_directives: Optional[Dict[str, Any]] = None
    # Optional: Aktualisierte globale Prioritäten für Güter/Ressourcen
    updated_priorities: Optional[Dict[str, Dict[str, float]]] = None # z.B. {'goods': {...}, 'resources': {...}}

# --- S2 -> S3/S4/S5 Kommunikation ---

# Definiere mögliche Payload-Typen für mehr Typsicherheit (optional)
class ResourceTransferPayload(BaseModel):
    resource: str
    amount: float = Field(..., gt=0)
    from_region: Union[int, str]
    to_region: Union[int, str]

class PlanAdjustmentRequestPayload(BaseModel):
    conflict_type: str
    regions: List[Union[int, str]]
    severity: Optional[float] = None
    rationale: str
    suggested_changes: Optional[Dict[str, Any]] = None

class InterventionRequestPayload(BaseModel):
    conflict_type: str
    regions: List[Union[int, str]]
    severity: Optional[float] = None
    description: str

# Typ für den action_type definieren
ConflictActionType = Literal[
    "resource_transfer",
    "request_plan_adjustment",
    "request_intervention",
    "negotiation_result", # Ergebnis einer Verhandlung
    "side_payment_executed" # Info über durchgeführtes Side Payment
]

class ConflictResolutionDirectiveS2(BaseVSMMessage):
    """
    Direktive oder Feedback von System 2 bezüglich eines gelösten oder
    eskalierten Konflikts.

    Wird an das zuständige System (S3, S4 oder S5) zur Ausführung
    oder Kenntnisnahme gesendet.
    """
    source_system: str = "System2"
    # target_system wird je nach action_type gesetzt (z.B. S3 für Transfer, S4 für Plan)
    action_type: ConflictActionType = Field(..., description="Art der durchzuführenden Aktion oder des Feedbacks")
    # Payload enthält die spezifischen Daten für die Aktion.
    # Die Struktur hängt vom action_type ab.
    # Besser: payload: Union[ResourceTransferPayload, PlanAdjustmentRequestPayload, ...]
    # Hier: Dict[str, Any] für Flexibilität, mit Kommentaren zur erwarteten Struktur.
    payload: Dict[str, Any] = Field(...)
    conflict_id: Optional[str] = Field(None, description="ID des ursprünglichen Konflikts (zur Nachverfolgung)")

    @validator('payload')
    def check_payload_structure(cls, payload, values):
        """Prüft grob, ob der Payload zum Action-Typ passt (Beispiel)."""
        action = values.get('action_type')
        required_keys = []
        if action == "resource_transfer":
            required_keys = ["resource", "amount", "from_region", "to_region"]
        elif action == "request_plan_adjustment":
            required_keys = ["conflict_type", "regions", "rationale"]
        elif action == "request_intervention":
            required_keys = ["conflict_type", "regions", "description"]

        missing_keys = [key for key in required_keys if key not in payload]
        if missing_keys:
            raise ValueError(f"Payload für Action '{action}' fehlen erforderliche Schlüssel: {missing_keys}")
        return payload

# --- S5 -> S4 Kommunikation ---

PolicyTargetParameter = Literal[
    "planning_priorities.fairness_weight",
    "planning_priorities.co2_weight",
    "planning_priorities.efficiency_weight",
    "planning_priorities.resilience_weight",
    "planning_priorities.goods_priority", # Spezifisches Gut muss im Payload sein
    "environment_config.environmental_capacities", # Spez. Impact muss im Payload sein
    "environment_config.sustainability_targets", # Spez. Impact muss im Payload sein
    # ... weitere mögliche Parameterpfade ...
]

class PolicyDirectiveS5(BaseVSMMessage):
    """
    Politische Direktive von System 5 an System 4 (oder andere).

    Enthält Anpassungen an globalen Zielen, Gewichtungen oder Limits.
    """
    source_system: str = "System5"
    target_system: str = "System4" # Hauptempfänger ist oft S4
    # Welcher Parameter soll geändert werden? (Punktnotation für verschachtelte Objekte)
    target_parameter: PolicyTargetParameter
    # Der neue Wert für den Parameter. Kann ein einfacher Wert oder ein Dict sein.
    new_value: Any
    # Optional: Spezifischer Schlüssel für Dict-Parameter (z.B. Gut-Name für goods_priority)
    parameter_key: Optional[str] = None
    rationale: Optional[str] = Field(None, description="Grund für die Änderung")


# --- Generische Feedback-Nachricht ---

class SystemFeedback(BaseVSMMessage):
    """
    Generische Nachricht für Feedback zwischen Systemen, das keine direkte
    Direktive ist (z.B. Status-Updates, Warnungen).
    """
    feedback_type: str = Field(..., description="Art des Feedbacks (z.B. 'low_performance_warning', 'resource_trend')")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Detaillierte Feedback-Daten")