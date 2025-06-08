# Impaler/governance/government.py
"""
Implementierung einer 'Regierungs'- oder 'Meta-Planungs'-Entität (analog zu VSM System 5).

Diese Klasse repräsentiert die oberste Ebene der Steuerung und Zielsetzung
in einer cybersozialistischen Plansimulation. Sie setzt globale Rahmenbedingungen,
wie z.B. Umweltlimits oder Fairness-Ziele, und interagiert mit System 4 (strategische Planung),
um sicherzustellen, dass die Gesamtstrategie mit den übergeordneten Zielen übereinstimmt.
Sie operiert ohne klassische fiskalische Instrumente (Steuern etc.).
"""

import logging
import random
import numpy as np
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Deque, Union

# Typ-Prüfung Imports
if TYPE_CHECKING:
    from ..core.model import EconomicModel
    from ..vsm.system4 import System4Planner
    # Importiere relevante Konfigurations- und Kommunikationsmodelle
    from ..core.config import EnvironmentConfig, PlanningPriorities
    from ..vsm.communication_contracts import PolicyDirective # Eigene Klasse definieren?

# Utils (Annahme: Pfad ist korrekt)
from ..utils.math_utils import gini_coefficient

class GovernmentAgent:
    """
    Zentraler Agent für globale Zielsetzung und Politik (analog VSM System 5).

    Legt Rahmenbedingungen wie Ressourcenlimits und Fairnessziele fest und
    passt diese basierend auf dem Systemzustand an. Interagiert primär mit
    System 4, um sicherzustellen, dass die strategische Planung die
    übergeordneten Ziele verfolgt.

    Attributes:
        unique_id (str): Eindeutige ID (z.B. "government").
        model (EconomicModel): Referenz zum Hauptmodell.
        logger (logging.Logger): Logger-Instanz.
        env_config (EnvironmentConfig): Referenz zur Umweltkonfiguration.
        prio_config (PlanningPriorities): Referenz zur Prioritätenkonfiguration.
        # Übergeordnete Ziele (Beispiele, werden aus Config gelesen)
        target_co2_reduction_rate (float): Jährliche Zielrate für CO2-Reduktion.
        min_target_fairness (float): Minimal angestrebter Fairness-Indikator (1-Gini).
        target_output_growth (Optional[float]): Angestrebtes Output-Wachstum (kann None sein).
        # Subsystem für Analyse
        intelligence_system ('GovernmentIntelligence'): Analysiert globale Trends.
    # Meta-Ziele für VSM-Rekonfiguration
    meta_goals: List[str] 
    """
    unique_id: str
    model: 'EconomicModel'
    logger: logging.Logger
    env_config: 'EnvironmentConfig' # type: ignore
    prio_config: 'PlanningPriorities' # type: ignore
    target_co2_reduction_rate: float
    min_target_fairness: float
    target_output_growth: Optional[float]
    intelligence_system: 'GovernmentIntelligence'
    meta_goals: List[str]

    def __init__(self, unique_id: str, model: 'EconomicModel', **kwargs):
        """
        Initialisiert den GovernmentAgent.

        Args:
            unique_id: Eindeutige ID.
            model: Referenz zum Hauptmodell.
            **kwargs: Zusätzliche Parameter (werden aktuell nicht direkt verwendet,
                      aber für Konsistenz mit Agenten-Erstellung beibehalten).
        """
        self.unique_id = unique_id
        self.model = model
        self.logger = model.logger.getChild('GovernmentAgent')

        # Hole relevante Konfigurationen aus dem Hauptmodell
        # Annahme: Diese Config-Objekte existieren im Model
        self.env_config = getattr(model.config, 'environment_config', None)
        self.prio_config = getattr(model.config, 'planning_priorities', None)
        gov_params = getattr(model.config, 'government_params', {}) # Spezifische Params für Gov

        if not self.env_config or not self.prio_config:
            self.logger.error("EnvironmentConfig oder PlanningPriorities fehlen in der Modellkonfiguration! GovernmentAgent kann nicht korrekt initialisieren.")
            # Setze Dummy-Werte, um Abstürze zu vermeiden
            self.env_config = type('DummyEnvCfg', (object,), {'environmental_capacities': {}, 'sustainability_targets': {}})() # type: ignore
            self.prio_config = type('DummyPrioCfg', (object,), {'fairness_weight': 1.0, 'co2_weight': 1.0, 'min_fairness': 0.7})() # type: ignore

        # Lese Ziele aus der Konfiguration (oder spezifischen Gov-Params)
        self.target_co2_reduction_rate = float(gov_params.get('target_co2_reduction_rate', 0.02)) # z.B. 2% pro Jahr (angepasst an Steps)
        self.min_target_fairness = float(getattr(self.prio_config, 'min_fairness', 0.7)) # Lese aus Prio-Config
        self.target_output_growth = float(gov_params.get('target_output_growth')) if 'target_output_growth' in gov_params else None
        
        # Meta-Ziele für VSM-Rekonfiguration
        self.meta_goals = list(gov_params.get('meta_goals', ["increase_system_resilience"])) # Default, falls nicht in Config

        # Intelligence Subsystem
        self.intelligence_system = GovernmentIntelligence(self)

        self.logger.info(f"GovernmentAgent '{self.unique_id}' initialisiert.")
        self.logger.info(f"  Ziele - CO2 Reduktion Rate: {self.target_co2_reduction_rate:.2%}, Min. Fairness (1-Gini): {self.min_target_fairness:.2f}")

    def handle_crisis_start(self, crisis_type: str, effects: Dict[str, Any]) -> None:
        """Placeholder reaction to crisis start events."""
        self.logger.warning(f"System 5 registriert Beginn einer Krise: {crisis_type}")

    def handle_crisis_end(self, crisis_type: str) -> None:
        """Placeholder reaction to crisis end events."""
        self.logger.info(f"System 5 registriert Ende der Krise: {crisis_type}")

    def step_stage(self, stage: str) -> None:
        """
        Führt Aktionen für die Regierungs-/Politik-Phase aus.

        Args:
            stage: Name der aktuellen Phase (z.B. "system5_policy").
        """
        if stage == "system5_policy":
            self.logger.debug("System 5 Politik-Phase: Formuliere Direktiven für System 4...")

            s4_planner = self.model.system4planner
            if not s4_planner:
                self.logger.error("System 4 Planner nicht gefunden. Kann Politik nicht anwenden.")
                return

            plan_params = self.model.config.planning_priorities.planwirtschaft_params
            updates = {"lambdas": {}, "priorities": {}}

            is_env_crisis = (
                self.model.crisis_manager.crisis_active
                and self.model.crisis_manager.current_crisis.get_type() == "environmental_catastrophe"
            ) if getattr(self.model, "crisis_manager", None) else False
            co2_penalty_multiplier = 2.0 if is_env_crisis else 1.0

            if plan_params.co2_penalties:
                for good, penalty in plan_params.co2_penalties.items():
                    lambda_key = f"co2_{good}"
                    updates["lambdas"][lambda_key] = penalty * co2_penalty_multiplier

            under_penalties = getattr(plan_params, "underproduction_penalties", None)
            if under_penalties:
                for i, penalty in enumerate(under_penalties):
                    if i < len(self.model.config.goods):
                        good_name = self.model.config.goods[i]
                        updates["lambdas"][f"underproduction_{good_name}"] = penalty
            else:
                for good_name in self.model.config.goods:
                    updates["lambdas"][f"underproduction_{good_name}"] = plan_params.underproduction_penalty

            over_penalties = getattr(plan_params, "overproduction_penalties", None)
            if over_penalties:
                for good, penalty in over_penalties.items():
                    updates["lambdas"][f"overproduction_{good}"] = penalty
            else:
                for good_name in self.model.config.goods:
                    updates["lambdas"][f"overproduction_{good_name}"] = plan_params.overproduction_penalty

            if plan_params.inventory_cost:
                for good_name in self.model.config.goods:
                    updates["lambdas"][f"inventory_{good_name}"] = plan_params.inventory_cost

            if plan_params.societal_bonuses:
                for good, bonus in plan_params.societal_bonuses.items():
                    updates["lambdas"][f"societal_bonus_{good}"] = -bonus

            updates["priorities"]["fairness_weight"] = self.prio_config.fairness_weight
            updates["priorities"]["resilience_weight"] = self.prio_config.resilience_weight

            s4_planner.update_policy_parameters(updates)

    def _perform_vsm_reconfiguration(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analysiert den Systemzustand und Meta-Ziele, um VSM-Strukturen anzupassen.
        Passt z.B. Autonomielevel von System 3 an oder initiiert Änderungen
        in der Verantwortung von VSM-Schichten.

        Args:
            summary: Zusammenfassung des aktuellen Zustands vom Intelligence-System.

        Returns:
            Dictionary mit Policy-Updates, die aus der Rekonfiguration resultieren.
        """
        reconfig_updates: Dict[str, Any] = {}
        self.logger.debug(f"Starte VSM Rekonfiguration basierend auf Meta-Zielen: {self.meta_goals}")

        # --- 1. Performance Analyse Integration ---
        # `summary` ist bereits der Output von GovernmentIntelligence.
        active_crises = getattr(self.model, 'crisis_manager', None)
        active_crises_info = active_crises.get_active_crises_info() if active_crises else {} # Annahme: Methode existiert
        # Beispiel: active_crises_info = {"resource_shortage": {"severity": "high", "affected_regions": [...]}}

        # --- 2. Dynamische Anpassung: Autonomielevel System 3 ---
        s3_manager = getattr(self.model, 'system3manager', None)
        if s3_manager and hasattr(s3_manager, 'autonomy_level') and hasattr(s3_manager, 'set_autonomy_level'):
            current_autonomy = s3_manager.autonomy_level
            new_autonomy = current_autonomy

            # Kriterien für Anpassung:
            # - Persistente Ineffizienzen (z.B. hohe Planungsfehler) -> Reduziere Autonomie
            #   Annahme: summary['planning']['s3_error_rate'] existiert oder kann abgeleitet werden
            s3_error_rate = summary.get('planning', {}).get('s3_error_rate', 0.0) # Placeholder
            if s3_error_rate > 0.2: # Schwellwert für "hohe Fehlerquote"
                new_autonomy = max(0.3, current_autonomy * 0.9) # Reduziere um 10%, min 0.3
                self.logger.info(f"Hohe S3 Fehlerquote ({s3_error_rate:.2f}), reduziere S3 Autonomie von {current_autonomy:.2f} auf {new_autonomy:.2f}.")
            elif active_crises_info and s3_error_rate < 0.05: # Gute Performance unter Stress -> Erhöhe Autonomie
                new_autonomy = min(1.0, current_autonomy * 1.1) # Erhöhe um 10%, max 1.0
                self.logger.info(f"System stabil unter Krise(n) und S3 Performance gut. Erhöhe S3 Autonomie von {current_autonomy:.2f} auf {new_autonomy:.2f}.")

            if abs(new_autonomy - current_autonomy) > 0.01:
                try:
                    s3_manager.set_autonomy_level(new_autonomy) # Annahme: Diese Methode existiert in System3Manager
                    self.logger.info(f"System3Manager Autonomielevel auf {new_autonomy:.2f} gesetzt.")
                    # Optional: Erstelle eine Direktive für diese Änderung (für Logging/Nachvollziehbarkeit)
                    # from ..vsm.communication_contracts import VSMReconfigurationDirective # Placeholder
                    # directive = VSMReconfigurationDirective(source="S5", target="S3", changes={"autonomy_level": new_autonomy})
                    # self.model.communication_hub.send(directive)
                except Exception as e:
                    self.logger.error(f"Fehler beim Setzen des S3 Autonomielevels: {e}")
        else:
            self.logger.warning("System3Manager oder Methode 'set_autonomy_level' nicht gefunden für VSM Rekonfiguration.")


        # --- 3. Meta-Goal Verarbeitung ---
        for goal in self.meta_goals:
            if goal == "increase_system_resilience":
                if active_crises_info:
                    self.logger.info(f"Meta-Ziel 'increase_system_resilience': Aktive Krisen detektiert ({list(active_crises_info.keys())}).")
                    # Erwäge Anpassung von Prioritäten für Resilienz, z.B. Fairness erhöhen
                    # Dies kann als Policy-Update an S4 gehen.
                    current_fairness_weight = getattr(self.prio_config, 'fairness_weight', 1.0)
                    new_fairness_weight = min(3.0, current_fairness_weight * 1.05) # Leicht erhöhen
                    reconfig_updates['planning_priorities.fairness_weight'] = new_fairness_weight
                    self.logger.info(f"  -> Erhöhe 'fairness_weight' für Resilienz auf {new_fairness_weight:.2f}.")

            elif goal == "accelerate_technological_change":
                self.logger.info("Meta-Ziel 'accelerate_technological_change':")
                # Dies würde typischerweise S4 beeinflussen, z.B. durch:
                # - Erhöhung der Gewichtung für R&D in Planungszielen.
                # - Spezifische Direktiven an S4 zur Förderung von Innovationsprojekten.
                # Beispiel: Erhöhe Gewichtung für einen 'tech_innovation_score' in S4 Prioritäten
                # current_tech_weight = getattr(self.prio_config, 'tech_innovation_weight', 0.5)
                # new_tech_weight = min(2.0, current_tech_weight * 1.1)
                # reconfig_updates['planning_priorities.tech_innovation_weight'] = new_tech_weight
                # self.logger.info(f"  -> (Konzept) Erhöhe 'tech_innovation_weight' auf {new_tech_weight:.2f}.")
                self.logger.info("  -> (Konzept) Sende Direktive an S4 zur Priorisierung von R&D.")
                # Placeholder für Direktive:
                # tech_directive = PolicyDirective(
                #    source_system="S5", target_system="S4",
                #    directive_type="strategic_focus",
                #    payload={"focus_area": "technological_innovation", "intensity": "high"}
                # )
                # self.model.communication_hub.send(tech_directive) # Annahme CommHub existiert

        # --- 4. Kommunikation Channel / VSM Layer Responsibility (Konzeptionell) ---
        self.logger.debug("Überlegungen zu komplexeren VSM-Rekonfigurationen:")
        # A. Dynamische Kommunikationskanäle:
        #    - Könnte bedeuten, dass Agenten Discovery-Mechanismen anpassen (z.B. Priorisierung bestimmter Informationsquellen).
        #    - Änderung der Struktur/Inhalte von 'CommunicationContracts'.
        #    - Beispiel: Wenn S4 überlastet ist, könnte S5 entscheiden, dass S3-Regionen
        #      temporär bestimmte Planungsaufgaben (normalerweise S4) selbst übernehmen (erfordert neue Contracts für S3).
        # B. VSM Layer Verantwortlichkeiten:
        #    - Verlagerung von Entscheidungsbefugnissen zwischen Ebenen.
        #    - Beispiel: Bei einer schweren globalen Krise könnte S5 direkt operatives Management
        #      in bestimmten Sektoren übernehmen, statt über S4/S3 zu gehen.
        #      Dies würde erfordern, dass S5 Direktiven an S2/S1 senden kann, die normalerweise
        #      von S3 kommen.
        #    - Konkreter erster Schritt: Ein Konfigurationsparameter, der bestimmt, welche Systemebene
        #      für eine bestimmte Art von Entscheidung zuständig ist (z.B. 'resource_allocation_authority: S4' vs 'S3_regional').
        #      S5 könnte diesen Parameter basierend auf der Situation ändern.

        if reconfig_updates:
            self.logger.info(f"VSM-Rekonfigurations-Updates bestimmt: {reconfig_updates}")
        return reconfig_updates

    def _evaluate_goals_and_determine_updates(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bewertet den aktuellen Systemzustand im Vergleich zu den Zielen und
        entscheidet über notwendige Anpassungen an Politikparametern.

        Args:
            summary: Zusammenfassung des aktuellen Zustands vom Intelligence-System.

        Returns:
            Dictionary mit den zu aktualisierenden Parametern für System 4.
            Format: {'config_path.parameter_name': neuer_wert}
        """
        updates: Dict[str, Any] = {}
        if not summary or not self.prio_config or not self.env_config:
             self.logger.warning("Keine Zusammenfassung oder Konfiguration für Zielbewertung verfügbar.")
             return updates

        self.logger.debug("Bewerte Ziele und bestimme Politik-Updates...")

        # --- Ziel 1: CO2 / Umwelt ---
        current_co2 = summary.get('environmental', {}).get('total_co2', None)
        co2_limit = self.env_config.environmental_capacities.get('co2', float('inf'))
        co2_target = self.env_config.get_current_sustainability_target('co2', self.model.current_step) # type: ignore

        if current_co2 is not None:
             # A. Ist das Limit überschritten?
             if current_co2 > co2_limit * 0.95: # Knapp unter Limit
                  old_weight = getattr(self.prio_config, 'co2_weight', 1.0)
                  new_weight = min(5.0, old_weight * 1.15) # Erhöhe Gewichtung stärker
                  updates['planning_priorities.co2_weight'] = new_weight
                  self.logger.warning(f"CO2-Limit ({co2_limit}) fast erreicht ({current_co2:.1f}). Erhöhe CO2-Gewichtung auf {new_weight:.2f}.")
             # B. Wird das Nachhaltigkeitsziel verfehlt?
             elif co2_target is not None and current_co2 > co2_target * 1.05: # Ziel verfehlt
                  old_weight = getattr(self.prio_config, 'co2_weight', 1.0)
                  new_weight = min(5.0, old_weight * 1.05) # Erhöhe Gewichtung leicht
                  updates['planning_priorities.co2_weight'] = new_weight
                  self.logger.info(f"CO2-Nachhaltigkeitsziel ({co2_target}) verfehlt ({current_co2:.1f}). Erhöhe CO2-Gewichtung auf {new_weight:.2f}.")
             # C. Liegen wir weit unter dem Ziel? Gewichtung ggf. leicht senken? (Optional)
             # elif co2_target is not None and current_co2 < co2_target * 0.7:
             #      # Senke Gewichtung leicht, um andere Ziele zu ermöglichen
             #      updates['planning_priorities.co2_weight'] = max(0.5, self.prio_config.co2_weight * 0.98)


        # --- Ziel 2: Fairness ---
        current_gini = summary.get('welfare', {}).get('satisfaction_gini', # Nutze Zufriedenheits-Gini? Oder Output-Gini?
                          summary.get('welfare', {}).get('production_gini', None))
        if current_gini is not None:
             current_fairness = 1.0 - current_gini
             if current_fairness < self.min_target_fairness:
                  old_weight = getattr(self.prio_config, 'fairness_weight', 1.0)
                  # Erhöhe Gewichtung stärker, je größer die Lücke
                  increase_factor = 1.0 + (self.min_target_fairness - current_fairness) * 0.5
                  new_weight = min(4.0, old_weight * increase_factor)
                  updates['planning_priorities.fairness_weight'] = new_weight
                  self.logger.warning(f"Fairness-Ziel ({self.min_target_fairness:.2f}) unterschritten (Aktuell: {current_fairness:.2f}). Erhöhe Fairness-Gewichtung auf {new_weight:.2f}.")
             # Optional: Gewichtung senken, wenn Fairness sehr hoch ist?

        # --- Ziel 3: Output / Wachstum (falls definiert) ---
        # TODO: Implementiere Logik für Output-Ziele, z.B. Anpassung von
        # globalen Zielvorgaben oder Effizienzgewichtung. (Bleibt im Standard Policy Review)

        if updates:
             self.logger.info(f"Standard Politik-Updates bestimmt: {updates}")
        else:
             self.logger.debug("Keine notwendigen Standard Politik-Updates identifiziert.")

        return updates

    def _send_policy_updates_to_system4(self, updates: Dict[str, Any]) -> None:
        """Sendet die beschlossenen Parameter-Updates an System 4."""
        if not updates: return

        s4_planner = getattr(self.model, 'system4planner', None)
        if s4_planner and hasattr(s4_planner, 'update_policy_parameters'):
            try:
                s4_planner.update_policy_parameters(updates)
                self.logger.info(f"Politik-Updates erfolgreich an System 4 gesendet: {list(updates.keys())}")
            except Exception as e:
                self.logger.error(f"Fehler beim Senden der Politik-Updates an System 4: {e}", exc_info=True)
        else:
            self.logger.error("System 4 oder dessen 'update_policy_parameters'-Methode nicht gefunden. Updates können nicht gesendet werden.")

    def handle_escalation(self, message: Union['ConflictResolutionDirectiveS2', 'SystemFeedback']) -> None: # type: ignore
         """Behandelt Eskalationen, die von untergeordneten Systemen kommen."""
         # Wird aufgerufen, wenn z.B. S2 eine 'request_intervention' Direktive sendet.
         payload = getattr(message, 'payload', {})
         self.logger.critical(f"!!! Eskalation von {getattr(message, 'source_system', 'Unknown')} empfangen !!! Typ: {payload.get('conflict_type')}, Regionen: {payload.get('regions')}. Grund: {payload.get('description', 'Keine Angabe')}")
         
         # Hier könnten auch VSM-Rekonfigurationslogiken getriggert werden,
         # z.B. temporäre Übernahme von Kontrolle oder drastische Autonomieänderungen.
         # Beispiel: Bei schwerem Umweltkonflikt nicht nur CO2-Gewichtung an S4 ändern,
         # sondern auch S3-Autonomie für betroffene Regionen reduzieren und direkte Vorgaben machen.

         # TODO: Implementiere Logik für System 5 Interventionen:
         # - Setze harte Limits für bestimmte Regionen/Producer.
         # - Ändere globale Prioritäten drastisch.
         # - Löse spezifische Szenario-Events aus (z.B. Investitionsprogramm).
         # - Sende Notfall-Direktiven direkt an System 3.
         # Beispiel: Erhöhe CO2-Gewichtung massiv bei Umweltkonflikt
         if payload.get('conflict_type') == 'emission_conflict':
              updates = {'planning_priorities.co2_weight': getattr(self.prio_config, 'co2_weight', 1.0) * 1.5}
              self._send_policy_updates_to_system4(updates)
              self.logger.warning(f"Eskalation 'emission_conflict': CO2 Gewichtung auf {updates['planning_priorities.co2_weight']:.2f} erhöht.")
              # Ggf. auch S3 Autonomie anpassen für die betroffenen Regionen, falls Info verfügbar.


# --- Government Intelligence Subsystem ---

# Placeholder für eine mögliche Kommunikationsdirektive für VSM Änderungen
# class VSMReconfigurationDirective:
#     def __init__(self, source: str, target: str, changes: Dict[str, Any]):
#         self.source_system = source
#         self.target_system = target
#         self.directive_type = "vsm_reconfiguration"
#         self.payload = changes
#         self.timestamp = # current model time

class GovernmentIntelligence:
    """
    Subsystem für Analyse und Prognose auf globaler Ebene (System 5).

    Sammelt und analysiert High-Level-Daten aus dem DataCollector oder direkt
    vom Modell, um dem GovernmentAgent Entscheidungsgrundlagen zu liefern.
    Fokus auf Makro-Indikatoren, Zielerreichung und Systemstabilität.
    """
    government: 'GovernmentAgent'
    model: 'EconomicModel'
    logger: logging.Logger
    analysis_history: Deque[Dict[str, Any]]
    # Konfiguration für Analyse (z.B. Fenstergrößen)
    config: Dict[str, Any]

    def __init__(self, government: 'GovernmentAgent'):
        """Initialisiert das Intelligence-Subsystem."""
        self.government = government
        self.model = government.model
        self.logger = government.logger.getChild('Intelligence')
        self.analysis_history = deque(maxlen=getattr(self.model.config, "intelligence_history_length", 100))
        self.config = getattr(self.model.config, "intelligence_params", {"trend_window": 10})
        self.logger.debug("GovernmentIntelligence Subsystem initialisiert.")

    def update(self) -> Dict[str, Any]:
        """
        Führt die Analyse des aktuellen Systemzustands durch.

        Returns:
            Ein Dictionary mit einer Zusammenfassung der Analyseergebnisse.
        """
        self.logger.debug("Starte globale Zustandsanalyse...")
        summary: Dict[str, Any] = {}

        # Greife auf Daten vom DataCollector oder direkt vom Modell zu
        # Annahme: model.detailed_metrics enthält die aktuellsten aggregierten Daten
        latest_metrics = getattr(self.model, 'detailed_metrics', {})
        if not latest_metrics:
             self.logger.warning("Keine aktuellen Metriken im Modell gefunden für Analyse.")
             return summary

        # Extrahiere und fasse relevante Metriken zusammen
        summary['step'] = latest_metrics.get('step', -1)
        summary['production'] = latest_metrics.get('production_metrics', {})
        summary['environmental'] = latest_metrics.get('environmental', {})
        summary['welfare'] = latest_metrics.get('welfare', {})
        summary['technology'] = latest_metrics.get('technology', {})
        summary['planning'] = latest_metrics.get('planning', {}) # Wichtig für S3 Fehlerquote
        summary['crisis'] = latest_metrics.get('crisis', {}) # Wird für Kriseninfo genutzt

        # Berechne Trends (optional)
        summary['trends'] = self._calculate_trends()

        # Speichere Analyse in History
        self.analysis_history.append(summary)
        self.logger.debug("Globale Zustandsanalyse abgeschlossen.")
        return summary

    def _calculate_trends(self) -> Dict[str, str]:
        """Berechnet einfache Trends für Schlüsselmetriken."""
        trends: Dict[str, str] = {}
        window = self.config.get("trend_window", 5) # Nutze Config

        # Nutze Zeitreihen vom DataCollector
        dc = getattr(self.model, 'data_collector', None)
        if not dc or not hasattr(dc, 'time_series'):
            return trends

        for metric, ts in dc.time_series.items():
            if len(ts) >= window:
                 # Einfacher Vergleich: letzter Wert vs. Durchschnitt der letzten 'window' Werte
                 recent_values = [v for _, v in list(ts)[-window:]]
                 if not recent_values: continue
                 current_value = recent_values[-1]
                 avg_value = np.mean(recent_values)
                 std_dev = np.std(recent_values)
                 
                 # Definiere EPSILON hier oder importiere es, falls es global verfügbar ist
                 # Für den Moment definieren wir es lokal für die Funktion.
                 FUNC_EPSILON = 1e-9 

                 if std_dev > FUNC_EPSILON: # Nur wenn es Varianz gibt
                      if current_value > avg_value + 0.5 * std_dev:
                           trends[metric] = "increasing"
                      elif current_value < avg_value - 0.5 * std_dev:
                           trends[metric] = "decreasing"
                      else:
                           trends[metric] = "stable"
                 else: # Keine Varianz
                      trends[metric] = "stable"

        self.logger.debug(f"Berechnete Trends: {trends}")
        return trends

    def get_latest_analysis(self) -> Optional[Dict[str, Any]]:
        """Gibt die letzte Analyse zurück."""
        return self.analysis_history[-1] if self.analysis_history else None
