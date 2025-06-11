# Impaler/agents/consumer.py

"""
Erweiterte ConsumerAgent-Implementierung für eine planwirtschaftliche Simulation.

Dieser Agent repräsentiert einen Endnutzer/Haushalt in einer cybersozialistischen
Planwirtschaft ohne Marktmechanismen. Fokus auf dynamische Bedarfe, Substitution,
Lernen und Zufriedenheit basierend auf Planerfüllung. Enthält auch den
CommunityConsumerAgent für kollektives Verhalten.
"""

import random
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, Optional, List, Tuple, Set, Union, Callable, Deque, DefaultDict, TYPE_CHECKING
from enum import Enum
import math
import logging

# Typ-Prüfung Imports
if TYPE_CHECKING:
    from ..core.model import EconomicModel
    from ..core.config import DemandConfig, ScenarioConfig
else:  # Laufzeit-Import fuer Defaultwerte
    from ..core.config import DemandConfig, ScenarioConfig

# Logger-Setup (wird vom Model geholt)
# logger = logging.getLogger(__name__) # Nicht mehr nötig

# --- Enums ---

class NeedPriority(Enum):
    """Prioritätsstufen von Bedürfnissen."""
    ESSENTIAL = 100
    HIGH = 80
    MEDIUM = 60
    LOW = 40
    LUXURY = 20

class SubstitutionType(Enum):
    """Typen von Substitutionsbeziehungen (aktuell nicht direkt verwendet, Faktor reicht)."""
    PERFECT = 1.0
    GOOD = 0.8
    PARTIAL = 0.5
    LIMITED = 0.3
    EMERGENCY = 0.1

# --- Consumer Agent Klasse ---

class ConsumerAgent:
    """
    Fortschrittlicher ConsumerAgent für planwirtschaftliche Simulationen.

    Attributes:
        unique_id (str): Eindeutige Kennung.
        model (EconomicModel): Referenz zum Hauptmodell.
        logger (logging.Logger): Logger-Instanz.
        region_id (Union[int, str]): Region des Agenten.
        demographic_type (str): Demographischer Typ (beeinflusst Bedarfe).
        community_id (Optional[str]): ID der Gemeinschaft, falls Teil einer.
        config_params (Dict[str, Any]): Agentenspezifische Konfigurations-Overrides.

        initial_base_needs (Dict[str, float]): Ursprüngliche Basisbedarfe.
        base_needs (Dict[str, float]): Aktuell angepasste Basisbedarfe.
        preference_weights (Dict[str, float]): Gewichtung der Güter im Nutzen.
        need_priorities (Dict[str, NeedPriority]): Priorität der Bedarfsdeckung.
        substitution_pairs (Dict[Tuple[str, str], float]): Substitutionsmöglichkeiten { (Original, Ersatz): Faktor }.
        seasonal_factors (Dict[str, Dict[str, float]]): Saisonale Bedarfsanpassungen.
        external_event_factors (Dict[str, Dict[str, float]]): Auswirkungen externer Events.
        active_events (Set[str]): Aktuell aktive externe Ereignisse.

        consumption_received (DefaultDict[str, float]): Zugewiesene Güter in diesem Step.
        consumption_actual (DefaultDict[str, float]): Tatsächlich konsumierte Güter (nach Substitution/Rationierung).
        fulfillment_ratios (DefaultDict[str, float]): Erfüllungsgrad pro Gut (actual/need).
        satisfaction (float): Gesamtzufriedenheit im letzten Step.

        # Historien
        consumption_history (Deque[Dict[str, float]]): Verlauf des Konsums.
        fulfillment_history (Deque[Dict[Dict[str, float]]]): Verlauf der Erfüllungsgrade.
        satisfaction_history (Deque[float]): Verlauf der Zufriedenheit.
        needs_history (Deque[Dict[str, float]]): Verlauf der angepassten Bedarfe.

        # Adaptionsparameter & Zustand
        adaptation_rate (float): Lerngeschwindigkeit für Bedarfsanpassung.
        preference_adaptation_rate (float): Lerngeschwindigkeit für Präferenzanpassung.
        substitution_learning_rate (float): Lerngeschwindigkeit für Substitutionsfaktoren.
        lifestyle_factor (float): Aktueller Lebensstil-Multiplikator.
        consecutive_shortage (DefaultDict[str, int]): Zähler für anhaltenden Mangel pro Gut.
        consecutive_surplus (DefaultDict[str, int]): Zähler für anhaltenden Überschuss pro Gut.
        failed_substitution_attempts (DefaultDict[Tuple[str, str], int]): Zähler für fehlgeschlagene Substitutionen.
        shortage_memory (float): "Erinnerung" an vergangene Knappheit.

        # Rationierung
        ration_active (bool): Ob Rationierung aktiv ist.
        ration_factor (float): Globaler Rationierungsfaktor (0-1).
        ration_strategy (str): Rationierungsstrategie ('priority', 'proportional', 'equal').

        # Nutzenfunktion
        elasticity_method (str): Methode ('cobb_douglas', 'ces', 'additive', 'simple_average').
        sigma (float): Substitutionselastizität für CES-Funktion.
    """
    unique_id: str
    model: 'EconomicModel'
    logger: logging.Logger
    region_id: Union[int, str]
    demographic_type: str
    community_id: Optional[str]
    config_params: Dict[str, Any] # Speichert individuelle Overrides

    initial_base_needs: Dict[str, float]
    base_needs: Dict[str, float]
    preference_weights: Dict[str, float]
    need_priorities: Dict[str, NeedPriority]
    substitution_pairs: Dict[Tuple[str, str], float]
    seasonal_factors: Dict[str, Dict[str, float]] # {Season: {Good: Factor}}
    external_event_factors: Dict[str, Dict[str, float]] # {Event: {Good: Factor}}
    active_events: Set[str]

    consumption_received: DefaultDict[str, float]
    consumption_actual: DefaultDict[str, float]
    fulfillment_ratios: DefaultDict[str, float]
    satisfaction: float

    consumption_history: Deque[Dict[str, float]]
    fulfillment_history: Deque[Dict[str, float]]
    satisfaction_history: Deque[float]
    needs_history: Deque[Dict[str, float]]

    adaptation_rate: float
    preference_adaptation_rate: float
    substitution_learning_rate: float
    social_influence_factor: float # New parameter
    lifestyle_factor: float
    consecutive_shortage: DefaultDict[str, int]
    consecutive_surplus: DefaultDict[str, int]
    failed_substitution_attempts: DefaultDict[Tuple[str, str], int]
    shortage_memory: float

    ration_active: bool
    ration_factor: float
    ration_strategy: str

    elasticity_method: str
    sigma: float # für CES
    available_labor_hours: float
    skills: Dict[str, float]
    work_satisfaction: float

    def __init__(self,
                 unique_id: str,
                 model: 'EconomicModel',
                 region_id: Union[int, str],
                 **kwargs): # Empfängt Parameter aus der Config über kwargs
        """
        Initialisiert einen neuen ConsumerAgent.

        Args:
            unique_id: Eindeutige Kennung.
            model: Referenz zum Hauptmodell.
            region_id: Region des Agenten.
            **kwargs: Parameter aus der Konfiguration (AgentPopulationConfig.params
                      oder SpecificAgentConfig.params). Erwartet u.a.:
                      'demographic_type', 'base_needs', 'preference_weights',
                      'need_priorities', 'substitution_pairs', 'lifestyle_factor', etc.
        """
        self.unique_id = unique_id
        self.model = model
        self.logger = model.logger
        self.region_id = region_id
        self.config_params = kwargs # Speichere individuelle Params

        # Lade Parameter aus kwargs mit Fallbacks auf globale Config oder Defaults
        self.demographic_type = kwargs.get("demographic_type", "standard")
        self.community_id = kwargs.get("community_id")

        # Globale Konsumenten-Konfiguration holen
        consumer_cfg = getattr(self.model.config, "demand_config", DemandConfig()) # type: ignore

        # Bedarfe: Nutze kwargs, dann Profil, dann Default aus DemandConfig
        default_needs_from_profile = create_consumer_need_profile(self.demographic_type)
        base_needs_input = kwargs.get("base_needs", default_needs_from_profile)
        self.initial_base_needs = base_needs_input.copy() if base_needs_input else {}
        self.base_needs = self.initial_base_needs.copy()

        # Präferenzen: Nutze kwargs, sonst Default (gleichverteilt)
        default_prefs = {g: 1.0 / len(self.base_needs) for g in self.base_needs} if self.base_needs else {}
        self.preference_weights = kwargs.get("preference_weights", default_prefs)
        self._normalize_preferences() # Sicherstellen, dass Summe 1 ist

        # Prioritäten: Nutze kwargs, sonst Default-Logik
        default_priorities = {
            g: NeedPriority.ESSENTIAL if g in ["food", "housing", "healthcare"] else
               NeedPriority.HIGH if g in ["clothing", "energy"] else
               NeedPriority.MEDIUM if g in ["education"] else
               NeedPriority.LOW
            for g in self.base_needs
        }
        priorities_input = kwargs.get("need_priorities", default_priorities)
        # Konvertiere Strings zu Enums, falls nötig
        self.need_priorities = {
            g: NeedPriority[p.upper()] if isinstance(p, str) else p
            for g, p in priorities_input.items() if isinstance(p, (str, NeedPriority))
        }
        # Fehlende Default setzen
        for g in self.base_needs:
             if g not in self.need_priorities: self.need_priorities[g] = NeedPriority.MEDIUM

        # Substitution: Nutze kwargs, sonst Default (leer)
        subs_input = kwargs.get("substitution_pairs", {})
        # Konvertiere Keys zu Tuples, falls sie als Strings kamen (z.B. "A->B")
        self.substitution_pairs = {
             tuple(k.split('->')) if isinstance(k, str) and '->' in k else k : v
             for k, v in subs_input.items() if isinstance(k, (tuple, str))
        }

        # Prepare substitution cache for quick lookup
        self._prepared_substitutes: Dict[str, List[Tuple[str, float]]] = {}
        self._prepare_substitute_cache()


        # Saisonalität & Events aus globaler Config (könnten auch pro Agent überschrieben werden)
        self.seasonal_factors = consumer_cfg.seasonal_factors
        # In a number of older tests ``ScenarioConfig`` defined ``external_events``.
        # Die aktuelle Konfiguration enthaelt dieses Feld nicht mehr.  Zur
        # Rueckwaertskompatibilitaet greifen wir defensiv darauf zu und
        # verwenden einen leeren Default.
        scenario_cfg = getattr(self.model.config, "scenario_config", None)
        self.external_event_factors = getattr(scenario_cfg, "external_events", {})
        self.active_events = set() # Wird vom Model aktualisiert

        # Lebensstil-Faktor
        self.lifestyle_factor = float(kwargs.get("lifestyle_factor", 1.0))

        # Adaptionsparameter aus Config lesen
        adapt_params = consumer_cfg.consumer_adaptation_params
        self.adaptation_rate = float(
            kwargs.get("adaptation_rate", adapt_params.get("adaptation_rate", 0.05))
        )
        self.preference_adaptation_rate = float(
            kwargs.get("preference_adaptation_rate", adapt_params.get("preference_adaptation_rate", 0.01))
        )
        self.substitution_learning_rate = float(
            kwargs.get("substitution_learning_rate", adapt_params.get("substitution_learning_rate", 0.02))
        )
        self.social_influence_factor = float(
            kwargs.get("social_influence_factor", adapt_params.get("social_influence_factor", 0.1))
        )  # Default 0.1

        # Arbeit & Skills
        self.available_labor_hours: float = float(kwargs.get("available_labor_hours", 40.0))
        self.skills: Dict[str, float] = kwargs.get("skills", {})
        self.work_satisfaction: float = 0.5

        # Parameter für lernbasiertes Substitutionsverhalten
        sub_cfg = adapt_params.get("substitution", {}) if isinstance(adapt_params, dict) else {}
        self.sub_cfg = {
            "learning_rate": sub_cfg.get("learning_rate", 0.15),
            "memory_decay": sub_cfg.get("memory_decay", 0.97),
            "shortage_threshold": sub_cfg.get("shortage_threshold", 0.25),
            "price_threshold": sub_cfg.get("price_threshold", 0.05),
            "elasticity_rho": sub_cfg.get("elasticity_rho", 0.3),
            "budget_fixed": sub_cfg.get("budget_fixed", True),
        }

        # Initialisiere Zustände
        self.consumption_received = defaultdict(float)
        self.consumption_actual = defaultdict(float)
        self.fulfillment_ratios = defaultdict(float)
        self.satisfaction = 0.5 # Startwert

        # Historien
        history_len = self.config_params.get("history_length", 50) # Individuelle Länge?
        self.consumption_history = deque(maxlen=history_len)
        self.fulfillment_history = deque(maxlen=history_len)
        self.satisfaction_history = deque(maxlen=history_len)
        self.needs_history = deque(maxlen=history_len)

        # Adaptionszustand
        self.consecutive_shortage = defaultdict(int)
        self.consecutive_surplus = defaultdict(int)
        self.failed_substitution_attempts = defaultdict(int) # Neuer Zähler
        self.shortage_memory = 0.0

        # Rationierung
        self.ration_active = False
        self.ration_factor = 1.0
        self.ration_strategy = kwargs.get("initial_ration_strategy", "priority")

        # Nutzenfunktion
        # ``DemandConfig`` ist ein Pydantic-Modell und stellt kein ``get`` wie
        # ein Dictionary zur Verfuegung.  Aeltere Tests erwarteten jedoch ein
        # Feld ``default_elasticity_method``.  Falls dieses nicht existiert,
        # verwenden wir "cobb_douglas" als Rueckfallwert.
        self.elasticity_method = kwargs.get(
            "elasticity_method",
            getattr(consumer_cfg, "default_elasticity_method", "cobb_douglas"),
        )
        self.sigma = float(kwargs.get("ces_sigma", 0.7))

        # Wende initiale demografische Faktoren an (nachdem Defaults gesetzt sind)
        self.apply_demographic_factors()
        # Speichere die *initial angepassten* Bedarfe für spätere Referenz bei Adaption
        self.initial_base_needs = self.base_needs.copy()

        # Vorbereitung für lernbasiertes Substitutionsverhalten
        self._goods_list = sorted(self.base_needs.keys())
        G = len(self._goods_list)
        self.S = np.zeros((G, G), dtype=float)
        self.preferences_vector = np.array(
            [self.preference_weights.get(g, 0.0) for g in self._goods_list], dtype=float
        )
        self.last_order = np.zeros(G, dtype=float)

        self.logger.debug(f"ConsumerAgent {self.unique_id} (Typ: {self.demographic_type}) in Region {self.region_id} initialisiert.")

    def _normalize_preferences(self) -> None:
        """Stellt sicher, dass die Summe der Präferenzgewichte 1.0 ist."""
        pref_sum = sum(self.preference_weights.values())
        if pref_sum > 1e-6: # Nur normalisieren, wenn Summe nicht 0 ist
            self.preference_weights = {g: w / pref_sum for g, w in self.preference_weights.items()}
        elif self.base_needs: # Fallback auf Gleichverteilung
            num_goods = len(self.base_needs)
            self.preference_weights = {g: 1.0 / num_goods for g in self.base_needs}

    def _prepare_substitute_cache(self) -> None:
        """Initialisiert den vorberechneten Substitutions-Cache."""
        cache: Dict[str, List[Tuple[str, float]]] = {}
        for (orig, sub), factor in self.substitution_pairs.items():
            cache.setdefault(orig, []).append((sub, factor))
        for orig, lst in cache.items():
            lst.sort(key=lambda x: x[1], reverse=True)
        self._prepared_substitutes = cache

    def _update_prepared_substitutes(self, original_good: str) -> None:
        """Aktualisiert die sortierte Substitute-Liste für ein Gut."""
        entries = [
            (sub, factor)
            for (orig, sub), factor in self.substitution_pairs.items()
            if orig == original_good
        ]
        entries.sort(key=lambda x: x[1], reverse=True)
        if entries:
            self._prepared_substitutes[original_good] = entries
        elif original_good in self._prepared_substitutes:
            del self._prepared_substitutes[original_good]

    def get_labor_offering(self) -> Dict[str, Any]:
        """Stellt die verfügbare Arbeitskraft und Skills zur Verfügung."""
        return {
            "hours": self.available_labor_hours,
            "skills": self.skills.copy(),
        }


    # --- Stage Execution ---
    def step_stage(self, stage: str) -> None:
        """Führt Aktionen für die angegebene Phase aus."""
        self.logger.debug(f"Consumer {self.unique_id}: Executing stage '{stage}'")

        if stage == "planning" or stage == "need_estimation": # Angepasst an Model-Stage
            # Bedarfe anpassen (intern, vor Meldung)
            self.adapt_needs_and_preferences(only_preferences=False)  # Bedarfe VOR Meldung anpassen? Oder danach? Hier: davor
            self.request_consumption_plan()
        elif stage == "event_processing":
            self._process_external_events() # Model setzt self.active_events
            self._apply_seasonal_factors()
        elif stage == "consumption_and_evaluation" or stage == "consumption": # Angepasst an Model-Stage
             # Diese Reihenfolge ist wichtig!
             self.apply_substitutions() # Versucht, Mangel durch Überschuss anderer Güter zu decken
             if self.ration_active:      # Wendet globale Kürzungen an
                  self.apply_rationing()
             self.evaluate_consumption()  # Bewertet, was *nach* Substitution/Rationierung konsumiert wurde
             self.adapt_needs_and_preferences(only_preferences=True) # Passe Präferenzen *nach* Konsum an
             self.update_histories()
             self.reset_consumption_received() # Wichtig: Nur Received zurücksetzen


    # --- Kernlogik Methoden (Teilweise Refactored) ---

    def apply_demographic_factors(self) -> None:
        """Wendet demografische Faktoren und Lifestyle auf Basisbedarfe an."""
        # Nutze Profile oder spezifische Faktoren
        # Beispielhafte Logik (kann komplexer sein)
        factor = 1.0
        if self.demographic_type == "family": factor = 1.4
        elif self.demographic_type == "elderly": factor = 0.9
        elif self.demographic_type == "young": factor = 1.1

        for good in self.base_needs:
            # Spezifischere Anpassungen
            mod = 1.0
            if self.demographic_type == "elderly" and good == "healthcare": mod = 1.8
            if self.demographic_type == "young" and good == "education": mod = 1.5
            # Wende Gesamt- und Lifestyle-Faktor an
            self.base_needs[good] *= factor * mod * self.lifestyle_factor

    def get_current_needs(self) -> Dict[str, float]:
         """Gibt die aktuellen (ggf. saisonal etc. angepassten) Bedarfe zurück."""
         # TODO: Diese Methode sollte die Bedarfe nach Anwendung von
         #       Saisonalität und Events zurückgeben, bevor sie an den Planer gehen.
         # Aktuell werden base_needs direkt modifiziert, was ggf. nicht ideal ist.
         # Besser: Temporäre Bedarfe für den Step berechnen.
         return self.base_needs.copy() # Gibt Kopie der aktuell gültigen Bedarfe zurück

    def request_consumption_plan(self) -> None:
        """Meldet aktuellen Bedarf an das Planungssystem (S4 oder S3)."""
        current_needs = self.get_current_needs()
        # Prioritäten als Enum-Werte oder Integer senden? Hier: Integer
        priorities_int = {g: p.value for g, p in self.need_priorities.items()}

        # Sende an das im Modell definierte Planungssystem
        if self.model.system4planner and hasattr(self.model.system4planner, 'receive_consumer_demand'):
             self.model.system4planner.receive_consumer_demand(
                 consumer_id=self.unique_id, region_id=self.region_id,
                 needs=current_needs, priorities=priorities_int # Sende int Werte
             )
        elif self.model.system3manager and hasattr(self.model.system3manager, 'receive_consumer_demand'):
             # Fallback auf S3, falls S4 nicht direkt Konsumenten managt
             self.model.system3manager.receive_consumer_demand(
                 consumer_id=self.unique_id, region_id=self.region_id,
                 needs=current_needs, priorities=priorities_int
             )
        else:
             self.logger.warning(f"Kein System (S4/S3) zum Empfangen von Konsumentenbedarf für {self.unique_id} gefunden.")


    def receive_allocation(self, allocations: Dict[str, float]) -> None:
        """Empfängt Güterzuteilungen."""
        self.logger.debug(f"Consumer {self.unique_id} empfängt Allokationen: {allocations}")
        for good, amount in allocations.items():
            if amount > 0:
                self.consumption_received[good] += amount

    def _process_external_events(self) -> None:
        """Verarbeitet aktive externe Ereignisse."""
        if hasattr(self.model, 'active_events'): # Prüfe ob Model Attribut hat
            self.active_events = self.model.active_events
        else:
             self.active_events = set() # Fallback

        # Reset temporäre Faktoren vom letzten Mal (wichtig!)
        self.base_needs = self.initial_base_needs.copy() # Beginne mit initialen Werten
        self.apply_demographic_factors() # Wende Demographie+Lifestyle wieder an

        # Wende aktive Events an
        for event in self.active_events:
            if event in self.external_event_factors:
                for good, factor in self.external_event_factors[event].items():
                    if good in self.base_needs:
                        self.base_needs[good] *= factor


    def _apply_seasonal_factors(self) -> None:
        """Wendet saisonale Faktoren an."""
        # Aktuelle Saison vom Modell holen
        current_season = getattr(self.model, "current_season", None)
        if current_season and current_season in self.seasonal_factors:
            for good, factor in self.seasonal_factors[current_season].items():
                if good in self.base_needs:
                    self.base_needs[good] *= factor


    def apply_substitutions(self) -> None:
        """Versucht, Güterknappheit durch vorhandene Substitute auszugleichen."""
        self.logger.debug(f"Consumer {self.unique_id}: Prüfe Substitutionen...")
        # Wichtig: Arbeite mit einer Kopie der erhaltenen Güter,
        # da wir den "Verbrauch" des Substituts simulieren.
        temp_consumption = self.consumption_received.copy()
        applied_substitutions: Dict[str, float] = defaultdict(float) # {OriginalGut: Menge_ersetzt}

        # Iteriere durch Güter mit potenziellem Mangel (sortiert nach Prio?)
        sorted_needs = sorted(
            [(g, n) for g, n in self.base_needs.items() if n > 0],
            key=lambda item: self.need_priorities.get(item[0], NeedPriority.MEDIUM).value,
            reverse=True # Höchste Prio zuerst
        )

        for original_good, original_need in sorted_needs:
            received_amount = temp_consumption.get(original_good, 0.0)
            fulfillment_ratio = received_amount / original_need if original_need > 0 else 1.0

            # Prüfe auf Mangel (z.B. unter 90%)
            if fulfillment_ratio < 0.9:
                 shortage = original_need - received_amount
                 needed_to_substitute = shortage

                 # Finde mögliche Substitute für dieses Gut über den vorbereiteten Cache
                 possible_subs = self._prepared_substitutes.get(original_good, [])

                 for substitute_good, sub_factor in possible_subs:
                     if needed_to_substitute <= 0: break # Mangel bereits gedeckt

                     available_substitute = temp_consumption.get(substitute_good, 0.0)
                     # Wie viel vom Ersatz brauchen wir pro Einheit Original? 1.0 / sub_factor
                     # Wie viel Ersatz ist über den *eigenen* Bedarf hinaus verfügbar?
                     substitute_need = self.base_needs.get(substitute_good, 0.0)
                     # Nehme an, dass der bereits für den eigenen Bedarf reservierte Teil nicht für Substitution zur Verfügung steht.
                     # Hier könnte man auch eine Prio-Logik einbauen (Ersatz für Essential > Eigener Bedarf an Luxury)
                     substitute_surplus = max(0.0, available_substitute - substitute_need)

                     if substitute_surplus > 0 and sub_factor > 1e-6:
                          # Wie viel Originalgut können wir mit dem verfügbaren Surplus ersetzen?
                          can_substitute_orig_amount = substitute_surplus * sub_factor
                          # Wie viel Originalgut müssen wir noch ersetzen?
                          sub_this_round_orig = min(needed_to_substitute, can_substitute_orig_amount)

                          # Wie viel vom Ersatzgut wird dafür *verbraucht*?
                          substitute_used = sub_this_round_orig / sub_factor

                          # Buche die Substitution
                          temp_consumption[substitute_good] -= substitute_used
                          # Erhöhe den "effektiv erhaltenen" Betrag des Originalguts
                          temp_consumption[original_good] = temp_consumption.get(original_good, 0.0) + sub_this_round_orig
                          applied_substitutions[original_good] += sub_this_round_orig
                          needed_to_substitute -= sub_this_round_orig

                          self.logger.debug(
                              f"  Substitution: {sub_this_round_orig:.2f} von {original_good} durch {substitute_used:.2f} von {substitute_good} ersetzt."
                          )
                          # Kein unmittelbares Lernen hier; das erfolgt separat
                     else:
                          # Kein verfügbares Substitut
                          pass


        # Übertrage das Ergebnis der Substitutionen auf consumption_actual
        # Was nicht substituiert wurde, bleibt wie es empfangen wurde
        self.consumption_actual = temp_consumption
        if applied_substitutions:
             self.logger.info(f"Consumer {self.unique_id}: Substitutionen angewendet. Ersetzt: {dict(applied_substitutions)}")


    def _learn_from_substitution(self, original_good: str, substitute_good: str, success: bool) -> None:
        """Lernt aus erfolgreichen oder fehlgeschlagenen Substitutionsversuchen."""
        sub_key = (original_good, substitute_good)

        if success:
            self.failed_substitution_attempts[sub_key] = 0 # Reset Fehlerzähler
            # Erhöhe langsam Präferenz für Ersatzgut
            if substitute_good in self.preference_weights:
                 self.preference_weights[substitute_good] *= (1.0 + self.substitution_learning_rate * 0.1)
                 self._normalize_preferences() # Normalisieren nach Änderung
            # Verbessere Substitutionsfaktor leicht bei Erfolg
            if sub_key in self.substitution_pairs:
                current = self.substitution_pairs[sub_key]
                improved = min(1.0, current * (1.0 + self.substitution_learning_rate))
                self.substitution_pairs[sub_key] = improved
                self._update_prepared_substitutes(original_good)
        else:
            # Erhöhe Fehlerzähler
            self.failed_substitution_attempts[sub_key] += 1

            # Wenn Substitution oft fehlschlägt -> reduziere Faktor
            if self.failed_substitution_attempts[sub_key] >= 5:  # Konfigurierbarer Schwellwert
                if sub_key in self.substitution_pairs:
                    current_factor = self.substitution_pairs[sub_key]
                    self.substitution_pairs[sub_key] = max(0.05, current_factor * 0.98)
                    self.logger.debug(
                        f"Substitutionsfaktor für {original_good}->{substitute_good} reduziert auf {self.substitution_pairs[sub_key]:.2f} wegen wiederholter Fehlschläge."
                    )
                    # Setze Zähler zurück, damit es nicht ewig sinkt
                    self.failed_substitution_attempts[sub_key] = 0
                    self._update_prepared_substitutes(original_good)


    def _apply_substitution(
        self,
        planned_demand: np.ndarray,
        received_last: np.ndarray,
        prices: np.ndarray,
        cfg: Dict[str, Any],
    ) -> np.ndarray:
        """Lernbasierte Substitution der geplanten Nachfrage."""

        eta = cfg.get("learning_rate", self.sub_cfg["learning_rate"])
        beta = cfg.get("memory_decay", self.sub_cfg["memory_decay"])
        theta_s = cfg.get("shortage_threshold", self.sub_cfg["shortage_threshold"])
        theta_p = cfg.get("price_threshold", self.sub_cfg["price_threshold"])
        budget_fixed = cfg.get("budget_fixed", self.sub_cfg["budget_fixed"])

        S = self.S
        shortage = 1.0 - received_last / (self.last_order + 1e-9)

        pref = self.preferences_vector
        mask_short = shortage > theta_s
        delta_price = prices[:, None] - prices[None, :]
        mask_price = delta_price > theta_p

        numer = shortage[:, None] * pref[None, :]
        numer *= mask_short[:, None] & mask_price
        denom = numer.sum(axis=1, keepdims=True) + 1e-9

        S = beta * S + eta * numer / denom
        row_sum = S.sum(axis=1, keepdims=True)
        over_one = row_sum > 1.0
        if np.any(over_one):
            S[over_one[:, 0]] /= row_sum[over_one]
        np.fill_diagonal(S, 0.0)
        self.S = S

        outflow = (S * planned_demand[:, None]).sum(axis=1)
        inflow = S.T @ planned_demand
        demand = planned_demand - outflow + inflow

        if budget_fixed:
            spend = (prices * demand).sum()
            if spend > 0:
                budget = (prices * planned_demand).sum()
                demand *= budget / spend

        self.last_order = planned_demand.copy()

        return np.clip(demand, 0.0, None)


    def apply_rationing(self) -> None:
        """Wendet Rationierungslogik auf `self.consumption_actual` an."""
        if not self.ration_active or self.ration_factor >= 1.0: return

        total_received = sum(self.consumption_actual.values()) # Nutze Wert nach Substitution
        target_total = total_received * self.ration_factor
        current_total = total_received

        if target_total >= current_total: # Rationierung nicht nötig
             self.ration_active = False # Deaktiviere für diesen Schritt
             return

        self.logger.info(f"Consumer {self.unique_id}: Rationierung aktiv (Faktor {self.ration_factor:.2f}, Strategie '{self.ration_strategy}'). Reduziere Konsum von {current_total:.2f} auf {target_total:.2f}.")

        if self.ration_strategy == "priority":
             self._apply_priority_rationing(target_total)
        elif self.ration_strategy == "proportional":
             self._apply_proportional_rationing(target_total)
        # Addiere ggf. "equal" Strategie
        else: # Default = proportional
             self._apply_proportional_rationing(target_total)

    def _apply_priority_rationing(self, target_total_consumption: float) -> None:
        """Rationiert nach Priorität."""
        # Sortiere Güter nach Priorität (höchste zuerst)
        prioritized_goods = sorted(
            self.consumption_actual.keys(),
            key=lambda g: self.need_priorities.get(g, NeedPriority.MEDIUM).value,
            reverse=True
        )

        final_consumption = defaultdict(float)
        allocated_so_far = 0.0

        # Fülle von oben nach unten auf, bis das Budget erschöpft ist
        for good in prioritized_goods:
            available_for_this = self.consumption_actual[good]
            can_allocate = min(available_for_this, target_total_consumption - allocated_so_far)

            if can_allocate > 0:
                final_consumption[good] = can_allocate
                allocated_so_far += can_allocate

            if allocated_so_far >= target_total_consumption - 1e-6: # Toleranz
                break

        self.consumption_actual = final_consumption

    def _apply_proportional_rationing(self, target_total_consumption: float) -> None:
        """Rationiert proportional zum ursprünglichen (nach Substitution) Konsum."""
        current_total = sum(self.consumption_actual.values())
        if current_total <= 0: return

        scale_factor = target_total_consumption / current_total
        final_consumption = defaultdict(float)
        for good, amount in self.consumption_actual.items():
            final_consumption[good] = amount * scale_factor

        self.consumption_actual = final_consumption


    def evaluate_consumption(self) -> None:
        """Bewertet den *finalen* Konsum (nach Substitution/Rationierung) und berechnet Zufriedenheit."""
        if not self.base_needs: # Keine Bedarfe definiert
            self.satisfaction = 1.0
            return

        self.fulfillment_ratios.clear()
        for good, need_amount in self.base_needs.items():
            consumed = self.consumption_actual.get(good, 0.0)
            if need_amount > 1e-9:
                # Erlaube Übererfüllung bis z.B. 200% in der Ratio,
                # aber die Nutzenfunktion begrenzt ggf. den Effekt.
                self.fulfillment_ratios[good] = consumed / need_amount
            else: # Bedarf ist 0 oder negativ
                self.fulfillment_ratios[good] = 1.0 if consumed <= 1e-9 else 2.0 # "Übererfüllt", wenn was konsumiert wurde

        # Zufriedenheit berechnen
        # (Die Nutzenfunktionen selbst bleiben unverändert)
        if self.elasticity_method == "cobb_douglas":
            self.satisfaction = self._utility_cobb_douglas()
        elif self.elasticity_method == "ces":
            self.satisfaction = self._utility_ces()
        elif self.elasticity_method == "additive":
            self.satisfaction = self._utility_additive()
        else:
            self.satisfaction = self._utility_simple_average()

        # Zufriedenheit clippen (z.B. 0 bis 1.5)
        self.satisfaction = np.clip(self.satisfaction, 0.0, 1.5)

        # Knappheitszähler aktualisieren (basierend auf *finalem* Konsum)
        self._update_shortage_counters()
        self.logger.debug(f"Consumer {self.unique_id}: Konsum bewertet. Zufriedenheit: {self.satisfaction:.3f}")


    def _utility_cobb_douglas(self) -> float:
        # ... (Implementierung wie zuvor, nutzt self.fulfillment_ratios und self.preference_weights) ...
        alpha_sum = sum(self.preference_weights.values())
        if alpha_sum <= 0: return 0.0
        normalized_weights = {g: w / alpha_sum for g, w in self.preference_weights.items()}
        utility = 1.0
        num_relevant_goods = 0
        for good, weight in normalized_weights.items():
            if good in self.base_needs: # Nur relevante Güter
                 num_relevant_goods += 1
                 ratio = max(0.01, self.fulfillment_ratios.get(good, 0.0)) # Min 1% Erfüllung für Multiplikation
                 utility *= ratio ** weight
        # Skalierung, wenn nicht alle Güter im Nutzen berücksichtigt? Ggf. nicht nötig.
        return utility if num_relevant_goods > 0 else 0.0


    def _utility_ces(self) -> float:
        # ... (Implementierung wie zuvor, nutzt self.fulfillment_ratios und self.preference_weights) ...
        rho = 1.0 - (1.0 / max(0.01, self.sigma)) # Sigma darf nicht 1 sein
        if abs(rho) < 1e-9: rho = 0.001 # Fall rho=0 (Cobb-Douglas) vermeiden

        w_sum = sum(self.preference_weights.values())
        if w_sum <= 0: return 0.0
        normalized_weights = {g: w / w_sum for g, w in self.preference_weights.items()}

        ces_sum = 0.0
        num_relevant_goods = 0
        for good, weight in normalized_weights.items():
            if good in self.base_needs:
                 num_relevant_goods += 1
                 ratio = max(0.01, self.fulfillment_ratios.get(good, 0.0))
                 ces_sum += weight * (ratio ** rho)

        if ces_sum <= 0 or num_relevant_goods == 0: return 0.0

        # Potenzieren mit 1/rho (Vorsicht bei rho nahe 0)
        try:
             utility = ces_sum ** (1.0 / rho)
        except OverflowError:
             utility = 0.0 # Oder anderer Grenzwert
        except ValueError: # z.B. negative Basis bei ungeradem Exponenten
             utility = 0.0
        return utility

    def _utility_additive(self) -> float:
        # ... (Implementierung wie zuvor) ...
        w_sum = sum(self.preference_weights.values())
        if w_sum <= 0: return 0.0
        normalized_weights = {g: w / w_sum for g, w in self.preference_weights.items()}
        utility = 0.0
        for good, weight in normalized_weights.items():
            if good in self.base_needs:
                # Begrenze Erfüllungsrate auf max. 100% für additiven Nutzen
                fulfillment = np.clip(self.fulfillment_ratios.get(good, 0.0), 0.0, 1.0)
                utility += weight * fulfillment
        return utility

    def _utility_simple_average(self) -> float:
        # ... (Implementierung wie zuvor) ...
        if not self.fulfillment_ratios: return 0.0
        avg_fulfillment = sum(self.fulfillment_ratios.values()) / len(self.fulfillment_ratios)
        return np.clip(avg_fulfillment, 0.0, 1.5)


    def _update_shortage_counters(self) -> None:
        """Aktualisiert Zähler für Mangel/Überschuss basierend auf fulfillment_ratios."""
        # Lese Schwellwerte aus Config
        adapt_params = self.model.config.demand_config.consumer_adaptation_params
        shortage_thresh_ratio = adapt_params.get("shortage_threshold_ratio", 0.85) # Bsp: Ratio statt absoluter Wert
        surplus_thresh_ratio = adapt_params.get("surplus_threshold_ratio", 1.15)

        for good in self.base_needs:
            fulfillment = self.fulfillment_ratios.get(good, 1.0) # Default 1.0 wenn nicht konsumiert/benötigt

            if fulfillment < shortage_thresh_ratio:
                self.consecutive_shortage[good] += 1
                self.consecutive_surplus[good] = 0
                if self.consecutive_shortage[good] > 3: # TODO: Konfigurierbar
                    self.shortage_memory = min(1.0, self.shortage_memory + 0.1)
            elif fulfillment > surplus_thresh_ratio:
                self.consecutive_surplus[good] += 1
                self.consecutive_shortage[good] = 0
                if self.consecutive_surplus[good] > 5: # TODO: Konfigurierbar
                    self.shortage_memory = max(0.0, self.shortage_memory - 0.05)
            else:
                self.consecutive_shortage[good] = 0
                self.consecutive_surplus[good] = 0


    def adapt_needs_and_preferences(self, only_preferences: bool = False) -> None:
        """Passt Bedarfe und/oder Präferenzen dynamisch an."""
        if not only_preferences:
             self._adapt_needs() # Passe Bedarfe an
        self._adapt_preferences() # Passe Präferenzen an

        # Lebensstil seltener anpassen
        if self.model.current_step % self.config_params.get("lifestyle_adapt_interval", 10) == 0:
            self._adapt_lifestyle()


    def _adapt_needs(self) -> None:
        """Passt Basisbedarfe an basierend auf historischer Versorgung (konfigurierbar)."""
        cfg = self.model.config.demand_config.consumer_adaptation_params
        if not cfg.get("enable_need_adaptation", True):
             return

        self.logger.debug(f"Consumer {self.unique_id}: Passe Bedarfe an...")
        # Wichtig: Beginne mit den *initialen* Bedarfen als Referenzpunkt
        current_needs = self.base_needs.copy() # Aktuelle Werte für Glättung

        for good, initial_need in self.initial_base_needs.items():
             if initial_need <= 0: continue # Nur positive Initialbedarfe anpassen

             shortage_count = self.consecutive_shortage.get(good, 0)
             surplus_count = self.consecutive_surplus.get(good, 0)
             adaptation_factor = 1.0

             # Config-Parameter holen
             s_params = cfg.get("shortage_adaptation", {})
             p_params = cfg.get("surplus_adaptation", {})
             g_specific = cfg.get("good_specific_adaptation", {}).get(good, {})

             s_thresh = g_specific.get("threshold_steps", s_params.get("threshold_steps", 5))
             s_sev = g_specific.get("severity_factor", s_params.get("severity_factor", 0.03))
             s_max_reduc = g_specific.get("max_reduction_factor", s_params.get("max_reduction_factor", 0.7)) # Faktor des Initialbedarfs

             p_thresh = g_specific.get("threshold_steps", p_params.get("threshold_steps", 7))
             p_inc = g_specific.get("increase_factor", p_params.get("increase_factor", 1.02))
             p_max_inc = g_specific.get("max_increase_factor", p_params.get("max_increase_factor", 1.5)) # Faktor des Initialbedarfs

             # Anpassung bei Mangel
             if shortage_count >= s_thresh:
                 reduction = min(1.0 - s_max_reduc, s_sev * (shortage_count - s_thresh + 1))
                 adaptation_factor *= (1.0 - reduction)
                 self.logger.debug(f"  Need reduction for {good} due to shortage (factor: {adaptation_factor:.3f})")

             # Anpassung bei Überschuss
             elif surplus_count >= p_thresh:
                  # Prüfe aktuelle Relation zum Max-Limit
                  current_adaptive_factor = current_needs.get(good, initial_need) / max(1e-9, initial_need * self.lifestyle_factor)
                  if current_adaptive_factor < p_max_inc:
                       adaptation_factor *= p_inc
                       self.logger.debug(f"  Need increase for {good} due to surplus (factor: {adaptation_factor:.3f})")

             # Berechne Ziel-Need (basierend auf initial + Adaption + Lifestyle)
             target_need = initial_need * adaptation_factor * self.lifestyle_factor
             # Begrenze durch Min/Max Faktoren
             min_limit = initial_need * s_max_reduc * self.lifestyle_factor
             max_limit = initial_need * p_max_inc * self.lifestyle_factor
             target_need = np.clip(target_need, min_limit, max_limit)

             # Wende Glättung an
             adapt_rate = cfg.get("adaptation_rate", 0.05)
             self.base_needs[good] = current_needs.get(good, target_need) * (1.0 - adapt_rate) + target_need * adapt_rate
             # Sicherstellen, dass Bedarf nicht negativ wird
             self.base_needs[good] = max(0.0, self.base_needs[good])

        # Speichere angepasste Bedarfe in Historie
        self.needs_history.append(self.base_needs.copy())


    def _adapt_preferences(self) -> None:
        """Passt Präferenzgewichte an."""
        adapt_params = self.model.config.demand_config.consumer_adaptation_params
        pref_rate = adapt_params.get("preference_adaptation_rate", self.preference_adaptation_rate) # Nutze spezifischen Rate aus Config

        preference_changed = False
        for good, weight in list(self.preference_weights.items()):
             fulfillment = self.fulfillment_ratios.get(good, 1.0)
             shortage_count = self.consecutive_shortage.get(good, 0)
             surplus_count = self.consecutive_surplus.get(good, 0)
             factor = 1.0

             # Erhöhe Präferenz bei Übererfüllung
             if surplus_count > 3: # Schwellwert konfigurierbar?
                  factor *= (1.0 + pref_rate * 0.1)
             # Senke Präferenz bei langanhaltendem Mangel
             elif shortage_count > 8: # Schwellwert konfigurierbar?
                  factor *= (1.0 - pref_rate * 0.05)

             if abs(factor - 1.0) > 1e-4:
                  self.preference_weights[good] = max(0.01, weight * factor) # Mindestgewicht
                  preference_changed = True

        if preference_changed:
             self._normalize_preferences()
             self.logger.debug(f"Consumer {self.unique_id}: Präferenzen nach individueller Adaption angepasst.")

        # 2. Social Learning (Anpassung durch Community/Region)
        social_prefs: Optional[Dict[str, float]] = None
        if self.community_id:
            community_agent = self.model.get_agent(self.community_id)
            if community_agent and isinstance(community_agent, CommunityConsumerAgent):
                social_prefs = community_agent.get_average_member_preferences(exclude_agent_id=self.unique_id)
                if social_prefs:
                    self.logger.debug(f"Consumer {self.unique_id}: Community-Präferenzen erhalten: {social_prefs}")

        # Optional: Implementiere regionalen Durchschnitt hier, falls Community-Präferenzen nicht verfügbar
        # For this subtask, we are limiting to community influence.
        # if not social_prefs and hasattr(self.model, 'get_average_regional_preferences'):
        #    social_prefs = self.model.get_average_regional_preferences(self.region_id, exclude_agent_id=self.unique_id)

        if social_prefs and self.social_influence_factor > 0:
            social_influence_applied = False
            for good, own_weight in list(self.preference_weights.items()): # Iterate over a copy for modification
                social_weight = social_prefs.get(good, own_weight) # Fallback to own weight if good not in social_prefs
                
                # Weighted average for social influence
                new_weight = (1 - self.social_influence_factor) * own_weight + \
                             self.social_influence_factor * social_weight
                
                if abs(new_weight - own_weight) > 1e-5: # Check if change is significant
                    self.preference_weights[good] = new_weight
                    social_influence_applied = True
            
            if social_influence_applied:
                self._normalize_preferences()
                self.logger.info(f"Consumer {self.unique_id}: Präferenzen nach sozialem Einfluss (Faktor: {self.social_influence_factor:.2f}) angepasst.")


    def _adapt_lifestyle(self) -> None:
        """Passt den Lebensstilfaktor an."""
        if len(self.satisfaction_history) < 5: return # Braucht Historie

        adapt_params = self.model.config.demand_config.consumer_adaptation_params
        lifestyle_params = adapt_params.get("lifestyle_adaptation", {
             "enable": True, "avg_satisfaction_window": 5,
             "high_satisfaction_threshold": 0.9, "low_satisfaction_threshold": 0.6,
             "increase_factor": 1.01, "decrease_factor": 0.98,
             "min_lifestyle_factor": 0.5, "max_lifestyle_factor": 2.0
        })

        if not lifestyle_params.get("enable", True): return

        window = lifestyle_params.get("avg_satisfaction_window", 5)
        recent_satisfaction = np.mean(list(self.satisfaction_history)[-window:])

        old_lifestyle = self.lifestyle_factor
        if recent_satisfaction > lifestyle_params.get("high_satisfaction_threshold", 0.9):
             self.lifestyle_factor *= lifestyle_params.get("increase_factor", 1.01)
        elif recent_satisfaction < lifestyle_params.get("low_satisfaction_threshold", 0.6):
             self.lifestyle_factor *= lifestyle_params.get("decrease_factor", 0.98)

        # Begrenze Lifestyle-Faktor
        self.lifestyle_factor = np.clip(self.lifestyle_factor,
                                        lifestyle_params.get("min_lifestyle_factor", 0.5),
                                        lifestyle_params.get("max_lifestyle_factor", 2.0))

        if abs(self.lifestyle_factor - old_lifestyle) > 0.001:
             self.logger.info(f"Consumer {self.unique_id}: Lifestyle-Faktor angepasst von {old_lifestyle:.2f} auf {self.lifestyle_factor:.2f} (Avg Sat: {recent_satisfaction:.2f})")
             # Nach Lifestyle-Änderung müssen Basisbedarfe neu skaliert werden
             self.apply_demographic_factors() # Wendet auch Lifestyle an


    def update_histories(self) -> None:
        """Aktualisiert Historien am Ende des Steps."""
        # consumption_actual enthält jetzt den finalen Konsum nach Substitution/Rationierung
        self.consumption_history.append(dict(self.consumption_actual))
        self.fulfillment_history.append(dict(self.fulfillment_ratios))
        self.satisfaction_history.append(self.satisfaction)
        # Needs history wird in _adapt_needs aktualisiert

    def reset_consumption_received(self) -> None:
        """Setzt nur den *Empfang* für den nächsten Zyklus zurück."""
        self.consumption_received.clear()
        # Wichtig: consumption_actual wird erst in evaluate_consumption gesetzt/zurückgesetzt

    # --- Öffentliche Methoden zur Steuerung ---
    def enable_rationing(self, factor: float, strategy: str) -> None:
        """Aktiviert Rationierung."""
        self.ration_active = True
        self.ration_factor = np.clip(factor, 0.0, 1.0)
        valid_strategies = ["priority", "proportional", "equal"]
        self.ration_strategy = strategy if strategy in valid_strategies else "priority"
        self.logger.info(f"Consumer {self.unique_id}: Rationierung aktiviert (Faktor: {self.ration_factor:.2f}, Strategie: {self.ration_strategy}).")

    def disable_rationing(self) -> None:
        """Deaktiviert Rationierung."""
        if self.ration_active:
            self.ration_active = False
            self.ration_factor = 1.0
            self.logger.info(f"Consumer {self.unique_id}: Rationierung deaktiviert.")

    # ... (andere öffentliche Methoden wie add_substitution_pair etc. bleiben ähnlich,
    #      aber mit Type Hints und Docstrings versehen) ...

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Gibt eine Zusammenfassung der wichtigsten Statistiken zurück."""
        avg_satisfaction = np.mean(list(self.satisfaction_history)) if self.satisfaction_history else self.satisfaction
        return {
            "agent_id": self.unique_id,
            "region_id": self.region_id,
            "demographic_type": self.demographic_type,
            "community_id": self.community_id,
            "current_satisfaction": round(self.satisfaction, 3),
            "average_satisfaction_hist": round(avg_satisfaction, 3),
            "lifestyle_factor": round(self.lifestyle_factor, 3),
            "shortage_memory": round(self.shortage_memory, 3),
            "current_needs": {k: round(v, 2) for k, v in self.base_needs.items()},
            "last_consumption": {k: round(v, 2) for k, v in self.consumption_actual.items()},
            "last_fulfillment": {k: round(v, 2) for k, v in self.fulfillment_ratios.items()},
            "rationing_active": self.ration_active,
        }


# --- Community Consumer Agent (ebenfalls überarbeitet) ---

class CommunityConsumerAgent(ConsumerAgent):
    """
    Erweiterung des ConsumerAgents für kollektive Bedarfe (z.B. Kommune).

    Aggregiert Bedarfe von Mitgliedern, berücksichtigt Skaleneffekte und
    managt kommunale Güter.
    """
    member_agents: List[ConsumerAgent]
    communal_goods: Set[str]
    scale_efficiency: float
    community_bonus: float

    def __init__(self,
                 unique_id: str,
                 model: 'EconomicModel',
                 region_id: Union[int, str],
                 member_agents: Optional[List[ConsumerAgent]] = None,
                 communal_goods: Optional[Set[str]] = None,
                 scale_efficiency: float = 1.1, # 10% Effizienzgewinn durch Skala
                 community_bonus: float = 0.05, # 5% Zufriedenheitsbonus
                 **kwargs): # Erbt andere Parameter von ConsumerAgent
        """
        Initialisiert einen CommunityConsumerAgent.

        Args:
            unique_id: Eindeutige Kennung.
            model: Referenz zum Hauptmodell.
            region_id: Region der Community.
            member_agents: Liste der Mitglieder (ConsumerAgent Instanzen).
            communal_goods: Set von Gütern, die als kommunal gelten.
            scale_efficiency: Faktor für Skaleneffizienz bei privaten Gütern (>1.0).
            community_bonus: Zusätzlicher Zufriedenheitsbonus durch Gemeinschaft.
            **kwargs: Weitere Argumente für den ConsumerAgent-Konstruktor.
        """
        # Initialisiere Basisklasse (könnte Default-Werte setzen, die wir überschreiben)
        super().__init__(unique_id, model, region_id, **kwargs)

        self.member_agents = member_agents or []
        for agent in self.member_agents:
            agent.community_id = self.unique_id

        self.communal_goods = communal_goods or {"infrastructure", "public_services"}
        self.scale_efficiency = max(1.0, scale_efficiency)
        self.community_bonus = np.clip(community_bonus, 0.0, 0.5)

        # Überschreibe/Initialisiere Bedarfe, Präferenzen, Prioritäten durch Aggregation
        self.aggregate_needs_preferences_priorities()

        self.logger.info(f"CommunityConsumerAgent {self.unique_id} in Region {self.region_id} mit {len(self.member_agents)} Mitgliedern initialisiert.")

    def aggregate_needs_preferences_priorities(self) -> None:
        """Aggregiert Bedarfe, Präferenzen und Prioritäten aller Mitglieder."""
        self.logger.debug(f"Community {self.unique_id}: Aggregiere Bedarfe/Präferenzen...")
        # Reset
        agg_needs: DefaultDict[str, float] = defaultdict(float)
        agg_prefs: DefaultDict[str, float] = defaultdict(float)
        agg_prios: Dict[str, NeedPriority] = {}
        member_prefs_count: DefaultDict[str, int] = defaultdict(int)

        all_goods_in_community = set(self.communal_goods)
        for agent in self.member_agents:
             all_goods_in_community.update(agent.base_needs.keys())

        # Aggregation
        for good in all_goods_in_community:
            if good in self.communal_goods:
                 # Kommunales Gut: Skaliert sublinear mit Mitgliederzahl
                 base_communal_need = self.initial_base_needs.get(good, 1.0) # Default 1 für kommunale Güter
                 member_count = len(self.member_agents)
                 # Beispiel: Wurzelfunktion für Skalierung
                 scaled_need = base_communal_need * math.sqrt(max(1, member_count))
                 agg_needs[good] = scaled_need
                 # Höhere Präferenz/Priorität für kommunale Güter annehmen
                 agg_prefs[good] = self.preference_weights.get(good, 0.2) # Eigener Default oder aus Config
                 agg_prios[good] = self.need_priorities.get(good, NeedPriority.HIGH)

            else:
                 # Privates Gut: Summe über Mitglieder, mit Skaleneffizienz
                 total_member_need = sum(agent.base_needs.get(good, 0.0) for agent in self.member_agents)
                 # Wende Skaleneffizienz an (außer bei 0 oder 1 Mitglied)
                 efficiency_factor = self.scale_efficiency if len(self.member_agents) > 1 else 1.0
                 agg_needs[good] = total_member_need / efficiency_factor

                 # Aggregiere Präferenzen (Durchschnitt) & Prioritäten (Maximum)
                 total_pref_weight = 0.0
                 max_prio = NeedPriority.LOW
                 for agent in self.member_agents:
                      if good in agent.preference_weights:
                           total_pref_weight += agent.preference_weights[good]
                           member_prefs_count[good] += 1
                      current_prio = agent.need_priorities.get(good, NeedPriority.LOW)
                      if current_prio.value > max_prio.value:
                           max_prio = current_prio

                 avg_pref = total_pref_weight / member_prefs_count[good] if member_prefs_count[good] > 0 else 0.0
                 agg_prefs[good] = avg_pref
                 agg_prios[good] = max_prio


        # Setze aggregierte Werte (überschreibe ggf. Basisklassen-Defaults)
        self.base_needs = dict(agg_needs)
        self.preference_weights = dict(agg_prefs)
        self.need_priorities = agg_prios
        self._normalize_preferences() # Wichtig: Nach Aggregation normalisieren

        # Initialisiere/Update initial_base_needs für Community (wichtig für Adaption)
        self.initial_base_needs = self.base_needs.copy()


    def distribute_allocations(self, community_allocations: Dict[str, float]) -> None:
        """Verteilt die an die Community gegangenen Allokationen an die Mitglieder."""
        self.logger.debug(f"Community {self.unique_id}: Verteile Allokationen an Mitglieder...")
        # Zuerst empfängt die Community selbst die Allokation
        super().receive_allocation(community_allocations)

        distributed_amounts = defaultdict(lambda: defaultdict(float)) # {member_id: {good: amount}}

        for good, total_amount in community_allocations.items():
            if total_amount <= 0: continue

            if good in self.communal_goods:
                # Kommunale Güter: Informiere alle Mitglieder (Nutzen ist nicht-rivalisierend)
                # Die individuelle Zufriedenheit wird ggf. durch Qualität/Verfügbarkeit beeinflusst
                for agent in self.member_agents:
                     # Sende eine "Nutzungs"-Information statt einer Menge?
                     # Oder jeder "erhält" symbolisch den vollen Betrag für seine Nutzenberechnung?
                     # Hier: Symbolisch den Betrag senden
                     if hasattr(agent, 'receive_allocation'):
                         agent.receive_allocation({good: total_amount})
                         distributed_amounts[agent.unique_id][good] = total_amount # Tracken
            else:
                # Private Güter: Verteile basierend auf dem Anteil am Gesamtbedarf der Community
                total_need_for_good = sum(agent.base_needs.get(good, 0.0) for agent in self.member_agents)

                if total_need_for_good > 1e-6:
                    for agent in self.member_agents:
                         agent_need = agent.base_needs.get(good, 0.0)
                         agent_share_fraction = agent_need / total_need_for_good
                         agent_allocation = total_amount * agent_share_fraction
                         if agent_allocation > 0:
                              if hasattr(agent, 'receive_allocation'):
                                   agent.receive_allocation({good: agent_allocation})
                                   distributed_amounts[agent.unique_id][good] = agent_allocation # Tracken
                else:
                    # Wenn kein Mitglied Bedarf hat, verteile gleichmäßig? Oder behalte?
                    # Hier: Nicht verteilen, wenn kein Bedarf.
                     self.logger.debug(f"Kein Bedarf für privates Gut {good} in Community {self.unique_id}, {total_amount:.2f} nicht verteilt.")


        self.logger.info(f"Community {self.unique_id}: Allokationen an {len(self.member_agents)} Mitglieder verteilt.")


    def step_stage(self, stage: str) -> None:
        """Koordinierter Step für die Community und ihre Mitglieder."""
        # 1. Aggregation / Planung (vor Mitglieder-Steps)
        if stage == "planning" or stage == "need_estimation":
            self.aggregate_needs_preferences_priorities() # Immer aktuell halten
            self.request_consumption_plan() # Community meldet Gesamtbedarf

        # 2. Event Processing (erst Community, dann Mitglieder)
        elif stage == "event_processing":
             super().step_stage(stage) # Community verarbeitet Events/Saisons
             for agent in self.member_agents:
                  if hasattr(agent, 'step_stage'): agent.step_stage(stage)

        # 3. Konsum/Evaluation (erst Mitglieder, dann Community-Aggregation)
        elif stage == "consumption_and_evaluation" or stage == "consumption":
             # Mitglieder konsumieren/evaluieren zuerst ihre (verteilten) Güter
             for agent in self.member_agents:
                  if hasattr(agent, 'step_stage'): agent.step_stage(stage)

             # Community wertet *ihren* Teil aus (kommunale Güter)
             # und berechnet Gesamt-Zufriedenheit
             super().step_stage(stage) # Wertet self.consumption_received/actual für kommunale Güter aus
             self._update_community_satisfaction() # Aggregiert Mitgliederzufriedenheit

        # Andere Stages ggf. nur an Mitglieder weiterleiten? Oder hat Community eigene Logik?
        # else:
        #      # Standard: An Mitglieder weiterleiten
        #      for agent in self.member_agents:
        #           if hasattr(agent, 'step_stage'): agent.step_stage(stage)


    def get_average_member_preferences(self, exclude_agent_id: Optional[str] = None) -> Dict[str, float]:
        """Berechnet die durchschnittlichen Präferenzgewichte der Community-Mitglieder."""
        if not self.member_agents:
            return {}

        avg_prefs: DefaultDict[str, float] = defaultdict(float)
        counts: DefaultDict[str, int] = defaultdict(int)
        
        members_to_consider = [m for m in self.member_agents if m.unique_id != exclude_agent_id]
        if not members_to_consider: # Avoid division by zero if only one member and it's excluded
            if self.member_agents and not exclude_agent_id: # If no exclusion, use all members
                 members_to_consider = self.member_agents
            else: # No members to average or only self
                 return {}


        for agent in members_to_consider:
            if hasattr(agent, 'preference_weights'):
                for good, weight in agent.preference_weights.items():
                    avg_prefs[good] += weight
                    counts[good] += 1
        
        final_avg_prefs = {
            good: total_weight / counts[good]
            for good, total_weight in avg_prefs.items() if counts[good] > 0
        }
        return final_avg_prefs

    def _update_community_satisfaction(self) -> None:
        """Berechnet die Gesamtzufriedenheit der Community."""
        if not self.member_agents:
            # Zufriedenheit der Community basiert nur auf ihren kommunalen Gütern
            super().evaluate_consumption() # Berechnet self.satisfaction
            return

        # Berechne Zufriedenheit der Community für kommunale Güter
        self.fulfillment_ratios.clear()
        communal_satisfaction = 0.0
        relevant_communal_goods = {g for g in self.communal_goods if g in self.base_needs}
        if relevant_communal_goods:
             # Berechne Nutzen nur für kommunale Güter
             # Temporäre Präferenzen nur für kommunale Güter
             temp_prefs = {g: self.preference_weights.get(g, 0) for g in relevant_communal_goods}
             temp_prefs_sum = sum(temp_prefs.values())
             if temp_prefs_sum > 0:
                  normalized_temp_prefs = {g: w/temp_prefs_sum for g, w in temp_prefs.items()}
                  # Hier nutzen wir die gleiche Nutzenfunktion wie konfiguriert
                  # TODO: Eigene Nutzenfunktion für Community?
                  if self.elasticity_method == "cobb_douglas":
                       communal_satisfaction = self._utility_cobb_douglas(weights=normalized_temp_prefs, goods=relevant_communal_goods)
                  # ... (andere Methoden analog) ...
                  else:
                       communal_satisfaction = self._utility_simple_average(goods=relevant_communal_goods)


        # Durchschnittliche Zufriedenheit der Mitglieder
        member_satisfactions = [getattr(agent, 'satisfaction', 0.0) for agent in self.member_agents]
        avg_member_satisfaction = np.mean(member_satisfactions) if member_satisfactions else 0.0

        # Kombiniere (Gewichtung konfigurierbar?)
        communal_weight = self.config_params.get("communal_satisfaction_weight", 0.3)
        self.satisfaction = (1.0 - communal_weight) * avg_member_satisfaction + communal_weight * communal_satisfaction
        self.satisfaction = np.clip(self.satisfaction, 0.0, 1.5)
        self.satisfaction_history.append(self.satisfaction)


    # --- Hilfsfunktionen für Nutzenberechnung mit spezifischen Gütern ---
    # (Diese überschreiben oder ergänzen die Basisklassen-Methoden leicht)
    def _utility_cobb_douglas(self, weights: Optional[Dict]=None, goods: Optional[Set]=None) -> float:
        alpha_sum = sum(weights.values()) if weights else sum(self.preference_weights.values())
        if alpha_sum <= 0: return 0.0
        normalized_weights = {g: w / alpha_sum for g, w in (weights or self.preference_weights).items()}
        target_goods = goods or self.base_needs.keys()
        utility = 1.0
        num_relevant = 0
        for good, weight in normalized_weights.items():
            if good in target_goods:
                 num_relevant += 1
                 ratio = max(0.01, self.fulfillment_ratios.get(good, 0.0))
                 utility *= ratio ** weight
        return utility if num_relevant > 0 else 0.0

    # ... (andere _utility_* Methoden ähnlich anpassen, um optionale 'goods' zu akzeptieren) ...
    def _utility_simple_average(self, goods: Optional[Set]=None) -> float:
        target_goods = goods or self.fulfillment_ratios.keys()
        relevant_ratios = [self.fulfillment_ratios.get(g, 0.0) for g in target_goods if g in self.fulfillment_ratios]
        if not relevant_ratios: return 0.0
        avg_fulfillment = sum(relevant_ratios) / len(relevant_ratios)
        return np.clip(avg_fulfillment, 0.0, 1.5)


    # --- Mitglieder Management ---
    def add_member(self, agent: ConsumerAgent) -> None:
        """Fügt ein neues Mitglied hinzu."""
        if agent not in self.member_agents:
            self.member_agents.append(agent)
            agent.community_id = self.unique_id
            self.aggregate_needs_preferences_priorities() # Update Community-Profil
            self.logger.info(f"Agent {agent.unique_id} zu Community {self.unique_id} hinzugefügt.")

    def remove_member(self, agent: ConsumerAgent) -> None:
        """Entfernt ein Mitglied."""
        if agent in self.member_agents:
            self.member_agents.remove(agent)
            agent.community_id = None
            self.aggregate_needs_preferences_priorities() # Update Community-Profil
            self.logger.info(f"Agent {agent.unique_id} aus Community {self.unique_id} entfernt.")


# --- Helper Functions (könnten nach utils verschoben werden) ---
# (Hier nur belassen, aber mit Type Hints und Docs versehen)

def create_consumer_need_profile(profile_type: str) -> Dict[str, float]:
    """Erstellt vordefinierte Bedarfsprofile."""
    # ... (Implementierung wie zuvor, aber mit Return Type Hint) ...
    profiles = {
        "basic": {"food": 2.0, "housing": 1.0, "clothing": 0.5, "healthcare": 0.5},
        "family": {"food": 5.0, "housing": 1.5, "clothing": 1.2, "healthcare": 1.0, "education": 1.0},
        # ...
    }
    return profiles.get(profile_type.lower(), profiles["basic"]).copy()

def create_substitution_network(goods: List[str]) -> Dict[Tuple[str, str], float]:
    """Erstellt ein einfaches Substitutionsnetzwerk."""
     # ... (Implementierung wie zuvor, aber mit Type Hints) ...
    substitutions: Dict[Tuple[str, str], float] = {}
    # ...
    return substitutions

# ... (andere Helper wie create_seasonal_factors etc. ebenfalls typisieren) ...