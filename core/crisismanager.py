# Impaler/core/crisismanager.py
"""
Verwaltet die Simulation von Krisen und deren Auswirkungen auf das Wirtschaftsmodell.

Ermöglicht die Definition verschiedener Krisentypen, deren Auslösung
(geplant oder zufällig), die Simulation ihrer Auswirkungen über Zeit und
die Interaktion mit anderen Systemkomponenten (insbesondere VSM).
"""

import random
import logging
import math
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Type, TYPE_CHECKING

# Typ-Prüfung Imports
if TYPE_CHECKING:
    from .model import EconomicModel
    # Importiere Agenten/Systeme, auf die Krisen wirken könnten (optional für Hints)
    # from ..agents import ProducerAgent, ResourceAgent, InfrastructureAgent
    # from ..vsm import System4Planner, System5Policy

# --- Basisklasse für Krisen ---

class BaseCrisis:
    """
    Abstrakte Basisklasse für alle Krisentypen.

    Definiert die grundlegende Struktur und Methoden, die jede Krise implementieren muss.

    Attributes:
        model (EconomicModel): Referenz zum Hauptmodell.
        crisis_manager (CrisisManager): Referenz zum übergeordneten Manager.
        manual_params (Dict[str, Any]): Manuell übergebene Parameter für diese Krise.
        logger (logging.Logger): Logger-Instanz.
        steps_active (int): Anzahl der Schritte, die die Krise bereits aktiv ist.
        duration (int): Geplante Dauer der Krise in Schritten.
        initial_severity (float): Anfänglicher Schweregrad (0.0 bis 1.0+).
        current_severity (float): Aktueller, potenziell angepasster Schweregrad.
        effects (Dict[str, Any]): Beschreibung der spezifischen Auswirkungen der Krise.
        requires_recovery (bool): Ob nach der Krise eine explizite Erholungsphase nötig ist.
    """
    model: 'EconomicModel'
    crisis_manager: 'CrisisManager'
    manual_params: Dict[str, Any]
    logger: logging.Logger
    steps_active: int
    duration: int
    initial_severity: float
    current_severity: float
    effects: Dict[str, Any]
    requires_recovery: bool

    def __init__(self, model: 'EconomicModel', crisis_manager: 'CrisisManager', manual_params: Optional[Dict[str, Any]] = None):
        """
        Initialisiert die Basiskrise.

        Args:
            model: Referenz zum EconomicModel.
            crisis_manager: Referenz zum CrisisManager.
            manual_params: Optionale manuelle Parameter zur Überschreibung von Defaults.
        """
        self.model = model
        self.crisis_manager = crisis_manager
        self.manual_params = manual_params or {}
        self.logger = model.logger.getChild(f"Crisis.{self.get_type()}") # Typ-spezifischer Sub-Logger

        self.steps_active = 0
        self.effects = {}
        self.requires_recovery = True  # Standardmäßig Erholung nötig

        # Parameter für dynamischen Krisenverlauf
        self.max_severity = float(self.manual_params.get("max_severity", 1.0))
        self.k_escal = float(self.manual_params.get("k_escal", 0.35))
        self.t_mid = int(self.manual_params.get("t_mid", 6))
        self.recovery_half_life = int(self.manual_params.get("recovery_half_life", 12))
        self.sigma_noise = float(self.manual_params.get("sigma_noise", 0.03))
        self.mitigation = float(self.manual_params.get("mitigation", 0.0))
        self.severity = 0.0
        self.is_active = True

        # Parameter initialisieren (Dauer, Schweregrad)
        self._init_crisis_parameters()

    def _init_crisis_parameters(self) -> None:
        """Initialisiert Dauer und Schweregrad basierend auf Config und manuellen Params."""
        crisis_type = self.get_type()

        # Dauer bestimmen
        if "duration" in self.manual_params:
            self.duration = int(self.manual_params["duration"])
        else:
            min_d, max_d = self.crisis_manager.crisis_durations.get(crisis_type, (2, 5))
            self.duration = random.randint(min_d, max_d)

        # Schweregrad bestimmen
        if "severity" in self.manual_params:
            self.initial_severity = float(self.manual_params["severity"])
        else:
            min_s, max_s = self.crisis_manager.crisis_severities.get(crisis_type, (0.3, 0.7))
            self.initial_severity = random.uniform(min_s, max_s)

        self.current_severity = self.initial_severity
        self.duration = max(1, self.duration) # Mindestens 1 Schritt
        self.initial_severity = max(0.0, self.initial_severity) # Nicht negativ
        self.logger.debug(f"Parameter initialisiert: Dauer={self.duration}, Init. Schweregrad={self.initial_severity:.2f}")

    def get_type(self) -> str:
        """Gibt den Typ der Krise als String zurück (muss von Subklassen überschrieben werden)."""
        raise NotImplementedError("Subklassen müssen get_type() implementieren.")

    def init_crisis(self) -> None:
        """
        Initialisiert die spezifischen Auswirkungen der Krise beim Start.
        Muss von Subklassen implementiert werden. Setzt self.effects.
        """
        raise NotImplementedError("Subklassen müssen init_crisis() implementieren.")

    def update(self) -> None:
        """Berechnet Eskalation oder Abklingen der Krise nach einer Logistikkurve."""
        if not self.is_active:
            return

        dt = self.steps_active + 1  # Schritte seit Start

        shortage_idx = getattr(self.model, "shortage_index", 0.0)

        if dt <= self.t_mid or shortage_idx > 0.1:
            logistic = self.max_severity / (1 + math.exp(-self.k_escal * (dt - self.t_mid)))

            adaptation_level = self.crisis_manager.get_adaptation_level()
            mitigation = adaptation_level * self.crisis_manager.config.get("adaptation_mitigation_factor", 0.7)

            effective = logistic * (1 - mitigation) * (1 - self.mitigation)
            noise = random.gauss(0.0, self.sigma_noise)
            self.severity = float(min(1.0, max(0.0, effective + noise)))
        else:
            decay = math.pow(0.5, 1 / max(1, self.recovery_half_life))
            self.severity = max(0.0, self.severity * decay)

        self.current_severity = self.severity
        self.steps_active += 1

        if self.severity < 1e-3 and dt > self.t_mid + 3:
            self.is_active = False

        self.logger.debug(
            f"Update Krise '{self.get_type()}' - Step {self.steps_active}/{self.duration}, Severity: {self.severity:.3f}"
        )

        try:
            self._apply_ongoing_effects()
        except Exception as e:
            self.logger.error(
                f"Fehler beim Anwenden der laufenden Effekte für Krise '{self.get_type()}': {e}",
                exc_info=True,
            )


    def _apply_ongoing_effects(self) -> None:
        """
        Wendet laufende Effekte der Krise in jedem Schritt an.
        Kann von Subklassen überschrieben werden. Standardmäßig keine laufenden Effekte.
        """
        pass # Default: Keine spezifischen laufenden Effekte

    def is_finished(self) -> bool:
        """Prüft, ob die geplante Dauer der Krise überschritten wurde."""
        return self.steps_active > self.duration

    def end_crisis(self) -> None:
        """
        Führt Aufräumarbeiten durch, wenn die Krise endet.
        Kann von Subklassen überschrieben werden, um z.B. permanente Effekte zu setzen
        oder den Übergang zur Erholungsphase vorzubereiten.
        """
        self.logger.info(f"Beende Effekte der Krise '{self.get_type()}' in Step {self.model.current_step}.")
        # Standard: Keine spezifischen Aufräumarbeiten nötig

    def get_severity(self) -> float:
        """Gibt den aktuellen effektiven Schweregrad der Krise zurück."""
        return self.current_severity

    def get_effects(self) -> Dict[str, Any]:
        """Gibt ein Dictionary mit den aktuellen Auswirkungen der Krise zurück."""
        # Füge aktuellen Schweregrad hinzu, da er sich ändern kann
        current_effects = self.effects.copy()
        current_effects['current_severity'] = self.current_severity
        current_effects['steps_active'] = self.steps_active
        current_effects['duration'] = self.duration
        return current_effects

# --- CrisisManager Klasse (überarbeitet) ---

class CrisisManager:
    """
    Verwaltet das Auftreten, die Auswirkungen und die Beendigung von Krisen
    innerhalb der Simulation.

    Liest Konfigurationen für Krisenwahrscheinlichkeit, -typen, -dauer und
    -schweregrad. Koordiniert die Interaktion von Krisen mit dem Modell
    und den VSM-Systemen. Verwaltet den globalen Adaptionslevel.

    Attributes:
        model (EconomicModel): Referenz zum Hauptmodell.
        logger (logging.Logger): Logger-Instanz.
        config (Dict[str, Any]): Spezifische Konfiguration für den CrisisManager.
        crisis_active (bool): Ob gerade eine Krise aktiv ist.
        current_crisis (Optional[BaseCrisis]): Die aktuell aktive Krise.
        crisis_history (List[Dict[str, Any]]): Historie vergangener Krisen.
        adaptation_level (float): Globaler Adaptionslevel an die aktuelle Krise (0-1).
        crisis_probability (float): Wahrscheinlichkeit pro Schritt, dass eine neue Krise beginnt.
        crisis_types (List[str]): Liste der möglichen Krisentypen.
        crisis_durations (Dict[str, Tuple[int, int]]): Min/Max Dauer pro Krisentyp.
        crisis_severities (Dict[str, Tuple[float, float]]): Min/Max Schweregrad pro Krisentyp.
        adaptation_factors (Dict[str, float]): Wie schnell Adaption pro Krisentyp erfolgt.
    """
    model: 'EconomicModel'
    logger: logging.Logger
    config: Dict[str, Any]
    crisis_active: bool
    current_crisis: Optional[BaseCrisis]
    crisis_history: List[Dict[str, Any]]
    adaptation_level: float
    crisis_probability: float
    crisis_types: List[str]
    crisis_durations: Dict[str, Tuple[int, int]]
    crisis_severities: Dict[str, Tuple[float, float]]
    adaptation_factors: Dict[str, float]

    # Mapping von Typ-String zu Klassen (erweiterbar)
    CRISIS_CLASS_MAP: Dict[str, Type[BaseCrisis]] = {} # Wird später gefüllt

    def __init__(self, model: 'EconomicModel'):
        """
        Initialisiert den CrisisManager.

        Args:
            model: Referenz zum EconomicModel.
        """
        self.model = model
        self.logger = model.logger.getChild('CrisisManager')
        base_cfg = getattr(model.config, "crisis_config", None)
        if base_cfg is None:
            base_cfg = getattr(model.config, "scenario_config", {})

        if hasattr(base_cfg, "dict"):
            self.config = base_cfg.dict()
        else:
            self.config = dict(base_cfg)

        # Lade Konfiguration mit Defaults
        self.crisis_probability = float(self.config.get("crisis_probability", 0.03))
        self.crisis_types = list(self.config.get("crisis_types", [
            "resource_shortage", "natural_disaster", "infrastructure_failure",
            "technological_disruption", "environmental_catastrophe"
        ]))
        self.crisis_durations = {k: tuple(v) for k, v in self.config.get("crisis_durations", {
            "resource_shortage": (3, 8), "natural_disaster": (2, 5),
            "infrastructure_failure": (1, 4), "technological_disruption": (2, 6),
            "environmental_catastrophe": (4, 8)
        }).items()}
        self.crisis_severities = {k: tuple(v) for k, v in self.config.get("crisis_severities", {
            "resource_shortage": (0.3, 0.7), "natural_disaster": (0.4, 0.8),
            "infrastructure_failure": (0.5, 0.9), "technological_disruption": (0.3, 0.6),
            "environmental_catastrophe": (0.4, 0.7)
        }).items()}
        self.adaptation_factors = {k: float(v) for k, v in self.config.get("adaptation_factors", {
             "resource_shortage": 0.2, "natural_disaster": 0.15, "infrastructure_failure": 0.25,
             "technological_disruption": 0.3, "environmental_catastrophe": 0.1
        }).items()}

        # Krisenstatus
        self.crisis_active = False
        self.current_crisis = None
        self.current_crisis_start_step = None
        self.crisis_history = []
        self.adaptation_level = 0.0 # Startet bei 0

        # Registriere die implementierten Krisenklassen
        self._register_crisis_types()

        self.logger.info("CrisisManager initialisiert.")

    def _register_crisis_types(self):
        """Registriert die bekannten Krisenklassen im CRISIS_CLASS_MAP."""
        # Füge hier alle implementierten Krisenklassen hinzu
        known_crises = [
            ResourceShortageCrisis, NaturalDisasterCrisis, InfrastructureFailureCrisis,
            TechnologicalDisruptionCrisis, EnvironmentalCatastropheCrisis
        ]
        for crisis_cls in known_crises:
            try:
                 # Instanziiere kurz, um get_type() aufzurufen (etwas unschön)
                 # Besser wäre ein Klassenattribut TYPE_NAME
                 type_name = crisis_cls.get_type_name() # Annahme: statische Methode/Attribut
                 if type_name:
                      self.CRISIS_CLASS_MAP[type_name] = crisis_cls
                      # Stelle sicher, dass der Typ auch in der Config ist
                      if type_name not in self.crisis_types:
                           self.logger.warning(f"Krisentyp '{type_name}' implementiert, aber nicht in 'crisis_types' der Konfiguration gelistet.")
            except Exception as e:
                 self.logger.error(f"Fehler beim Registrieren der Krisenklasse {crisis_cls.__name__}: {e}")

        # Filtere crisis_types auf nur registrierte Klassen
        self.crisis_types = [ct for ct in self.crisis_types if ct in self.CRISIS_CLASS_MAP]
        self.logger.info(f"Verfügbare Krisentypen: {list(self.CRISIS_CLASS_MAP.keys())}")


    def update_crisis_states(self) -> None:
        """
        Haupt-Update-Methode, wird in jedem Simulationsschritt aufgerufen.
        Prüft auf neue Krisen oder aktualisiert die laufende Krise.
        """
        if not getattr(self.model.config, "crisis_management_on", True): # Prüfe globalen Schalter
             if self.crisis_active: self._end_crisis() # Beende aktive Krise, wenn deaktiviert
             return

        current_step = getattr(self.model, 'current_step', 0)
        self.logger.debug(f"Update Krisenstatus für Step {current_step}.")

        if not self.crisis_active:
            # Prüfe auf Start einer neuen Krise (zufällig oder geplant)
            # 1. Geplante Events aus ScenarioConfig prüfen
            planned_crisis_event = self._check_planned_crisis(current_step)
            if planned_crisis_event:
                 self.trigger_crisis(
                      crisis_type=planned_crisis_event['type'], # Annahme: Event-Dict hat 'type'
                      manual_params=planned_crisis_event.get('params', {}) # Übergib spezifische Params
                 )
            # 2. Zufällige Krise prüfen (nur wenn keine geplante gestartet wurde)
            elif self._check_random_crisis_initiation():
                 # trigger_crisis wird innerhalb von _check_random_crisis_initiation aufgerufen
                 pass
        else:
            # Laufende Krise aktualisieren
            if self.current_crisis:
                try:
                    # Adaptionslevel erhöhen (langsamer über Zeit)
                    self.adaptation_level = min(1.0, self.adaptation_level + self._calculate_adaptation_increase())
                    self.current_crisis.update()  # Wendet Effekte an, passt Severity an

                    # Prüfen, ob Krise beendet werden soll
                    if self.current_crisis.is_finished():
                        self._end_crisis()
                except Exception as e:
                     self.logger.error(f"Fehler beim Update der Krise '{self.current_crisis.get_type()}': {e}. Versuche, Krise zu beenden.", exc_info=True)
                     self._end_crisis() # Notfall-Ende

    def _check_planned_crisis(self, step: int) -> Optional[Dict[str, Any]]:
        """Prüft, ob ein geplantes Krisenereignis für diesen Schritt existiert."""
        if not hasattr(self.model.config, 'scenario_config'): return None
        scenario_cfg = self.model.config.scenario_config
        if not getattr(scenario_cfg, 'enable_scenario_events', False): return None

        # Suche nach Krisen-Events (angenommen, sie haben einen 'event_type' == 'crisis')
        # Die Struktur der ScenarioConfig muss dies unterstützen
        # Beispielhafte Annahme:
        crisis_events = [evt for evt in getattr(scenario_cfg, 'crisis_events', []) if evt.get('step') == step]
        if crisis_events:
             # Nimm das erste gefundene geplante Krisen-Event
             event_data = crisis_events[0]
             if event_data.get('type') in self.CRISIS_CLASS_MAP:
                 self.logger.info(f"Geplantes Krisen-Event '{event_data.get('name', 'Unnamed Crisis')}' wird in Step {step} ausgelöst.")
                 return event_data
             else:
                 self.logger.warning(f"Geplantes Krisen-Event in Step {step} hat unbekannten Typ: {event_data.get('type')}")
        return None


    def _check_random_crisis_initiation(self) -> bool:
        """Prüft, ob zufällig eine neue Krise ausgelöst wird."""
        if random.random() < self.crisis_probability:
            if self.crisis_types: # Nur wenn es mögliche Typen gibt
                chosen_type = random.choice(self.crisis_types)
                self.logger.info(f"Zufällige Krise wird ausgelöst: Typ '{chosen_type}'")
                try:
                    self.trigger_crisis(chosen_type)
                    return True
                except ValueError as e:
                    self.logger.error(f"Fehler beim Auslösen der zufälligen Krise '{chosen_type}': {e}")
            else:
                self.logger.warning("Zufällige Krise ausgelöst, aber keine Krisentypen in Konfiguration gefunden/registriert.")
        return False

    def _calculate_adaptation_increase(self) -> float:
        """Berechnet, wie stark der Adaptionslevel in diesem Schritt steigt."""
        if not self.current_crisis: return 0.0
        crisis_type = self.current_crisis.get_type()
        base_factor = self.adaptation_factors.get(crisis_type, 0.1)
        # Adaption ist schneller am Anfang, langsamer wenn schon hoch adaptiert
        increase = base_factor * (1.2 - self.adaptation_level) * random.uniform(0.8, 1.2)
        return max(0.0, increase / max(1, self.current_crisis.steps_active * 0.5)) # Anfangs schneller

    def trigger_crisis(self, crisis_type: str, manual_params: Optional[Dict[str, Any]] = None) -> None:
        """Löst eine spezifische Krise aus."""
        if crisis_type not in self.CRISIS_CLASS_MAP:
            self.logger.error(f"Versuch, unbekannten Krisentyp auszulösen: '{crisis_type}'")
            raise ValueError(f"Unbekannter Krisentyp: {crisis_type}")

        if self.crisis_active:
            self.logger.warning(f"Krise '{crisis_type}' kann nicht gestartet werden, da Krise '{self.current_crisis.get_type()}' bereits aktiv ist. Beende laufende Krise zuerst.")
            self._end_crisis()

        self.logger.warning(f"=== KRISE AUSGELÖST (Step {self.model.current_step}): Typ='{crisis_type}' ===")
        self.crisis_active = True
        self.adaptation_level = 0.0 # Reset Adaption bei neuer Krise
        self.current_crisis_start_step = self.model.current_step

        # Erzeuge Krisenobjekt
        CrisisClass = self.CRISIS_CLASS_MAP[crisis_type]
        try:
            self.current_crisis = CrisisClass(self.model, self, manual_params)
            self.current_crisis.init_crisis() # Führe Initialisierungseffekte aus
            self.logger.info(f"Krise '{crisis_type}' initialisiert. Dauer: {self.current_crisis.duration} Schritte, Init. Schweregrad: {self.current_crisis.initial_severity:.2f}.")

            # Benachrichtige VSM-Systeme über Krisenstart
            self._notify_vsm_crisis_start(crisis_type, self.current_crisis.get_effects())

        except Exception as e:
            self.logger.critical(f"!!! Fehler bei der Initialisierung der Krise '{crisis_type}': {e}. Krise wird abgebrochen.", exc_info=True)
            self.crisis_active = False
            self.current_crisis = None

    def _end_crisis(self) -> None:
        """Beendet die aktuelle Krise und startet ggf. Erholung."""
        if not self.crisis_active or not self.current_crisis:
            return

        crisis_type = self.current_crisis.get_type()
        duration = self.current_crisis.duration
        initial_severity = self.current_crisis.initial_severity
        final_adaptation = self.adaptation_level
        # ``current_crisis_start_step`` wurde beim Auslösen der Krise gesetzt und
        # repräsentiert den Step, in dem die Krise begann.
        start_step = self.current_crisis_start_step

        self.logger.warning(f"=== KRISE BEENDET (Step {self.model.current_step}): Typ='{crisis_type}', Dauer={duration} Schritte ===")

        # Füge zur Historie hinzu
        self.crisis_history.append({
            "type": crisis_type,
            "start_step": start_step,
            "end_step": self.current_crisis_start_step + duration,
            "duration": duration,
            "initial_severity": initial_severity,
            "adaptation_achieved": final_adaptation,
            "effects": self.current_crisis.get_effects() # Letzte Effekte speichern
        })

        # Führe Aufräumlogik der Krise aus
        try:
            self.current_crisis.end_crisis()
        except Exception as e:
            self.logger.error(f"Fehler beim Beenden der Krise '{crisis_type}': {e}", exc_info=True)

        # Zustand zurücksetzen
        requires_recovery = self.current_crisis.requires_recovery
        self.crisis_active = False
        self.current_crisis = None
        self.current_crisis_start_step = None
        # Adaptation sinkt langsam wieder ab, wenn keine Krise aktiv ist?
        # self.adaptation_level *= 0.9 # Beispiel: Langsames "Vergessen"

        # Benachrichtige VSM-Systeme über Krisenende
        self._notify_vsm_crisis_end(crisis_type)

        # Starte Erholungsphase, falls nötig
        if requires_recovery:
            self._begin_recovery_phase(crisis_type, duration, initial_severity)

    def _notify_vsm_crisis_start(self, crisis_type: str, effects: Dict[str, Any]) -> None:
        """Informiert relevante VSM-Systeme über den Krisenstart."""
        self.logger.debug(f"Benachrichtige VSM über Start der Krise '{crisis_type}'...")
        if getattr(self.model, 'system5policy', None) and hasattr(self.model.system5policy, 'handle_crisis_start'):
            try: self.model.system5policy.handle_crisis_start(crisis_type, effects)
            except Exception as e: self.logger.error(f"Fehler in System5.handle_crisis_start: {e}", exc_info=False)

        if getattr(self.model, 'system4planner', None) and hasattr(self.model.system4planner, 'handle_crisis_start'):
            try: self.model.system4planner.handle_crisis_start(crisis_type, effects)
            except Exception as e: self.logger.error(f"Fehler in System4.handle_crisis_start: {e}", exc_info=False)
        # TODO: Ggf. weitere Systeme informieren (z.B. S3 direkt?)

    def _notify_vsm_crisis_end(self, crisis_type: str) -> None:
        """Informiert relevante VSM-Systeme über das Krisenende."""
        self.logger.debug(f"Benachrichtige VSM über Ende der Krise '{crisis_type}'...")
        if getattr(self.model, 'system5policy', None) and hasattr(self.model.system5policy, 'handle_crisis_end'):
            try: self.model.system5policy.handle_crisis_end(crisis_type)
            except Exception as e: self.logger.error(f"Fehler in System5.handle_crisis_end: {e}", exc_info=False)

        if getattr(self.model, 'system4planner', None) and hasattr(self.model.system4planner, 'handle_crisis_end'):
            try: self.model.system4planner.handle_crisis_end(crisis_type)
            except Exception as e: self.logger.error(f"Fehler in System4.handle_crisis_end: {e}", exc_info=False)
        # TODO: Ggf. weitere Systeme informieren

    def _begin_recovery_phase(self, crisis_type: str, duration: int, severity: float) -> None:
        """Startet eine Erholungsphase nach einer Krise."""
        # Aktuell nur ein Log-Eintrag. Eine volle Implementierung würde
        # z.B. temporär Investitions-Prioritäten ändern, Reparatur-Tasks
        # erstellen oder Regenerationsraten modifizieren.
        recovery_duration = int(duration * self.config.get("recovery_duration_factor", 1.5)) # Erholung dauert länger
        self.logger.info(f"Beginne Erholungsphase von Krise '{crisis_type}' (Dauer ca. {recovery_duration} Schritte).")
        # TODO: Implementiere tatsächliche Recovery-Mechanismen, evtl. als eigene Stage oder Zustand im Model.

    def _apply_crisis_effects(self, state: Any) -> None:
        """Aggregiert die Effekte aller aktiven Krisen und schreibt sie in den Simulationszustand."""
        from collections import defaultdict

        total_capacity_delta: Dict[str, float] = defaultdict(float)
        price_multiplier: float = 1.0

        active_crises = [c for c in [self.current_crisis] if c and c.is_active]

        for crisis in active_crises:
            for res, weight in getattr(crisis, "affected_resources", {}).items():
                total_capacity_delta[res] -= crisis.get_severity() * weight

            price_multiplier *= 1 + 0.6 * crisis.get_severity()

        if hasattr(state, "capacity"):
            for res, delta in total_capacity_delta.items():
                state.capacity[res] = max(0.0, state.capacity.get(res, 0.0) + delta)

        if hasattr(state, "price_level"):
            state.price_level *= price_multiplier

    def get_adaptation_level(self) -> float:
        """Gibt den aktuellen globalen Adaptionslevel zurück."""
        return self.adaptation_level

    def get_crisis_history(self) -> List[Dict[str, Any]]:
        """Gibt die Historie aller abgeschlossenen Krisen zurück."""
        return self.crisis_history.copy()

    def get_crisis_info(self) -> Optional[Dict[str, Any]]:
         """Gibt Informationen zur aktuellen Krise zurück (für DataCollector)."""
         if self.crisis_active and self.current_crisis:
              return {
                   "active": True,
                   "type": self.current_crisis.get_type(),
                   "severity": self.current_crisis.get_severity(),
                   "steps_active": self.current_crisis.steps_active,
                   "duration": self.current_crisis.duration,
                   "adaptation": self.adaptation_level
              }
         else:
              return {"active": False}

    def get_potential_shock_scenarios(self, num_scenarios: int, base_severity: float) -> List[Dict[str, Any]]:
        """
        Generates a list of potential shock scenarios for 'what-if' analysis by System 4.

        Args:
            num_scenarios: The number of shock scenarios to generate.
            base_severity: A base severity level (e.g., 0.0 to 1.0) that influences
                           the magnitude of the generated shocks.

        Returns:
            A list of dictionaries, where each dictionary defines a shock scenario.
            Example scenarios:
            - {'shock_type': 'resource_availability_shock', 'resource': 'energy', 'reduction_percentage': 0.2}
            - {'shock_type': 'capacity_reduction_shock', 'good_category': 'food', 'reduction_percentage': 0.15, 'scope': 'all_regions'}
            - {'shock_type': 'capacity_reduction_shock', 'good_category': 'electronics', 'reduction_percentage': 0.25, 'scope': 'random_region'}
        """
        if not (0.0 <= base_severity <= 1.0):
            self.logger.warning(f"base_severity ({base_severity}) for get_potential_shock_scenarios out of expected range [0,1]. Clamping.")
            base_severity = max(0.0, min(1.0, base_severity))

        scenarios: List[Dict[str, Any]] = []
        
        # Define potential resources and good categories that can be affected.
        # Ideally, these would be dynamically sourced from model configuration or state.
        # For now, using placeholder lists.
        available_resources = list(getattr(self.model, "resource_types", ["energy", "water", "critical_minerals", "industrial_parts"]))
        available_good_categories = list(getattr(self.model, "good_categories_for_shocks", ["food", "consumer_goods", "heavy_industry", "electronics", "construction_materials"]))

        if not available_resources:
            available_resources = ["dummy_resource"] # Ensure not empty
            self.logger.warning("No available_resources defined in model for shock generation, using 'dummy_resource'.")
        if not available_good_categories:
            available_good_categories = ["dummy_good_category"] # Ensure not empty
            self.logger.warning("No available_good_categories defined in model for shock generation, using 'dummy_good_category'.")

        possible_shock_types = ["resource_availability_shock", "capacity_reduction_shock"]
        
        if num_scenarios <= 0:
            self.logger.warning("num_scenarios requested is 0 or negative. Returning empty list.")
            return scenarios

        for i in range(num_scenarios):
            scenario: Dict[str, Any] = {}
            shock_type = random.choice(possible_shock_types)
            scenario["shock_type"] = shock_type
            
            # Severity for this specific shock, varying around base_severity
            # Allow severity to sometimes exceed 1.0 for very extreme (but less frequent) shocks if base_severity is high.
            # For example, if base_severity is 0.8, actual severity could go up to 0.8 * 1.5 = 1.2 (120% reduction - total wipeout)
            # If base_severity is 0.1, actual could go up to 0.1 * 1.5 = 0.15 (15% reduction)
            severity_multiplier = random.uniform(0.7, 1.5) # Variance around base
            current_shock_severity_value = min(base_severity * severity_multiplier, 1.0) # Cap reduction at 100% for most resources/capacities

            if shock_type == "resource_availability_shock":
                scenario["resource"] = random.choice(available_resources)
                scenario["reduction_percentage"] = round(current_shock_severity_value, 3)
            
            elif shock_type == "capacity_reduction_shock":
                scenario["good_category"] = random.choice(available_good_categories) # System4 needs to map this to specific goods
                scenario["reduction_percentage"] = round(current_shock_severity_value, 3)
                # Scope could be 'all_regions' or 'random_region'.
                # If 'random_region', PlannerIntelligence would need to know about regions if it were to simulate this precisely.
                # For now, this is mostly a label.
                scenario["scope"] = random.choice(["all_regions", "random_region_conceptual"]) 
            
            # Could add other shock types:
            # - "input_efficiency_shock": Multiplies input needs for certain goods.
            # - "transport_disruption_shock": Affects delivery of goods between regions.
            # - "labor_shortage_shock": Reduces available labor for production.

            scenarios.append(scenario)
            self.logger.debug(f"Generated shock scenario {i+1}/{num_scenarios}: {scenario}")
            
        self.logger.info(f"Generated {len(scenarios)} potential shock scenarios with base_severity {base_severity:.2f}.")
        return scenarios


# ===== Spezifische Krisen Implementierungen (Stark gekürzt - müssen detailliert werden) =====
# (Füge hier die Klassen ResourceShortageCrisis, NaturalDisasterCrisis etc. ein,
#  angepasst mit Type Hints, Docstrings und robusterer Logik)

class ResourceShortageCrisis(BaseCrisis):
    """Krise durch Knappheit einer spezifischen Ressource."""
    @staticmethod
    def get_type_name() -> str: return "resource_shortage" # Statische Methode für Registrierung

    def get_type(self) -> str: return self.get_type_name()

    def init_crisis(self) -> None:
        """Initialisiert Ressourcenknappheit."""
        all_resources = list(getattr(self.model.resource_agent, 'resource_capacity', {}).keys())
        if not all_resources:
             self.logger.error("Keine Ressourcen im ResourceAgent gefunden für ResourceShortageCrisis.")
             raise ValueError("Resource agent has no resources.")

        affected_resource = self.manual_params.get("resource", random.choice(all_resources))
        if not hasattr(self.model.resource_agent, 'resource_capacity') or \
           affected_resource not in self.model.resource_agent.resource_capacity:
            self.logger.error(f"Betroffene Ressource '{affected_resource}' nicht in ResourceAgent gefunden.")
            # Wähle andere Ressource oder brich ab
            affected_resource = random.choice(all_resources) if all_resources else None
            if not affected_resource: raise ValueError("Cannot determine affected resource.")


        # Speichere Effekte
        self.effects = {
            "resource": affected_resource,
            "initial_reduction": self.initial_severity,
            # "affected_regions": [...] # Optional
        }
        # Wende initialen Effekt an
        self.original_capacity = self.model.resource_agent.resource_capacity[affected_resource]
        new_capacity = self.original_capacity * (1.0 - self.initial_severity)
        self.model.resource_agent.resource_capacity[affected_resource] = max(0.0, new_capacity) # Nicht negativ
        self.logger.warning(f"ResourceShortageCrisis: Kapazität von '{affected_resource}' reduziert auf {new_capacity:.2f} (-{self.initial_severity:.1%}).")

    def _apply_ongoing_effects(self) -> None:
        """Passt die Kapazität während der Krise dynamisch an den Schweregrad an."""
        resource = self.effects.get("resource")
        if not resource:
            return

        # Zielkapazität anhand des aktuellen Schweregrads berechnen
        # Kapazität nicht unter das Niveau der initialen Reduktion erhöhen
        effective_severity = max(self.initial_severity, self.current_severity)
        target_capacity = max(0.0, self.original_capacity * (1.0 - effective_severity))
        self.model.resource_agent.resource_capacity[resource] = target_capacity

    def end_crisis(self) -> None:
        """Stellt Kapazität teilweise wieder her."""
        resource = self.effects["resource"]
        current_capacity = self.model.resource_agent.resource_capacity[resource]
        # Stelle z.B. 60% des Verlusts wieder her, Rest in Recovery Phase
        recovery_amount = (self.original_capacity - current_capacity) * self.crisis_manager.config.get("crisis_end_recovery_factor", 0.6)
        new_capacity = min(self.original_capacity, current_capacity + recovery_amount)
        self.model.resource_agent.resource_capacity[resource] = new_capacity
        self.logger.info(f"ResourceShortageCrisis Ende: Kapazität von '{resource}' teilweise auf {new_capacity:.2f} wiederhergestellt.")


class NaturalDisasterCrisis(BaseCrisis):
    """Krise durch Naturkatastrophe mit regionalen Auswirkungen."""
    @staticmethod
    def get_type_name() -> str: return "natural_disaster"
    def get_type(self) -> str: return self.get_type_name()

    def init_crisis(self) -> None:
        # ... (Logik zur Bestimmung betroffener Regionen und Schäden) ...
        num_regions = len(self.model.regional_managers)
        k = max(1, num_regions // 3)
        affected_regions = self.manual_params.get(
            "affected_regions",
            random.sample(list(self.model.regional_managers.keys()), k),
        )

        # Allow explicit damage factors via manual params for deterministic tests
        infra_dmg_base = self.manual_params.get(
            "infrastructure_damage",
            random.uniform(0.4, 0.8),
        )
        prod_dmg_base = self.manual_params.get(
            "production_damage",
            random.uniform(0.5, 0.9),
        )
        infra_dmg = self.initial_severity * infra_dmg_base
        prod_dmg = self.initial_severity * prod_dmg_base
        self.effects = {
            "affected_regions": affected_regions,
            "infrastructure_damage_factor": 1.0 - infra_dmg,
            "production_damage_factor": 1.0 - prod_dmg,
        }
        self.logger.warning(f"NaturalDisasterCrisis: Betrifft Regionen {affected_regions}. Infra-Schaden: {infra_dmg:.1%}, Prod-Schaden: {prod_dmg:.1%}")
        # Wende Effekte an
        for r_id in affected_regions:
            rm = self.model.regional_managers.get(r_id)
            if rm is None and isinstance(r_id, int):
                rm = self.model.regional_managers.get(str(r_id)) or self.model.regional_managers.get(f"R{r_id}")
            if rm:
                 for p in getattr(rm, 'local_producers', []):
                      if hasattr(p, 'productive_capacity'):
                           p.productive_capacity *= self.effects["production_damage_factor"]
                 for i in getattr(rm, 'infrastructure', []):
                      if hasattr(i, 'capacity'):
                           i.capacity *= self.effects["infrastructure_damage_factor"]


# --- Additional simple crisis types used in tests ---

class InfrastructureFailureCrisis(BaseCrisis):
    """Simplified crisis type representing infrastructure failures."""
    @staticmethod
    def get_type_name() -> str:
        return "infrastructure_failure"

    def get_type(self) -> str:
        return self.get_type_name()

    def init_crisis(self) -> None:
        self.effects = {"infrastructure_damage_factor": 0.5}
        self.logger.warning("Infrastructure failure crisis started.")


class TechnologicalDisruptionCrisis(BaseCrisis):
    """Simplified technological disruption crisis."""
    @staticmethod
    def get_type_name() -> str:
        return "technological_disruption"

    def get_type(self) -> str:
        return self.get_type_name()

    def init_crisis(self) -> None:
        self.effects = {"tech_disruption": True}
        self.logger.warning("Technological disruption crisis started.")


class EnvironmentalCatastropheCrisis(BaseCrisis):
    """Simplified environmental catastrophe crisis."""
    @staticmethod
    def get_type_name() -> str:
        return "environmental_catastrophe"

    def get_type(self) -> str:
        return self.get_type_name()

    def init_crisis(self) -> None:
        self.effects = {"environmental_damage": 1.0}
        self.logger.warning("Environmental catastrophe crisis started.")

# Registriere die Klassen nach ihrer Definition
CrisisManager.CRISIS_CLASS_MAP = {
    cls.get_type_name(): cls
    for cls in [
        ResourceShortageCrisis,
        NaturalDisasterCrisis,
        InfrastructureFailureCrisis,
        TechnologicalDisruptionCrisis,
        EnvironmentalCatastropheCrisis,
    ]
    if hasattr(cls, 'get_type_name') # Sicherstellen, dass Methode existiert
}
