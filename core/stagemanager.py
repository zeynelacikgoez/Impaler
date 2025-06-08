# Impaler/core/stagemanager.py
"""
Verwaltet die verschiedenen Phasen (Stages) einer Simulationsrunde.

Ermöglicht das dynamische Hinzufügen, Entfernen und Konfigurieren von Stages,
definiert deren Ausführungsreihenfolge und unterstützt optionale parallele
Ausführung für entsprechend markierte Stages. Enthält jetzt auch eine
Validierung für explizit definierte Abhängigkeiten zwischen Stages.
"""

import logging
from typing import Callable, Dict, List, Any, Optional, Set, TYPE_CHECKING, Tuple
from collections import defaultdict, deque
from datetime import datetime

# Parallelisierung (optional aber empfohlen)
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    # Logger wird im __init__ gesetzt, kann hier noch nicht loggen

# Typ-Prüfung Imports
if TYPE_CHECKING:
    from .model import EconomicModel

class StageManager:
    """
    Verwaltet und führt die verschiedenen Phasen einer Simulationsrunde aus.

    Ermöglicht flexible Konfiguration des Ablaufs, Abhängigkeitsmanagement
    und optionale Parallelisierung.

    Attributes:
        model (EconomicModel): Referenz zum Hauptmodell.
        logger (logging.Logger): Logger-Instanz.
        stages (Dict[str, Dict[str, Any]]): Registrierte Stages und ihre Eigenschaften.
            Format: { 'stage_name': {'func': Callable, 'parallelizable': bool,
                                     'order': int, 'enabled': bool, 'depends_on': List[str]} }
    """
    model: 'EconomicModel'
    logger: logging.Logger
    stages: Dict[str, Dict[str, Any]]

    def __init__(self, model: 'EconomicModel'):
        """
        Initialisiert den StageManager.

        Args:
            model: Referenz auf das übergeordnete EconomicModel.
        """
        self.model = model
        # Nutze den zentralen Logger des Modells
        self.logger = model.logger.getChild('StageManager') # Erzeugt Sub-Logger
        self.stages = {}
        if not JOBLIB_AVAILABLE:
            self.logger.warning("Joblib nicht gefunden. Parallele Stage-Ausführung ist deaktiviert.")

    def add_stage(self,
                  name: str,
                  func: Callable[[], Any], # Funktion ohne Argumente (greift auf model zu)
                  parallelizable: bool = False,
                  order: int = 100,
                  enabled: bool = True,
                  depends_on: Optional[List[str]] = None) -> None:
        """
        Fügt eine neue Phase zur Simulation hinzu.

        Args:
            name: Eindeutiger Name der Phase (z.B. "production_execution").
            func: Die Funktion, die für diese Phase ausgeführt werden soll.
                  Die Funktion sollte keine Argumente erwarten (sie kann über self.model
                  auf den Modellzustand zugreifen) und kann optional etwas zurückgeben
                  (wird aktuell aber nicht verwendet).
            parallelizable: Gibt an, ob diese Phase sicher parallel mit anderen
                            Stages derselben 'order' ausgeführt werden kann.
                            Achtung: Die Funktion 'func' selbst muss thread-/prozesssicher sein!
            order: Bestimmt die Ausführungsreihenfolge (kleinere Zahlen zuerst).
                   Stages mit gleicher 'order' können potenziell parallel laufen.
            enabled: Ob die Phase standardmäßig aktiviert ist.
            depends_on: Eine optionale Liste von Stage-Namen, von denen diese Stage abhängt.
                        Diese Abhängigkeiten müssen vor oder in derselben Ausführungsgruppe
                        (gleiche 'order') laufen.

        Raises:
            ValueError: Wenn eine Phase mit diesem Namen bereits registriert ist.
        """
        if name in self.stages:
            self.logger.error(f"Fehler beim Hinzufügen: Stage '{name}' ist bereits registriert.")
            raise ValueError(f"Stage '{name}' already registered. Use update_stage to modify.")

        if not callable(func):
             self.logger.error(f"Fehler beim Hinzufügen der Stage '{name}': 'func' ist nicht aufrufbar.")
             raise TypeError("'func' must be a callable function or method.")

        self.stages[name] = {
            "func": func,
            "parallelizable": parallelizable and JOBLIB_AVAILABLE, # Parallel nur wenn joblib da
            "order": order,
            "enabled": enabled,
            "depends_on": depends_on or [] # Immer eine Liste speichern
        }

        dep_str = f"depends on {self.stages[name]['depends_on']}" if depends_on else "no dependencies"
        self.logger.info(f"Stage '{name}' registriert (Order: {order}, Parallel: {self.stages[name]['parallelizable']}, Enabled: {enabled}, {dep_str})")

    def remove_stage(self, name: str) -> None:
        """
        Entfernt eine Phase aus der Simulation.

        Args:
            name: Name der zu entfernenden Phase.

        Raises:
            KeyError: Wenn die Phase nicht existiert.
        """
        if name not in self.stages:
            self.logger.error(f"Fehler beim Entfernen: Stage '{name}' nicht gefunden.")
            raise KeyError(f"Stage '{name}' not registered, cannot remove.")

        del self.stages[name]
        self.logger.info(f"Stage '{name}' entfernt.")

    def update_stage(self, name: str, **kwargs: Any) -> None:
        """
        Aktualisiert Eigenschaften einer existierenden Phase.

        Args:
            name: Name der zu aktualisierenden Phase.
            **kwargs: Zu aktualisierende Eigenschaften (func, parallelizable, order, enabled, depends_on).

        Raises:
            KeyError: Wenn die Phase nicht existiert.
            ValueError: Bei ungültigen Werten für 'depends_on'.
        """
        if name not in self.stages:
            self.logger.error(f"Fehler beim Update: Stage '{name}' nicht gefunden.")
            raise KeyError(f"Stage '{name}' not registered, cannot update.")

        updated_props = []
        for key, value in kwargs.items():
            if key in ["func", "parallelizable", "order", "enabled", "depends_on"]:
                if key == "parallelizable" and value and not JOBLIB_AVAILABLE:
                    self.logger.warning(f"Kann Stage '{name}' nicht als parallelisierbar markieren, da Joblib fehlt.")
                    value = False # Setze zurück auf False
                elif key == "depends_on":
                    if value is None:
                        value = []
                    elif not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                         self.logger.error(f"Ungültiger Typ für 'depends_on' bei Stage '{name}'. Erwarte Optional[List[str]]. Update ignoriert.")
                         continue # Überspringe dieses ungültige Update
                elif key == "func" and not callable(value):
                     self.logger.error(f"Ungültiger Typ für 'func' bei Stage '{name}'. Erwarte Callable. Update ignoriert.")
                     continue
                elif key == "order" and not isinstance(value, int):
                     self.logger.error(f"Ungültiger Typ für 'order' bei Stage '{name}'. Erwarte int. Update ignoriert.")
                     continue
                elif key == "enabled" and not isinstance(value, bool):
                     self.logger.error(f"Ungültiger Typ für 'enabled' bei Stage '{name}'. Erwarte bool. Update ignoriert.")
                     continue


                self.stages[name][key] = value
                updated_props.append(key)
            else:
                self.logger.warning(f"Unbekannte Eigenschaft '{key}' für Stage '{name}' beim Update ignoriert.")

        if updated_props:
            self.logger.info(f"Stage '{name}' aktualisiert. Geänderte Eigenschaften: {updated_props}")
        else:
            self.logger.debug(f"Keine gültigen Eigenschaften zum Aktualisieren für Stage '{name}' gefunden.")


    def enable_stage(self, name: str) -> None:
        """Aktiviert eine Phase."""
        try:
            self.update_stage(name, enabled=True)
        except KeyError:
            # Fehler wurde schon in update_stage geloggt
            pass

    def disable_stage(self, name: str) -> None:
        """Deaktiviert eine Phase."""
        try:
            self.update_stage(name, enabled=False)
        except KeyError:
            # Fehler wurde schon in update_stage geloggt
            pass

    def get_stage(self, name: str) -> Dict[str, Any]:
        """Gibt die Informationen einer spezifischen Phase zurück."""
        if name not in self.stages:
            self.logger.error(f"Fehler beim Abrufen: Stage '{name}' nicht gefunden.")
            raise KeyError(f"Stage '{name}' not registered.")
        return self.stages[name].copy() # Kopie zurückgeben

    def list_stages(self, only_enabled: bool = False) -> List[Dict[str, Any]]:
        """
        Listet alle (oder nur die aktivierten) registrierten Phasen in Ausführungsreihenfolge auf.

        Args:
            only_enabled: Wenn True, werden nur aktivierte Stages zurückgegeben.

        Returns:
            Liste von Dictionaries mit Phaseninformationen, sortiert nach 'order'.
        """
        stages_to_list = self.stages.items()
        if only_enabled:
            stages_to_list = [(n, i) for n, i in stages_to_list if i["enabled"]]

        # Sortiere nach 'order', dann nach Name für stabile Reihenfolge
        sorted_stages = sorted(
            stages_to_list,
            key=lambda item: (item[1]["order"], item[0])
        )

        return [{"name": name, **info} for name, info in sorted_stages]

    # ---------------------------------------------------------------
    # Dependency planning utilities
    # ---------------------------------------------------------------

    def _build_graph(self, stages: List[Tuple[str, Dict[str, Any]]]) -> Tuple[Dict[str, int], Dict[str, Set[str]]]:
        """Create adjacency and indegree maps for the given stages."""
        stage_names = {name for name, _ in stages}
        indeg: Dict[str, int] = {name: 0 for name in stage_names}
        adj: Dict[str, Set[str]] = {name: set() for name in stage_names}

        # Explicit dependencies
        for name, info in stages:
            for dep in info.get("depends_on", []):
                if dep in stage_names:
                    if name not in adj[dep]:
                        adj[dep].add(name)
                        indeg[name] += 1

        # Order-based dependencies ensure old semantics (earlier order before later)
        ordered = sorted(stages, key=lambda x: (x[1]["order"], x[0]))
        for i, (src, src_info) in enumerate(ordered[:-1]):
            for dst, dst_info in ordered[i + 1 :]:
                if src_info["order"] < dst_info["order"] and dst not in adj[src]:
                    adj[src].add(dst)
                    indeg[dst] += 1

        return indeg, adj

    def _level_plan(self, stages: List[Tuple[str, Dict[str, Any]]]) -> List[List[str]]:
        """Return execution levels respecting dependencies."""
        indeg, adj = self._build_graph(stages)
        queue = deque([n for n, d in indeg.items() if d == 0])
        plan: List[List[str]] = []
        visited = 0

        while queue:
            level = list(queue)
            plan.append(level)
            queue.clear()
            for u in level:
                visited += 1
                for v in adj[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        queue.append(v)

        if visited != len(indeg):
            raise RuntimeError("Cycle detected in stage dependency graph")

        return plan

    def _validate_stage_dependencies(self, enabled_stages_execution_plan: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Basic validation and cycle detection for enabled stages."""
        self.logger.debug("Validiere Stage-Abhängigkeiten...")

        # Prüfe, ob alle referenzierten Dependencies existieren
        for name, info in enabled_stages_execution_plan:
            for dep in info.get("depends_on", []):
                if dep not in self.stages:
                    raise ValueError(
                        f"Stage '{name}' hat eine ungültige Abhängigkeit: '{dep}' ist nicht registriert."
                    )

        # Nutze den internen Graphaufbau zur Zyklenerkennung
        indeg, adj = self._build_graph(enabled_stages_execution_plan)
        queue = deque([n for n, d in indeg.items() if d == 0])
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for succ in adj[node]:
                indeg[succ] -= 1
                if indeg[succ] == 0:
                    queue.append(succ)

        if visited != len(indeg):
            raise ValueError("Zirkuläre Abhängigkeiten zwischen Stages gefunden.")

        self.logger.debug("Stage-Abhängigkeiten erfolgreich validiert.")


    def run_stages(self, parallel: bool = False, stages_to_run: Optional[List[str]] = None) -> None:
        """
        Führt alle aktivierten Stages in der korrekten Reihenfolge aus,
        nachdem die Abhängigkeiten validiert wurden.

        Args:
            parallel: Ob parallelisierbare Stages innerhalb einer Order-Gruppe
                      parallel ausgeführt werden sollen (benötigt Joblib).
        """
        effective_parallel = parallel and JOBLIB_AVAILABLE
        if parallel and not JOBLIB_AVAILABLE:
             self.logger.warning("Parallele Ausführung angefordert, aber Joblib nicht verfügbar. Führe sequenziell aus.")

        # 1. Hole aktivierte Stages
        if stages_to_run is not None:
            enabled_stage_items = [
                (name, info)
                for name, info in self.stages.items()
                if info["enabled"] and name in stages_to_run
            ]
        else:
            enabled_stage_items = [
                (name, info) for name, info in self.stages.items() if info["enabled"]
            ]

        enabled_stages_tuples = sorted(
            enabled_stage_items, key=lambda item: (item[1]["order"], item[0])
        )

        # 2. Validire Abhängigkeiten und berechne Ausführungsplan
        try:
            self._validate_stage_dependencies(enabled_stages_tuples)
        except ValueError as e:
            self.logger.critical(f"!!! Validierung der Stage-Abhängigkeiten fehlgeschlagen: {e}. Simulation wird gestoppt.")
            raise # Stoppt die weitere Ausführung

        execution_plan = self._level_plan(enabled_stages_tuples)

        # 3. Führe Stages gemäß Plan aus
        self.logger.info(f"Starte Ausführung von {len(enabled_stages_tuples)} aktivierten Stages...")
        start_time_step = datetime.now()

        for level in execution_plan:
            stage_group = [(name, self.stages[name]) for name in level]
            self.logger.debug(f"--- Executing stage batch: {level} ---")
            group_start_time = datetime.now()

            self._execute_stage_group(stage_group, effective_parallel)

            group_duration = (datetime.now() - group_start_time).total_seconds()
            self.logger.debug(f"--- Stage batch finished in {group_duration:.3f}s ---")

        step_duration = (datetime.now() - start_time_step).total_seconds()
        self.logger.info(f"Alle Stages für Step {self.model.current_step} abgeschlossen in {step_duration:.3f}s.")


    def _execute_stage_group(self, stage_group: List[Tuple[str, Dict[str, Any]]], parallel: bool) -> None:
        """
        Führt eine Gruppe von Stages mit derselben Order aus.

        Args:
            stage_group: Liste von Tupeln (name, info) der Stages.
            parallel: Ob Parallelisierung versucht werden soll.
        """
        parallel_candidates = [(name, info) for name, info in stage_group if info["parallelizable"]]
        serial_stages = [(name, info) for name, info in stage_group if not info["parallelizable"]]

        # Führe zuerst die seriellen Stages dieser Gruppe aus (in ihrer Reihenfolge innerhalb der Gruppe)
        for name, info in serial_stages:
            stage_start_time = datetime.now()
            self.logger.debug(f"Running stage '{name}' (Order={info['order']}, Serial)")
            try:
                info["func"]() # Führe die Funktion aus
            except Exception as e:
                self.logger.exception(f"!!! Fehler während der Ausführung der seriellen Stage '{name}': {e}")
                # Entscheiden, ob hier abgebrochen werden soll oder nur geloggt
                raise # Fehler weitergeben, um Simulation zu stoppen

            duration = (datetime.now() - stage_start_time).total_seconds()
            self.logger.debug(f"Stage '{name}' finished in {duration:.3f}s")


        # Führe dann die parallelisierbaren Stages aus (entweder parallel oder sequenziell)
        if not parallel_candidates:
             return # Nichts mehr zu tun

        if parallel and len(parallel_candidates) > 0: # Parallel nur, wenn aktiviert UND möglich
            self.logger.debug(f"Running {len(parallel_candidates)} stages in parallel (Order={stage_group[0][1]['order']})...")
            parallel_start_time = datetime.now()
            try:
                # Nutze joblib.Parallel - n_jobs=-1 nutzt alle verfügbaren Kerne
                # 'loky' ist oft robuster als 'threading' oder 'multiprocessing'
                Parallel(n_jobs=getattr(self.model.config, 'num_workers', -1), backend='loky')(
                    delayed(info["func"])() for _, info in parallel_candidates
                )
            except Exception as e:
                self.logger.exception(f"!!! Fehler während der parallelen Ausführung von Stages (Order={stage_group[0][1]['order']}): {e}")
                raise # Fehler weitergeben
            duration = (datetime.now() - parallel_start_time).total_seconds()
            self.logger.debug(f"Parallel stages finished in {duration:.3f}s")

        else:
            # Führe parallelisierbare Stages sequenziell aus, wenn parallel=False oder joblib fehlt
            for name, info in parallel_candidates:
                 stage_start_time = datetime.now()
                 # Logge, dass es sequenziell läuft, obwohl parallelisierbar markiert
                 mode = "Parallelizable but running Serially"
                 self.logger.debug(f"Running stage '{name}' (Order={info['order']}, {mode})")
                 try:
                     info["func"]()
                 except Exception as e:
                     self.logger.exception(f"!!! Fehler während der Ausführung der Stage '{name}' (sequenziell trotz parallelizable): {e}")
                     raise
                 duration = (datetime.now() - stage_start_time).total_seconds()
                 self.logger.debug(f"Stage '{name}' finished in {duration:.3f}s")