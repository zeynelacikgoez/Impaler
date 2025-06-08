# Impaler/core/datacollector.py
"""
Modul zur zentralen Sammlung, Speicherung und zum Export von Simulationsdaten.

Ermöglicht die modulare Berechnung von Metriken, konfigurierbare Sammlungsintervalle
und den Export in verschiedene Formate.
"""

import logging
import json
import csv
import io
import os
import random # Added for agent sampling
import datetime # Added for collection_start_time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Callable, Union, Deque, DefaultDict, Tuple, Set, TYPE_CHECKING

# Utils / Metrics (Annahme: Pfade sind korrekt)
from ..utils.math_utils import gini_coefficient
from ..utils.metrics import (
    calculate_price_index,
    calculate_gdp,
    calculate_unemployment_rate,
    calculate_distribution_metrics,
    calculate_producer_metrics,
    calculate_consumer_metrics,
)

# Typ-Prüfung Imports
if TYPE_CHECKING:
    from .model import EconomicModel
    from ..agents.producer import ProducerAgent # Assuming ProducerAgent is available for isinstance
    from ..agents.consumer import ConsumerAgent # Assuming ConsumerAgent is available for isinstance
    # from ..vsm import RegionalManager

class DataCollector:
    """
    Sammelt, speichert und exportiert Daten und Metriken aus der Simulation.

    Bietet eine flexible Architektur zur Registrierung von Metrik-Berechnungen
    und unterstützt verschiedene Detaillevel und Exportformate.

    Attributes:
        config (Dict[str, Any]): Konfiguration für den DataCollector.
        logger (logging.Logger): Logger-Instanz.
        collection_frequency (int): Frequenz der Datensammlung (jeder n-te Schritt).
        store_agent_data (bool): Ob detaillierte Daten pro Agent gespeichert werden sollen.
        detail_level (str): Detailgrad der Agentendaten ('low', 'medium', 'high').
        max_history (int): Maximale Anzahl an Schritten, die in Historien gespeichert werden.
        agent_sampling_rate (float): Anteil der Agenten, von denen Daten gesammelt werden (0.0 bis 1.0).

        model_data (Deque[Dict[str, Any]]): Historie der globalen Modellmetriken pro Schritt.
        agent_data (DefaultDict[str, Deque[Dict[str, Any]]]): Historie der Daten pro Agent {agent_id: deque}.
        regional_data (DefaultDict[Union[int,str], Deque[Dict[str, Any]]]): Historie pro Region {region_id: deque}.
        resource_data (Deque[Dict[str, Any]]): Historie der globalen Ressourcendaten.
        time_series (DefaultDict[str, Deque[Tuple[int, float]]]): Spezifische Zeitreihen für Plotting {metrik_name: deque[(step, value)]}.
        metric_calculators (Dict[str, Callable[['EconomicModel'], Optional[Dict[str, Any]]]]): Registrierte Funktionen zur Metrikberechnung.
    """
    model_data: Deque[Dict[str, Any]]
    agent_data: DefaultDict[str, Deque[Dict[str, Any]]]
    regional_data: DefaultDict[Union[int, str], Deque[Dict[str, Any]]]
    resource_data: Deque[Dict[str, Any]]
    time_series: DefaultDict[str, Deque[Tuple[int, float]]]
    metric_calculators: Dict[str, Callable[['EconomicModel'], Optional[Dict[str, Any]]]]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialisiert den DataCollector.

        Args:
            config: Konfigurationsparameter für die Datensammlung.
                    Erwartet Schlüssel wie 'collection_frequency', 'store_agent_data',
                    'detail_level', 'max_history', 'agent_sampling_rate', 'logging_config' (optional).
        """
        self.config = config or {}
        # Logger holen (wird im Model initialisiert, hier nur holen)
        self.logger = logging.getLogger("DataCollector") # Eigener Logger-Name

        # Konfiguration laden mit Defaults
        self.collection_frequency = int(self.config.get("collection_frequency", 1))
        self.store_agent_data = bool(self.config.get("store_agent_data", True))
        self.detail_level = str(self.config.get("detail_level", "medium")).lower()
        self.max_history = int(self.config.get("max_history", 1000))
        
        # Agent sampling rate
        try:
            sampling_rate = float(self.config.get("agent_sampling_rate", 1.0))
        except ValueError:
            self.logger.warning("Ungültiger Wert für agent_sampling_rate in der Konfiguration. Verwende 1.0.")
            sampling_rate = 1.0
        self.agent_sampling_rate = np.clip(sampling_rate, 0.0, 1.0)

        if self.collection_frequency < 1: self.collection_frequency = 1
        if self.max_history < 1: self.max_history = 1
        if self.detail_level not in ['low', 'medium', 'high']: self.detail_level = 'medium'

        # Storage Container mit maxlen für automatische History-Begrenzung
        self.model_data = deque(maxlen=self.max_history)
        self.agent_data = defaultdict(lambda: deque(maxlen=self.max_history))
        self.regional_data = defaultdict(lambda: deque(maxlen=self.max_history))
        self.resource_data = deque(maxlen=self.max_history)
        self.time_series = defaultdict(lambda: deque(maxlen=self.max_history))

        # Metrik-Rechner registrieren
        self.metric_calculators = {}
        self._register_default_calculators()

        self.logger.info(f"DataCollector initialisiert: Frequenz={self.collection_frequency}, "
                         f"AgentenDaten={self.store_agent_data} ({self.detail_level}), "
                         f"MaxHistory={self.max_history}, AgentSamplingRate={self.agent_sampling_rate:.2f}")

    def _register_default_calculators(self) -> None:
        """Registriert die Standard-Metrik-Rechner."""
        self.logger.debug("Registriere Standard-Metrik-Rechner...")
        calculators = {
            "core_stats": self._calculate_core_stats,
            "production_metrics": self._calculate_production_metrics,
            "resource_metrics": self._calculate_resource_metrics,
            "environmental_metrics": self._calculate_environmental_metrics,
            "welfare_metrics": self._calculate_welfare_metrics,
            "technology_metrics": self._calculate_technology_metrics,
            "infrastructure_metrics": self._calculate_infrastructure_metrics,
            "planning_metrics": self._calculate_planning_metrics,
            "crisis_metrics": self._calculate_crisis_metrics,
            # Füge hier weitere hinzu...
            # "financial_metrics": self._calculate_financial_metrics, # Beispiel
            # "labor_metrics": self._calculate_labor_metrics, # Beispiel
        }
        for name, func in calculators.items():
            self.register_calculator(name, func)

    def register_calculator(self, name: str, calculator_func: Callable[['EconomicModel'], Optional[Dict[str, Any]]]) -> None:
        """Registriert eine Funktion zur Metrikberechnung."""
        if name in self.metric_calculators:
            self.logger.warning(f"Metrik-Rechner '{name}' wird überschrieben.")
        if not callable(calculator_func):
             raise TypeError(f"calculator_func für '{name}' muss aufrufbar sein.")
        self.metric_calculators[name] = calculator_func
        self.logger.debug(f"Metrik-Rechner registriert: {name}")

    def remove_calculator(self, name: str) -> None:
        """Entfernt einen Metrik-Rechner."""
        if name not in self.metric_calculators:
            self.logger.warning(f"Metrik-Rechner '{name}' nicht gefunden, kann nicht entfernt werden.")
            return
        del self.metric_calculators[name]
        self.logger.debug(f"Metrik-Rechner entfernt: {name}")

    # --- Haupt-Sammelmethode ---

    def collect_data(self, model: 'EconomicModel') -> None:
        """
        Sammelt Daten vom Modell und seinen Komponenten für den aktuellen Schritt.
        Wird typischerweise in der 'bookkeeping'-Phase aufgerufen.

        Args:
            model: Das EconomicModel, von dem Daten gesammelt werden sollen.
        """
        current_step = getattr(model, 'current_step', -1)
        if current_step < 0 or current_step % self.collection_frequency != 0:
            return # Nicht in diesem Schritt sammeln

        self.logger.info(f"Datensammlung für Step {current_step}...")
        collection_start_time = datetime.datetime.now() # Use datetime.datetime

        # 1. Berechne alle registrierten Metriken
        metrics = self._calculate_all_metrics(model)

        # 2. Speichere Modell-Level Metriken
        self.model_data.append(metrics)

        # 3. Extrahiere und speichere Zeitreihen
        self._update_time_series(metrics)

        # 4. Sammle Agenten-Daten (falls aktiviert)
        if self.store_agent_data:
            self._collect_agent_data(model)

        # 5. Sammle regionale Daten
        self._collect_regional_data(model)

        # 6. Sammle Ressourcen-Daten
        self._collect_resource_data(model)

        collection_duration = (datetime.datetime.now() - collection_start_time).total_seconds()
        self.logger.info(f"Datensammlung für Step {current_step} abgeschlossen ({collection_duration:.3f}s).")

    # --- Private Hilfsmethoden zur Datensammlung ---

    def _calculate_all_metrics(self, model: 'EconomicModel') -> Dict[str, Any]:
        """Ruft alle registrierten Metrik-Rechner auf und fasst Ergebnisse zusammen."""
        metrics: Dict[str, Any] = {
            "step": getattr(model, 'current_step', -1) # Immer den Step hinzufügen
        }
        for name, calculator in self.metric_calculators.items():
            try:
                result = calculator(model)
                if result is not None:
                    if not isinstance(result, dict):
                         self.logger.warning(f"Metrik-Rechner '{name}' gab keinen Dictionary zurück (Typ: {type(result)}). Wird ignoriert.")
                         continue
                    # Prüfe auf Schlüsselkollisionen
                    overlapping_keys = metrics.keys() & result.keys()
                    if overlapping_keys:
                         self.logger.warning(f"Metrik-Rechner '{name}' überschreibt existierende Schlüssel: {overlapping_keys}. Wert von '{name}' wird verwendet.")
                    metrics.update(result)
                self.logger.debug(f"Metrik-Rechner '{name}' erfolgreich ausgeführt.")
            except Exception as e:
                self.logger.error(f"Fehler im Metrik-Rechner '{name}': {e}", exc_info=True)
                metrics[f"error_{name}"] = str(e) # Fehler loggen

        return metrics

    def _update_time_series(self, metrics: Dict[str, Any]) -> None:
        """Extrahiert spezifische Metriken für schnelle Zeitreihen-Plots."""
        step = metrics.get("step", -1)
        if step < 0: return

        # Mapping von Metrik-Pfad zu Zeitreihen-Schlüssel
        # Pfade verwenden Punktnotation für verschachtelte Dictionaries
        ts_map: Dict[str, str] = {
            "core_stats.producer_count": "producer_count",
            "core_stats.consumer_count": "consumer_count",
            "production_metrics.total_output": "total_output",
            "environmental.total_co2": "co2",
            "environmental.total_pollution": "pollution",
            "welfare.production_gini": "production_gini",
            "welfare.avg_satisfaction": "avg_satisfaction",
            "technology.avg_level": "avg_tech_level",
            "planning.accuracy": "plan_accuracy",
            "crisis.active": "crisis_active", # 1 if active, 0 if not
        }

        for path, ts_key in ts_map.items():
            value = self._get_nested_metric(metrics, path)
            if value is not None and isinstance(value, (int, float)):
                 # Speziell für boolean crisis.active
                 if ts_key == 'crisis_active':
                      value = 1.0 if value else 0.0
                 self.time_series[ts_key].append((step, float(value)))
            # elif value is not None:
            #      self.logger.warning(f"Metrik für Zeitreihe '{ts_key}' (Pfad: '{path}') ist kein numerischer Wert: {type(value)}")

    def _get_nested_metric(self, data: Dict, path: str) -> Optional[Any]:
        """Hilfsfunktion zum Abrufen verschachtelter Metriken."""
        keys = path.split('.')
        value = data
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None # Wert nicht gefunden oder Struktur passt nicht

    def _collect_agent_data(self, model: 'EconomicModel') -> None:
        """Sammelt Daten von einzelnen Agenten (Producer, Consumer, etc.)."""
        current_step = getattr(model, 'current_step', -1)

        # Assuming ProducerAgent and ConsumerAgent are defined and accessible for isinstance
        # If not, use string comparison of agent class names or a more generic approach
        from ..agents.producer import ProducerAgent # Moved import here to ensure availability
        from ..agents.consumer import ConsumerAgent # Moved import here to ensure availability

        agent_list_full = getattr(model, 'producers', []) + \
                          getattr(model, 'consumers', []) + \
                          getattr(model, 'infrastructure_agents', [])
                     # Füge hier ggf. weitere Agententypen hinzu

        agents_to_collect_from = agent_list_full
        if self.agent_sampling_rate < 1.0 and agent_list_full:
            num_to_sample = max(1, int(len(agent_list_full) * self.agent_sampling_rate))
            if num_to_sample < len(agent_list_full): # Avoid sampling if num_to_sample is full list
                 agents_to_collect_from = random.sample(agent_list_full, num_to_sample)
                 self.logger.debug(f"Sampling {len(agents_to_collect_from)}/{len(agent_list_full)} agents for data collection.")

        for agent in agents_to_collect_from:
            agent_id = getattr(agent, 'unique_id', None)
            if not agent_id: continue

            agent_report: Dict[str, Any] = {"step": current_step}
            try:
                # Versuche, eine standardisierte Report-Methode aufzurufen
                if hasattr(agent, 'generate_report'):
                     agent_report.update(agent.generate_report())
                # Fallback auf manuelles Sammeln basierend auf Detaillevel
                elif isinstance(agent, ProducerAgent): 
                     agent_report['type'] = 'Producer'
                     agent_report['output'] = sum(getattr(agent, 'market_production', {}).values())
                     agent_report['capacity_util'] = getattr(agent, 'capacity_utilization', 0.0)
                     if self.detail_level in ['medium', 'high']:
                          agent_report['capacity'] = getattr(agent, 'productive_capacity', 0.0)
                          agent_report['tech_level'] = getattr(agent, 'tech_level', 0.0)
                     if self.detail_level == 'high':
                          agent_report['resource_stock'] = dict(getattr(agent, 'resource_stock', {}))
                          # Füge weitere Details hinzu...
                elif isinstance(agent, ConsumerAgent): 
                     agent_report['type'] = 'Consumer'
                     agent_report['satisfaction'] = getattr(agent, 'satisfaction', 0.0)
                     if self.detail_level in ['medium', 'high']:
                          agent_report['lifestyle_factor'] = getattr(agent, 'lifestyle_factor', 1.0)
                     if self.detail_level == 'high':
                          agent_report['needs'] = getattr(agent, 'base_needs', {})
                          agent_report['consumption'] = getattr(agent, 'consumption_actual', {})
                # ... (Logik für andere Agententypen) ...

                self.agent_data[agent_id].append(agent_report)

            except Exception as e:
                 self.logger.error(f"Fehler beim Sammeln der Daten von Agent {agent_id}: {e}", exc_info=False)


    def _collect_regional_data(self, model: 'EconomicModel') -> None:
        """Sammelt aggregierte Daten pro Region."""
        current_step = getattr(model, 'current_step', -1)
        for region_id, rm in getattr(model, 'regional_managers', {}).items():
             try:
                  if hasattr(rm, 'get_summary_status'):
                       status = rm.get_summary_status() # RM sollte diese Methode haben
                       status['step'] = current_step
                       self.regional_data[region_id].append(status)
                  else:
                       # Manueller Fallback (weniger ideal)
                       self.regional_data[region_id].append({"step": current_step}) # Mindestens Step speichern
             except Exception as e:
                  self.logger.error(f"Fehler beim Sammeln der Daten von Region {region_id}: {e}", exc_info=False)

    def _collect_resource_data(self, model: 'EconomicModel') -> None:
        """Sammelt Daten vom ResourceAgent."""
        current_step = getattr(model, 'current_step', -1)
        if not hasattr(model, 'resource_agent') or model.resource_agent is None:
             return

        resource_agent = model.resource_agent
        try:
             data = {
                 "step": current_step,
                 "inventory": dict(getattr(resource_agent, 'inventory', {})),
                 "capacities": dict(getattr(resource_agent, 'resource_capacity', {})),
                 "regeneration_rates": dict(getattr(resource_agent, 'resource_regeneration', {})),
                 "cumulative_impact": getattr(resource_agent, 'cumulative_impact', 0.0),
                 "sustainability_index": getattr(resource_agent, 'sustainability_index', 1.0)
             }
             # Füge berechnete Verbrauchsdaten hinzu (aus Metriken)
             resource_metrics = self.model_data[-1].get('resource_metrics', {}) if self.model_data else {}
             data["consumption_this_step"] = resource_metrics.get('resource_usage', {})
             data["utilization_this_step"] = resource_metrics.get('resource_utilization', {})

             self.resource_data.append(data)
        except Exception as e:
             self.logger.error(f"Fehler beim Sammeln der Daten vom ResourceAgent: {e}", exc_info=False)


    # --- Metrik-Rechner Implementierungen (Beispiele) ---
    # Diese Methoden berechnen spezifische Metriken basierend auf dem Modellzustand.

    def _calculate_core_stats(self, model: 'EconomicModel') -> Optional[Dict[str, Any]]:
        """Berechnet grundlegende Zählungen und Statistiken."""
        return {
            "core_stats": {
                 "producer_count": len(getattr(model, 'producers', [])),
                 "consumer_count": len(getattr(model, 'consumers', [])),
                 "region_count": len(getattr(model, 'regional_managers', {})),
                 # "active_crisis": getattr(model.crisis_manager, 'crisis_active', False) # Wird in crisis_metrics gemacht
            }
        }

    def _calculate_production_metrics(self, model: 'EconomicModel') -> Optional[Dict[str, Any]]:
        """Berechnet produktionsbezogene Metriken."""
        # Nutze die bereits im Model aggregierten Werte, falls vorhanden
        if hasattr(model, 'total_production_this_step'):
             total_output = sum(model.total_production_this_step.values())
             output_by_good = dict(model.total_production_this_step)
        else: # Fallback: Neu berechnen
             total_output = 0
             output_by_good = defaultdict(float)
             for p in getattr(model, 'producers', []):
                  prod = sum(getattr(p, 'market_production', {}).values())
                  total_output += prod
                  for good, amount in getattr(p, 'market_production', {}).items():
                       output_by_good[good] += amount
             output_by_good = dict(output_by_good)

        # Kapazitätsauslastung
        total_capacity = sum(getattr(p, 'productive_capacity', 0) for p in getattr(model, 'producers', []))
        capacity_utilization = total_output / total_capacity if total_capacity > 0 else 0.0

        return {
            "production_metrics": {
                "total_output": total_output,
                "output_by_good": output_by_good,
                "overall_capacity_utilization": capacity_utilization
            }
        }

    def _calculate_resource_metrics(self, model: 'EconomicModel') -> Optional[Dict[str, Any]]:
        """Berechnet Metriken zur Ressourcennutzung."""
        # Diese Daten werden oft schon in _collect_resource_data gesammelt/berechnet
        # Wir können sie hier ggf. nur strukturieren
        if self.resource_data:
             last_res_data = self.resource_data[-1]
             return {
                 "resource_metrics": {
                      "consumption": last_res_data.get("consumption_this_step", {}),
                      "utilization": last_res_data.get("utilization_this_step", {})
                 }
             }
        return {"resource_metrics": None} # Keine Daten verfügbar

    def _calculate_environmental_metrics(self, model: 'EconomicModel') -> Optional[Dict[str, Any]]:
        """Berechnet Umweltmetriken."""
        # Nutze die im Model berechneten Werte
        return {"environmental": getattr(model, 'environmental_impacts', {})}

    def _calculate_welfare_metrics(self, model: 'EconomicModel') -> Optional[Dict[str, Any]]:
        """Berechnet Wohlfahrts- und Ungleichheitsmetriken."""
        # Nutze die im Model berechneten Werte
        return {"welfare": getattr(model, 'welfare_metrics', {})}

    def _calculate_technology_metrics(self, model: 'EconomicModel') -> Optional[Dict[str, Any]]:
        """Berechnet Technologiemetriken."""
        producers = getattr(model, 'producers', [])
        if not producers: return {"technology": None}
        tech_levels = [getattr(p, 'tech_level', 1.0) for p in producers]
        avg_tech = np.mean(tech_levels)
        gini_tech = gini_coefficient(tech_levels)
        return {"technology": {"avg_level": avg_tech, "inequality": gini_tech}}

    def _calculate_infrastructure_metrics(self, model: 'EconomicModel') -> Optional[Dict[str, Any]]:
        """Berechnet Infrastrukturmetriken."""
        infra_agents = getattr(model, 'infrastructure_agents', [])
        if not infra_agents: return {"infrastructure": None}
        caps_by_type = defaultdict(list)
        for i in infra_agents:
            caps_by_type[getattr(i, 'infra_type', 'unknown')].append(getattr(i, 'capacity', 0))
        avg_caps = {t: np.mean(caps) for t, caps in caps_by_type.items()}
        return {"infrastructure": {"average_capacity_by_type": avg_caps}}

    def _calculate_planning_metrics(self, model: 'EconomicModel') -> Optional[Dict[str, Any]]:
        """Berechnet Metriken zur Planungsgenauigkeit."""
        # Nutze die im Model berechneten Werte
        plan_disc = getattr(model, '_last_plan_discrepancies', {}) # Annahme: wird gespeichert
        return {"planning": plan_disc}

    def _calculate_crisis_metrics(self, model: 'EconomicModel') -> Optional[Dict[str, Any]]:
        """Sammelt Informationen über aktive Krisen."""
        if not hasattr(model, 'crisis_manager'): return {"crisis": None}
        cm = model.crisis_manager
        current = None
        if cm.crisis_active and cm.current_crisis:
             current = {
                 "type": cm.current_crisis.get_type(),
                 "severity": cm.current_crisis.get_severity(),
                 "steps_active": cm.current_crisis.steps_active
             }
        return {"crisis": {"active": cm.crisis_active, "current": current, "adaptation": cm.get_adaptation_level()}}

    # --- Datenexport (CSV Teil überarbeitet) ---

    def export_data(self, format: str = "dict", path: str = None,
                    include_agents: bool = True, max_agent_history: Optional[int] = 50) -> Union[Dict, str, None]:
        """
        Exportiert gesammelte Daten im angegebenen Format.

        Args:
            format: Output-Format ('dict', 'json', 'jsonl', 'csv').
                    'jsonl' (JSON Lines) ist gut für Agentendaten.
                    'csv' exportiert nur model_data und resource_data (flach).
            path: Optionaler Basispfad zum Speichern der Daten. Wenn angegeben,
                  werden ggf. mehrere Dateien erzeugt (z.B. model.csv, agents.jsonl).
            include_agents: Ob Agentendaten exportiert werden sollen (kann groß sein).
            max_agent_history: Maximale Anzahl letzter Einträge pro Agent beim Export (None=alle).

        Returns:
            Daten im angeforderten Format (bei 'dict', 'json') oder None, wenn in Datei gespeichert.
            Für 'csv' wird der CSV-String von model_data zurückgegeben, wenn path=None.
            Für 'jsonl' wird None zurückgegeben (nur Datei-Export sinnvoll).

        Raises:
            ValueError: Bei nicht unterstützten Formaten.
        """
        self.logger.info(f"Exportiere Daten im Format: {format.upper()}")

        # Daten vorbereiten
        export_content = {}
        export_content["model_metadata"] = {"steps_total": len(self.model_data), "max_history": self.max_history}
        export_content["model_data"] = list(self.model_data)
        export_content["time_series"] = {k: list(v) for k, v in self.time_series.items()}
        export_content["regional_data"] = {k: list(v) for k, v in self.regional_data.items()}
        export_content["resource_data"] = list(self.resource_data)

        # Agentendaten (optional und ggf. gekürzt)
        agent_data_to_export = {}
        if include_agents and self.store_agent_data:
            for agent_id, history_deque in self.agent_data.items():
                 if max_agent_history is None:
                      agent_data_to_export[agent_id] = list(history_deque)
                 else:
                      # Nimm nur die letzten N Einträge
                      start_index = max(0, len(history_deque) - max_agent_history)
                      agent_data_to_export[agent_id] = list(history_deque)[start_index:]


        # --- Format-spezifischer Export ---

        if format == "dict":
            if include_agents: export_content["agent_data"] = agent_data_to_export
            result = export_content
            if path: self._save_to_file(result, path, "json") # Dict als JSON speichern
            return result if not path else None

        elif format == "json":
            if include_agents: export_content["agent_data"] = agent_data_to_export
            try:
                 # Versuche, direkt mit Pydantic-kompatiblem Encoder zu serialisieren, falls nötig
                 # Einfacher Ansatz: Konvertiere Deques etc. in Listen (oben schon geschehen)
                 result_str = json.dumps(export_content, indent=2, default=str) # default=str für nicht-serialisierbare Typen
            except TypeError as e:
                 self.logger.error(f"Fehler bei JSON-Serialisierung: {e}. Versuche einfacheren Export.")
                 # Fallback: Nur oberste Ebene exportieren
                 simple_export = {k: v for k, v in export_content.items() if k != 'agent_data'}
                 result_str = json.dumps(simple_export, indent=2, default=str)

            if path: self._save_to_file(result_str, path, "json")
            return result_str if not path else None

        elif format == "jsonl":
            if not path:
                 self.logger.error("JSON Lines (jsonl) Export ist nur mit Angabe eines 'path' sinnvoll.")
                 raise ValueError("JSONL export requires a file path.")
            self._export_to_jsonl(export_content, agent_data_to_export, path)
            return None # Nur Datei-Export

        elif format == "csv":
            # CSV ist schwierig für komplexe/verschachtelte Daten.
            # Exportiere hier nur model_data und resource_data als separate CSVs.
            if not path:
                 self.logger.warning("CSV Export ohne 'path' gibt nur model_data als String zurück.")
                 csv_string = self._generate_csv_string(self.model_data)
                 return csv_string

            # Speichere mehrere CSV-Dateien
            base_path, _ = os.path.splitext(path)
            self._save_to_file(self._generate_csv_string(self.model_data), f"{base_path}_model.csv", "csv")
            self._save_to_file(self._generate_csv_string(self.resource_data), f"{base_path}_resource.csv", "csv")
            # Optional: Regional Data
            for region_id, data in self.regional_data.items():
                 self._save_to_file(self._generate_csv_string(data), f"{base_path}_region_{region_id}.csv", "csv")

            self.logger.warning("CSV Export enthält nur Model-, Ressourcen- und Regionaldaten. Agentendaten und Zeitreihen sind in CSV nicht praktikabel.")
            return None

        else:
            raise ValueError(f"Nicht unterstütztes Exportformat: {format}. Optionen: dict, json, jsonl, csv")

    def _generate_csv_string(self, data_deque: Deque[Dict[str, Any]]) -> str:
        """Hilfsfunktion zum Erzeugen eines CSV-Strings aus einer Deque von Dictionaries."""
        if not data_deque: return ""
        data_list = list(data_deque) # Konvertiere Deque zu Liste für Zugriff

        output = io.StringIO()
        try:
             # Verwende Spalten aus dem ersten Eintrag als Basis (kann unvollständig sein!)
             # Besser: Sammle alle möglichen Spalten (wie zuvor)
             columns: Set[str] = set()
             for entry in data_list:
                  self._gather_column_names(entry, "", columns)
             if not columns: return ""

             sorted_columns = sorted(list(columns))
             writer = csv.DictWriter(output, fieldnames=sorted_columns, restval="", extrasaction='ignore')
             writer.writeheader()

             for entry in data_list:
                 # Flache das Dictionary für den CSV Writer
                 flat_entry = self._flatten_dict(entry)
                 writer.writerow(flat_entry)

        except Exception as e:
             self.logger.error(f"Fehler beim Generieren des CSV-Strings: {e}", exc_info=True)
             return "" # Leerer String bei Fehler

        return output.getvalue()

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flacht ein verschachteltes Dictionary für CSV ab."""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                 # Limitiere Tiefe oder wandle Dicts/Listen in JSON-String um, um CSV nicht zu sprengen
                 if len(parent_key.split(sep)) < 3: # Beispiel: Max 3 Ebenen tief
                    items.update(self._flatten_dict(v, new_key, sep=sep))
                 else:
                    try: items[new_key] = json.dumps(v) # Rest als JSON String
                    except: items[new_key] = str(v) # Fallback auf String
            elif isinstance(v, list):
                 try: items[new_key] = json.dumps(v) # Liste als JSON String
                 except: items[new_key] = str(v)
            else:
                items[new_key] = v
        return items


    def _gather_column_names(self, data: Any, prefix: str, columns: Set[str], max_depth=3) -> None:
        """Sammelt rekursiv Spaltennamen aus (verschachtelten) Dictionaries."""
        if not isinstance(data, dict) or max_depth < 0:
            return

        for key, value in data.items():
            col_name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                 # Nur weiter rekursieren, wenn Tiefe nicht überschritten
                 if max_depth > 0:
                      self._gather_column_names(value, col_name, columns, max_depth - 1)
                 else:
                      columns.add(col_name) # Füge den Key für das Dict hinzu, aber nicht dessen Inhalt
            # Listen werden typischerweise nicht als separate Spalten im flachen CSV dargestellt
            # elif isinstance(value, list):
            #      columns.add(col_name) # Füge Key für die Liste hinzu
            else:
                columns.add(col_name)


    def _get_nested_value(self, data: Dict, column: str) -> Any:
        """Holt verschachtelten Wert sicher ab."""
        # Diese Funktion wird ggf. nicht mehr direkt für DictWriter benötigt,
        # wenn _flatten_dict verwendet wird. Behalte sie als Utility.
        keys = column.split('.')
        value = data
        try:
            for key in keys:
                value = value[key]
            # Konvertiere komplexe Typen für CSV
            if isinstance(value, (dict, list)):
                 try: return json.dumps(value)
                 except: return str(value)
            return value
        except (KeyError, TypeError, IndexError):
            return "" # Leerer String, wenn Pfad nicht existiert


    def _export_to_jsonl(self, model_info: Dict, agent_data: Dict[str, List[Dict]], base_path: str) -> None:
        """Exportiert Daten in mehrere JSON Lines Dateien."""
        base_dir = os.path.dirname(base_path)
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        os.makedirs(base_dir, exist_ok=True)

        # 1. Modelldaten
        model_path = os.path.join(base_dir, f"{base_name}_model.jsonl")
        try:
             with open(model_path, 'w', encoding='utf-8') as f:
                  for entry in model_info.get("model_data", []):
                       json.dump(entry, f, default=str)
                       f.write('\n')
             self.logger.info(f"Modelldaten exportiert nach: {model_path}")
        except IOError as e:
             self.logger.error(f"Fehler beim Schreiben der Modelldaten (JSONL) nach {model_path}: {e}")

        # 2. Agentendaten (jeder Agent eine Zeile pro Step)
        agent_path = os.path.join(base_dir, f"{base_name}_agents.jsonl")
        try:
             with open(agent_path, 'w', encoding='utf-8') as f:
                  for agent_id, history in agent_data.items():
                       for entry in history:
                            entry_with_id = {'agent_id': agent_id, **entry}
                            json.dump(entry_with_id, f, default=str)
                            f.write('\n')
             self.logger.info(f"Agentendaten exportiert nach: {agent_path}")
        except IOError as e:
             self.logger.error(f"Fehler beim Schreiben der Agentendaten (JSONL) nach {agent_path}: {e}")

        # 3. Optional: Regionale Daten, Ressourcendaten etc. in separate JSONL Dateien

    def _save_to_file(self, data_content: Union[str, Dict], file_path: str, format: str) -> None:
        """Speichert aufbereitete Daten sicher in eine Datei."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            mode = 'w'
            encoding = 'utf-8'

            with open(file_path, mode, encoding=encoding) as f:
                 if isinstance(data_content, dict) and format == "json":
                      json.dump(data_content, f, indent=2, default=str)
                 elif isinstance(data_content, str):
                      f.write(data_content)
                 else:
                      # Fallback für unerwartete Typen
                      f.write(str(data_content))

            self.logger.info(f"Daten erfolgreich exportiert nach: {file_path} (Format: {format.upper()})")

        except IOError as e:
            self.logger.error(f"Fehler beim Schreiben der Datei '{file_path}': {e}")
        except TypeError as e:
             self.logger.error(f"Fehler bei der Serialisierung der Daten für '{file_path}': {e}")
        except Exception as e:
             self.logger.error(f"Unerwarteter Fehler beim Speichern nach '{file_path}': {e}", exc_info=True)


    # --- Analyse-Methoden ---

    def get_time_series(self, metric: str) -> List[Tuple[int, float]]:
        """Gibt die Zeitreihe für eine spezifische Metrik zurück."""
        if metric not in self.time_series:
            self.logger.warning(f"Zeitreihe für Metrik '{metric}' nicht gefunden.")
            return []
        # Konvertiere Deque zu Liste für Rückgabe
        return list(self.time_series[metric])

    def analyze_metric_correlation(self, metric1: str, metric2: str) -> Optional[float]:
        """Berechnet die Korrelation zwischen zwei Zeitreihen-Metriken."""
        series1 = dict(self.time_series.get(metric1, [])) # Konvertiere zu Dict für einfachen Lookup
        series2 = dict(self.time_series.get(metric2, []))

        if not series1 or not series2:
            self.logger.warning(f"Nicht genügend Daten für Korrelation zwischen '{metric1}' und '{metric2}'.")
            return None

        # Finde gemeinsame Zeitschritte
        common_steps = sorted(list(set(series1.keys()) & set(series2.keys())))
        if len(common_steps) < 2: # Korrelation braucht mind. 2 Punkte
             self.logger.warning(f"Nicht genügend gemeinsame Datenpunkte für Korrelation zwischen '{metric1}' und '{metric2}'.")
             return None

        values1 = np.array([series1[step] for step in common_steps])
        values2 = np.array([series2[step] for step in common_steps])

        try:
             correlation_matrix = np.corrcoef(values1, values2)
             # Prüfe auf NaN (z.B. wenn eine Serie konstante Werte hat)
             correlation = correlation_matrix[0, 1]
             return float(correlation) if not np.isnan(correlation) else 0.0
        except Exception as e:
             self.logger.error(f"Fehler bei der Korrelationsberechnung zwischen '{metric1}' und '{metric2}': {e}")
             return None


    def get_summary_statistics(self) -> Dict[str, Any]:
        """Berechnet zusammenfassende Statistiken für Hauptzeitreihen."""
        summary: Dict[str, Any] = {"status": "No data collected yet", "steps_recorded": 0}
        if not self.model_data: return summary

        summary["status"] = "Data available"
        summary["steps_recorded"] = len(self.model_data)

        for metric_name, ts_deque in self.time_series.items():
            if ts_deque:
                 values = [v for _, v in ts_deque]
                 try:
                      summary[metric_name] = {
                           "mean": round(float(np.mean(values)), 4),
                           "min": round(float(min(values)), 4),
                           "max": round(float(max(values)), 4),
                           "std": round(float(np.std(values)), 4),
                           "last": round(float(values[-1]), 4) if values else None
                      }
                 except Exception as e:
                      self.logger.warning(f"Fehler bei der Berechnung der Zusammenfassung für Metrik '{metric_name}': {e}")
                      summary[metric_name] = {"error": str(e)}

        return summary