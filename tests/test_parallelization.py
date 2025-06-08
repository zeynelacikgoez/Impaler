# tests/test_parallelization.py
"""
Tests für die Parallelisierungsfunktionalität des EconomicModel via StageManager.

Diese Tests vergleichen die Ergebnisse einer sequenziellen Ausführung mit einer
parallelen Ausführung (falls 'joblib' verfügbar ist), um sicherzustellen,
dass die Parallelisierung keine unerwarteten Abweichungen oder Fehler verursacht.

Hinweis: Perfekte Bit-für-Bit-Gleichheit ist bei Parallelisierung aufgrund von
Fließkommaarithmetik und potenziell unterschiedlicher Operationsreihenfolgen
nicht immer garantiert oder notwendig. Die Tests konzentrieren sich daher auf
funktionale Äquivalenz und die Annäherung wichtiger Metriken.
"""

import pytest
import numpy as np
import copy
import logging
import json
from typing import Dict, Any

# Importiere notwendige Klassen (Pfade ggf. anpassen)
from impaler.core.model import EconomicModel
from impaler.core.config import SimulationConfig, create_default_config
from impaler.core.stagemanager import JOBLIB_AVAILABLE # Importiere Flag

# --- Test Klasse ---

# Markiere die gesamte Klasse, um sie zu überspringen, wenn joblib fehlt
@pytest.mark.skipif(not JOBLIB_AVAILABLE, reason="joblib nicht installiert, Parallelisierungstests übersprungen.")
class TestParallelization:

    @pytest.fixture(scope="class")
    def parallel_test_config(self) -> SimulationConfig:
        """
        Erstellt eine Konfiguration, die für Parallelisierungstests geeignet ist.
        - Mehrere Agenten und Regionen, um Parallelität zu ermöglichen.
        - VSM aktiviert, da viele Stages davon abhängen.
        - Genügend Schritte, um potenzielle Probleme zu sehen.
        - Gleicher Random Seed für beide Läufe.
        """
        # Nutze die Default-Config als Basis und passe sie an
        cfg = create_default_config()

        # Passe Agentenzahlen an (nicht zu klein, nicht riesig für Testdauer)
        cfg.agent_populations = {
             "producers_p": {
                 "agent_class": "ProducerAgent", "count": 12, "id_prefix": "p_",
                 "params": {"initial_capacity": (80, 120)},
                 "region_distribution": {0: 0.5, 1: 0.5} # Auf 2 Regionen verteilen
             },
             "consumers_c": {
                 "agent_class": "ConsumerAgent", "count": 20, "id_prefix": "c_",
                 "params": {},
                 "region_distribution": {0: 0.5, 1: 0.5}
             }
        }
        cfg.regional_config.regions = ["Region_0", "Region_1"] # 2 Regionen
        cfg.simulation_steps = 15 # Ausreichend Schritte für Interaktionen
        cfg.random_seed = 42 # Fester Seed für Vergleichbarkeit
        cfg.vsm_on = True
        cfg.admm_on = True # Aktiviere ADMM, falls es parallelisierbare Teile hat/haben könnte
        cfg.logging_config.log_level = "INFO" # INFO sollte reichen, DEBUG kann sehr verbose sein

        return cfg

    def run_simulation_and_collect(self, config: SimulationConfig, parallel: bool) -> Dict[str, Any]:
        """
        Führt eine Simulation mit gegebener Konfiguration und Parallelisierungsflag aus
        und gibt die finalen Metriken zurück.

        Args:
            config: Die zu verwendende SimulationConfig.
            parallel: True, um Parallelisierung zu aktivieren, False für sequenziell.

        Returns:
            Ein Dictionary mit den gesammelten Ergebnissen/Metriken.
        """
        # Wichtig: Erstelle eine tiefe Kopie der Config, um Seiteneffekte zu vermeiden
        run_config = config.copy(deep=True)
        run_config.parallel_execution = parallel

        # Erstelle das Modell
        model = EconomicModel(config=run_config)

        # Führe Simulation durch
        try:
             model.run_simulation()
        except Exception as e:
             pytest.fail(f"Simulation fehlgeschlagen (parallel={parallel}): {e}", pytrace=True)


        # Sammle Ergebnisse (z.B. die zuletzt berechneten detaillierten Metriken)
        # Oder nutze den DataCollector, wenn verfügbar und konfiguriert
        if hasattr(model, 'data_collector') and model.data_collector.model_data:
             # Gib den letzten Eintrag der Modelldaten zurück
             return model.data_collector.model_data[-1]
        elif hasattr(model, 'detailed_metrics'):
             # Fallback auf die internen Metriken des Modells
             return model.detailed_metrics
        else:
             # Fallback, falls keine Metriken gesammelt wurden
             return {"error": "No metrics collected"}

    def compare_metrics(self, serial_metrics: Dict[str, Any], parallel_metrics: Dict[str, Any]):
        """
        Vergleicht die Metriken aus seriellen und parallelen Läufen.
        Nutzt pytest.approx für Fließkommazahlen.

        Args:
            serial_metrics: Ergebnisse des sequenziellen Laufs.
            parallel_metrics: Ergebnisse des parallelen Laufs.
        """
        assert serial_metrics.keys() == parallel_metrics.keys(), "Metrik-Schlüssel stimmen nicht überein"

        for key, serial_value in serial_metrics.items():
            parallel_value = parallel_metrics[key]

            # Ignoriere bestimmte Metriken, die abweichen können (z.B. exakte Timings)
            if key in ["step_duration", "simulation_duration"]: # Beispiel
                continue

            if isinstance(serial_value, (float, np.floating)):
                # Fließkommazahlen mit Toleranz vergleichen
                assert serial_value == pytest.approx(parallel_value), f"Metrik '{key}' weicht signifikant ab (Serial: {serial_value}, Parallel: {parallel_value})"
            elif isinstance(serial_value, dict):
                 # Verschachtelte Dictionaries rekursiv vergleichen (vereinfacht)
                 assert isinstance(parallel_value, dict), f"Typ-Mismatch für '{key}': Serial ist dict, Parallel ist {type(parallel_value)}"
                 # Einfacher Vergleich der Keys und Längen
                 assert sorted(serial_value.keys()) == sorted(parallel_value.keys()), f"Keys für verschachtelte Metrik '{key}' stimmen nicht überein"
                 # TODO: Hier könnte ein tieferer Vergleich implementiert werden
                 logging.debug(f"Vergleiche verschachtelte Dict-Metrik '{key}' oberflächlich.")
            elif isinstance(serial_value, list):
                 # Listen nur auf Länge prüfen (Reihenfolge kann abweichen)
                  assert isinstance(parallel_value, list), f"Typ-Mismatch für '{key}': Serial ist list, Parallel ist {type(parallel_value)}"
                  assert len(serial_value) == len(parallel_value), f"Länge der Liste für Metrik '{key}' weicht ab"
                  logging.debug(f"Vergleiche Listen-Metrik '{key}' nur auf Länge.")
            else:
                # Exakter Vergleich für andere Typen (int, str, bool)
                assert serial_value == parallel_value, f"Metrik '{key}' weicht ab (Serial: {serial_value}, Parallel: {parallel_value})"

    def test_serial_vs_parallel_equivalence(self, parallel_test_config: SimulationConfig):
        """
        Haupttest: Führt die Simulation einmal sequenziell und einmal parallel
        mit identischer Konfiguration durch und vergleicht die Ergebnisse.
        """
        logging.info("Starte sequenziellen Simulationslauf...")
        serial_results = self.run_simulation_and_collect(parallel_test_config, parallel=False)
        logging.info("Sequenzieller Lauf beendet.")

        logging.info("Starte parallelen Simulationslauf...")
        parallel_results = self.run_simulation_and_collect(parallel_test_config, parallel=True)
        logging.info("Paralleler Lauf beendet.")

        # Prüfe, ob beide Läufe Metriken geliefert haben
        assert "error" not in serial_results, f"Fehler im sequenziellen Lauf: {serial_results.get('error')}"
        assert "error" not in parallel_results, f"Fehler im parallelen Lauf: {parallel_results.get('error')}"

        # Vergleiche die Ergebnisse
        logging.info("Vergleiche Ergebnisse zwischen sequentiellem und parallelem Lauf...")
        try:
            self.compare_metrics(serial_results, parallel_results)
            logging.info("Ergebnisse von seriellem und parallelem Lauf sind äquivalent (innerhalb der Toleranz).")
        except AssertionError as e:
            # Bei Fehlern detailliertere Infos ausgeben
            logging.error(f"Abweichung zwischen seriellem und parallelem Lauf festgestellt!")
            logging.error(f"Serial Metrics: {json.dumps(serial_results, indent=2, default=str)}")
            logging.error(f"Parallel Metrics: {json.dumps(parallel_results, indent=2, default=str)}")
            pytest.fail(f"Parallelisierungs-Äquivalenztest fehlgeschlagen: {e}", pytrace=False)


    # Optional: Test für Stabilität über mehrere Läufe
    # @pytest.mark.parametrize("run_index", range(3)) # Beispiel: 3 mal ausführen
    # def test_parallel_stability_multiple_runs(self, parallel_test_config: SimulationConfig, run_index: int):
    #     """
    #     Führt die parallele Simulation mehrmals aus, um auf nicht-deterministische
    #     Fehler (Race Conditions) zu prüfen. Vergleicht nicht die Ergebnisse,
    #     prüft nur auf Abstürze.
    #     """
    #     logging.info(f"Starte parallelen Stabilitätslauf Nr. {run_index + 1}...")
    #     try:
    #          # Führe nur aus, prüfe nicht das Ergebnis im Detail
    #          self.run_simulation_and_collect(parallel_test_config, parallel=True)
    #          logging.info(f"Paralleler Stabilitätslauf Nr. {run_index + 1} erfolgreich.")
    #          assert True # Test erfolgreich, wenn keine Exception auftritt
    #     except Exception as e:
    #          pytest.fail(f"Parallele Simulation (Lauf {run_index + 1}) fehlgeschlagen: {e}", pytrace=True)