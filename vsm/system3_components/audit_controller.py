# Impaler/vsm/system3_components/audit_controller.py
"""
Komponente für die Durchführung von System 3* Audits.

Diese Klasse kapselt die Logik zur Auswahl von Audit-Zielen,
der Initiierung von Audits durch die RegionalSystem3Manager und
der Verarbeitung der Ergebnisse, einschließlich der Anwendung von Strafen.
"""

import logging
import random
from typing import Dict, List, Any, Optional, Set, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    # Type Hinting Imports ohne Zirkelbezüge zur Laufzeit
    from ..system3 import System3Manager, RegionalSystem3Manager

    # Import ProducerAgent nur für Type Hinting bei agent lookup
    from ...agents.producer import ProducerAgent


class System3AuditController:
    """
    Verwaltet den S3* Audit-Prozess innerhalb des System 3 VSM-Frameworks.
    Selektiert Audit-Ziele, koordiniert die Durchführung mit den RegionalManagern
    und verarbeitet die Audit-Ergebnisse zur Anwendung von Maßnahmen.
    """

    def __init__(self, s3_manager: "System3Manager"):
        """
        Initialisiert den AuditController.

        Args:
            s3_manager: Eine Referenz auf die übergeordnete System3Manager-Instanz.
        """
        self.s3_manager: "System3Manager" = s3_manager
        self.logger: logging.Logger = s3_manager.logger.getChild("AuditController")
        # Zugriff auf die System3-Konfiguration für Audit-Parameter
        # System3Manager.config enthält bereits die S3-spezifischen Parameter
        self.config: Dict[str, Any] = s3_manager.config

        # Zustandsspeicher für den Audit-Prozess
        # Hält manuell gesetzte Ziele für den nächsten Lauf
        self.audit_target_producers: Set[str] = set()
        # Speichert Ergebnisse des letzten durchgeführten Audit-Zyklus
        self.last_audit_results: List[Dict[str, Any]] = []

        self.logger.debug(
            f"System3AuditController initialisiert (Audit Aktiv: {self.config.get('audit_active', False)})"
        )

    def run_audit_cycle(self) -> List[Dict[str, Any]]:
        """
        Führt einen kompletten Audit-Zyklus durch: Zielauswahl, Durchführung, Analyse.
        Diese Methode wird typischerweise von System3Manager in der 'audit'-Stage aufgerufen.

        Returns:
            Eine Liste der gesammelten Audit-Ergebnisse aus diesem Zyklus.
        """
        # Prüfe, ob Audits global aktiviert sind
        if not self.config.get("audit_active", False):
            self.logger.debug("Audit-Zyklus übersprungen (global deaktiviert).")
            return []

        self.logger.info("Starte Audit-Zyklus (System 3*)...")
        self.last_audit_results = []  # Ergebnisse für diesen Lauf zurücksetzen

        # 1. Wähle die Producer aus, die auditiert werden sollen
        targets = self._select_audit_targets()
        if not targets:
            self.logger.info("Keine Audit-Ziele für diesen Zyklus ausgewählt.")
            return []

        self.logger.info(f"Audit-Ziele ausgewählt ({len(targets)}): {list(targets)}")

        # 2. Führe Audits über die zuständigen RegionalManager durch
        audit_results_list: List[Dict[str, Any]] = []
        for producer_id in targets:
            rm = self._find_regional_manager_for_producer(producer_id)
            # Stelle sicher, dass RM existiert und die Methode hat
            if rm and hasattr(rm, "audit_producer"):
                try:
                    result = rm.audit_producer(
                        producer_id
                    )  # RM führt eigentliches Audit durch
                    if result and isinstance(result, dict):
                        # Füge IDs hinzu für leichtere Zuordnung
                        result["producer_id"] = producer_id
                        result["region_id"] = rm.region_id
                        audit_results_list.append(result)
                    elif result:
                        self.logger.warning(
                            f"Audit von Producer {producer_id} in Region {rm.region_id} gab unerwartetes Ergebnis zurück (kein Dict): {type(result)}"
                        )
                except Exception as e:
                    self.logger.error(
                        f"Fehler beim Ausführen des Audits von Producer {producer_id} in Region {rm.region_id}: {e}",
                        exc_info=False,
                    )
            elif not rm:
                self.logger.warning(
                    f"Konnte keinen RegionalManager für Audit von Producer {producer_id} finden."
                )
            else:  # rm existiert, aber Methode fehlt
                self.logger.error(
                    f"RegionalManager für Region {getattr(rm, 'region_id', 'Unbekannt')} fehlt die Methode 'audit_producer'. Audit für {producer_id} übersprungen."
                )

        # 3. Analysiere gesammelte Ergebnisse und leite Maßnahmen ein
        if audit_results_list:
            self._analyze_and_apply_audit_results(audit_results_list)
            self.last_audit_results = (
                audit_results_list  # Speichere letzte Ergebnisse hier
            )

            # Optional: Aktualisiere zentrale History im S3 Manager, falls gewünscht
            if hasattr(self.s3_manager, "audit_results") and isinstance(
                self.s3_manager.audit_results, list
            ):
                self.s3_manager.audit_results.extend(audit_results_list)

        # 4. Manuell gesetzte Ziele für nächsten Lauf zurücksetzen
        self.audit_target_producers.clear()

        self.logger.info(
            f"Audit-Zyklus abgeschlossen. {len(audit_results_list)} Audits durchgeführt und analysiert."
        )
        return self.last_audit_results

    # ---------------------------------------------------------------
    # Private Hilfsmethoden
    # (Logik aus dem ursprünglichen System3Manager übernommen und angepasst)
    # ---------------------------------------------------------------

    def _select_audit_targets(self) -> Set[str]:
        """
        Wählt Producer für den nächsten Audit-Lauf aus.
        Nutzt manuell gesetzte Ziele oder eine automatische Auswahlstrategie.
        """
        # Wenn Ziele manuell gesetzt wurden, diese verwenden
        if self.audit_target_producers:
            self.logger.debug(
                f"Verwende manuell gesetzte Audit-Ziele: {self.audit_target_producers}"
            )
            return self.audit_target_producers.copy()  # Kopie zurückgeben

        # Automatische Auswahlstrategie (Beispiel: Zufällig + Problemfälle)
        # Zugriff auf Agentenliste über das Model im S3 Manager
        all_agents = self.s3_manager.model.agents_by_id
        if not all_agents:
            self.logger.warning(
                "Keine Agenten im Modell für Audit-Zielauswahl gefunden."
            )
            return set()

        # Finde alle Producer Agents
        # Wichtig: Importiere ProducerAgent nur für Type Checking!
        from ...agents.producer import ProducerAgent  # Import für isinstance Check

        producer_ids = [
            pid for pid, agent in all_agents.items() if isinstance(agent, ProducerAgent)
        ]

        if not producer_ids:
            self.logger.info("Keine Producer im Modell gefunden, keine Audit-Ziele.")
            return set()

        # Konfigurierbare Parameter für Auswahl holen
        audit_sample_fraction = self.config.get(
            "audit_sample_fraction", 0.05
        )  # z.B. 5% auditieren
        min_audit_count = self.config.get("min_audit_count", 1)
        max_audit_count = self.config.get("max_audit_count", 10)  # Obergrenze
        # TODO: Strategie für Problemfälle implementieren (z.B. basierend auf History oder RM Feedback)
        problem_case_ids: Set[str] = set()  # Platzhalter

        # Berechne Anzahl der zu auditierenden Producer
        num_total_producers = len(producer_ids)
        num_to_audit = int(num_total_producers * audit_sample_fraction)
        num_to_audit = max(min_audit_count, num_to_audit)  # Mindestanzahl
        num_to_audit = min(max_audit_count, num_to_audit)  # Maximalanzahl
        num_to_audit = min(
            num_to_audit, num_total_producers
        )  # Nicht mehr als vorhanden

        # Kombiniere Problemfälle und zufällige Auswahl
        selected_targets = set(problem_case_ids)
        remaining_slots = num_to_audit - len(selected_targets)

        if remaining_slots > 0:
            # Wähle zufällig aus den *noch nicht* ausgewählten Producern
            available_random_ids = [
                pid for pid in producer_ids if pid not in selected_targets
            ]
            num_random_to_select = min(remaining_slots, len(available_random_ids))
            if num_random_to_select > 0:
                random_selection = random.sample(
                    available_random_ids, num_random_to_select
                )
                selected_targets.update(random_selection)

        self.logger.debug(
            f"Automatisch ausgewählte Audit-Ziele ({len(selected_targets)}): {list(selected_targets)}"
        )
        return selected_targets

    def _find_regional_manager_for_producer(
        self, producer_id: str
    ) -> Optional["RegionalSystem3Manager"]:
        """
        Findet den zuständigen RegionalSystem3Manager für einen gegebenen Producer.
        Delegiert idealerweise an den S3Manager oder repliziert die Logik.
        """
        # Versuche, Methode im S3Manager aufzurufen
        if hasattr(self.s3_manager, "_find_regional_manager_for_producer"):
            return self.s3_manager._find_regional_manager_for_producer(producer_id)
        else:
            # Eigene Implementierung als Fallback (kopiert aus S3M)
            self.logger.warning(
                "Methode '_find_regional_manager_for_producer' nicht in S3Manager gefunden. Verwende Fallback im AuditController."
            )
            agent = self.s3_manager.model.agents_by_id.get(producer_id)
            if agent and hasattr(agent, "region_id"):
                region_id = agent.region_id
                return self.s3_manager.regional_managers.get(region_id)
            self.logger.debug(
                f"Konnte Agenten {producer_id} oder seine Region nicht finden."
            )
            return None

    def _analyze_and_apply_audit_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Analysiert die gesammelten Audit-Ergebnisse und weist die zuständigen
        RegionalManager an, notwendige Maßnahmen (Strafen) zu ergreifen.
        """
        self.logger.info(f"Analysiere {len(results)} Audit-Ergebnisse...")
        applied_penalties_count = 0

        # Schwellwerte für Strafen aus Config holen
        waste_threshold = self.config.get("audit_waste_threshold", 0.25)
        waste_penalty_factor = self.config.get(
            "audit_waste_penalty_factor", 0.85
        )  # 15% weniger Ressourcen
        util_threshold = self.config.get("audit_underutil_threshold", 0.4)
        util_penalty_factor = self.config.get(
            "audit_underutil_penalty_factor", 0.90
        )  # 10% weniger Ziel

        for result in results:
            producer_id = result.get("producer_id")
            region_id = result.get("region_id")
            if not producer_id or not region_id:
                self.logger.warning(
                    f"Unvollständiges Audit-Ergebnis übersprungen: {result}"
                )
                continue

            # Finde zuständigen RM
            rm = self.s3_manager.regional_managers.get(region_id)
            if not rm:
                self.logger.warning(
                    f"Kein RM für Region {region_id} gefunden, um Audit-Ergebnis für {producer_id} anzuwenden."
                )
                continue
            if not hasattr(rm, "apply_producer_penalties"):
                self.logger.error(
                    f"RM für Region {region_id} fehlt Methode 'apply_producer_penalties'."
                )
                continue

            penalties_to_apply: List[Dict[str, Any]] = []

            # 1. Check für Ressourcenverschwendung
            waste = result.get("resource_waste", 0.0)
            if waste > waste_threshold:
                self.logger.warning(
                    f"Audit: Hohe Ressourcenverschwendung ({waste:.1%}) bei Producer {producer_id} (Region {region_id})."
                )
                penalties_to_apply.append(
                    {"type": "resource_allocation", "factor": waste_penalty_factor}
                )

            # 2. Check für Kapazitätsunternutzung
            underutil = result.get("capacity_underutilization", 0.0)
            if underutil > util_threshold:
                self.logger.warning(
                    f"Audit: Hohe Kapazitätsunternutzung ({underutil:.1%}) bei Producer {producer_id} (Region {region_id})."
                )
                penalties_to_apply.append(
                    {"type": "production_target", "factor": util_penalty_factor}
                )

            # 3. Weitere Checks (z.B. Emissionsüberschreitung, Qualitätsmängel) könnten hier folgen

            # 4. Wende gesammelte Strafen über den RegionalManager an
            if penalties_to_apply:
                try:
                    rm.apply_producer_penalties(producer_id, penalties_to_apply)
                    applied_penalties_count += len(penalties_to_apply)
                    self.logger.info(
                        f"Strafen für Producer {producer_id} an RM {region_id} übermittelt: {penalties_to_apply}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Fehler beim Anwenden der Strafen für Producer {producer_id} durch RM {region_id}: {e}"
                    )

        if applied_penalties_count > 0:
            self.logger.info(
                f"Analyse der Audit-Ergebnisse abgeschlossen. {applied_penalties_count} Strafen angewendet/initiiert."
            )
        else:
            self.logger.debug(
                "Analyse der Audit-Ergebnisse abgeschlossen. Keine Strafen angewendet."
            )
