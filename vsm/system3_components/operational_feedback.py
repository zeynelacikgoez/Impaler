# Impaler/vsm/system3_components/operational_feedback.py
"""
Komponente für die Verwaltung interner Feedback-Zyklen in System 3.

Sammelt Performance-Metriken von regionalen Managern, entscheidet über
notwendige Anpassungen (Ressourcenumverteilung, Prioritäten, Zielvorschläge)
und initiiert deren Anwendung über den System3Manager.
"""

import logging
import numpy as np
from collections import defaultdict, deque # deque falls für History in Metriken benötigt
from typing import Dict, List, Any, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    # Type Hinting für System3Manager und RegionalSystem3Manager
    # ohne tatsächlichen Import zur Laufzeit, um Zirkelbezüge zu vermeiden.
    from ..system3 import System3Manager, RegionalSystem3Manager
    # Importiere Contracts, falls für Feedback nötig (hier nicht direkt, aber ggf. in S3M)
    # from ..communication_contracts import SystemFeedback

class OperationalFeedbackLoop:
    """
    Verwaltet interne operative Feedback-Zyklen zur schnellen Anpassung
    innerhalb eines Simulationsschritts von System 3.
    """

    def __init__(self, s3_manager: 'System3Manager'):
        """
        Initialisiert den OperationalFeedbackLoop.

        Args:
            s3_manager: Eine Referenz auf die übergeordnete System3Manager-Instanz.
        """
        self.s3_manager: 'System3Manager' = s3_manager
        self.logger: logging.Logger = s3_manager.logger.getChild('FeedbackLoop')
        # Hole Konfiguration aus dem S3 Manager
        self.config: Dict[str, Any] = getattr(s3_manager.config, "system3_params", {})

        self.max_cycles: int = self.config.get("max_feedback_cycles", 3)
        self.stability_factor: float = self.config.get("stability_factor", 0.7)

        # Zustand für den aktuellen Simulationsschritt
        self.feedback_cycles_this_step: int = 0
        self.last_calculated_adjustments: Dict[str, Dict] = {} # Speichert letzte Anpassungen für Analyse/Debugging

        self.logger.debug(f"OperationalFeedbackLoop initialisiert (Max Cycles: {self.max_cycles}, Stability: {self.stability_factor})")

    def run_cycles(self) -> Dict[str, Any]:
        """
        Startet und verwaltet die Feedback-Zyklen für den aktuellen Simulationsschritt.

        Returns:
            Ein Dictionary mit einer Zusammenfassung der durchgeführten Zyklen und Anpassungen.
        """
        self.feedback_cycles_this_step = 0
        self.last_calculated_adjustments = {}
        summary: Dict[str, Any] = {"cycles_run": 0, "adjustments_made": 0, "adjustments_proposed": {}}

        self._run_single_cycle(summary) # Starte rekursiven Loop

        self.logger.info(f"Feedback-Zyklen abgeschlossen ({summary['cycles_run']}/{self.max_cycles}). "
                         f"{summary['adjustments_made']} Anpassungen initiiert.")
        # Optional: Gib die tatsächlich berechneten Anpassungen zurück
        summary['adjustments_proposed'] = self.last_calculated_adjustments
        return summary

    def _run_single_cycle(self, summary: Dict[str, Any]) -> None:
        """Führt einen einzelnen Feedback-Zyklus rekursiv aus."""
        if self.feedback_cycles_this_step >= self.max_cycles:
            self.logger.debug(f"Maximale Feedback-Zyklen ({self.max_cycles}) erreicht.")
            return

        self.feedback_cycles_this_step += 1
        summary["cycles_run"] = self.feedback_cycles_this_step
        self.logger.debug(f"Starte Feedback-Zyklus {self.feedback_cycles_this_step}/{self.max_cycles}")

        # 1. Metriken sammeln
        performance_metrics = self._collect_performance_metrics()

        # 2. Prüfen, ob Anpassungen nötig sind
        if self._requires_adjustments(performance_metrics):
            self.logger.info("Anpassungen im Feedback-Zyklus erforderlich.")

            # 3. Anpassungen berechnen
            adjustments = self._calculate_adjustments(performance_metrics)
            self.last_calculated_adjustments = adjustments # Für Analyse speichern

            # 4. Anpassungen über S3Manager anwenden lassen
            adjustments_applied_count = self._apply_adjustments(adjustments)
            summary["adjustments_made"] += adjustments_applied_count

            # 5. Rekursiver Aufruf für nächsten Zyklus, falls Anpassungen erfolgt sind
            #    oder bis max_cycles erreicht ist.
            if adjustments_applied_count > 0:
                self._run_single_cycle(summary)
            else:
                 self.logger.debug("Keine weiteren Anpassungen angewendet in diesem Zyklus, stoppe Feedback-Loop.")

        else:
            self.logger.debug("Keine weiteren Anpassungen in diesem Feedback-Zyklus benötigt.")


    # ---------------------------------------------------------------
    # Private Methoden zur Metriksammlung und Anpassungsberechnung
    # (Diese Methoden entsprechen denen aus dem ursprünglichen System3Manager)
    # ---------------------------------------------------------------

    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Sammelt aktuelle Leistungsmetriken von allen RegionalManagern."""
        metrics: Dict[str, Any] = {
            'resource_utilization': defaultdict(list), # {resource: [(region, utilization)]}
            'production_fulfillment': defaultdict(list), # {good: [(region, fulfillment_ratio)]}
            'regional_stress': {}, # {region: stress_level}
            'critical_shortages': [] # Liste kritischer Ressourcen
        }
        all_regional_metrics = {}

        # Sammle Daten von allen RMs über den S3Manager
        for region_id, rm in self.s3_manager.regional_managers.items():
            if hasattr(rm, 'get_performance_metrics'):
                try:
                    rm_metrics = rm.get_performance_metrics()
                    all_regional_metrics[region_id] = rm_metrics

                    # Aggregiere
                    for res, util in rm_metrics.get('resource_utilization', {}).items():
                         metrics['resource_utilization'][res].append((region_id, util))
                    for good, fulfill in rm_metrics.get('production_fulfillment', {}).items():
                         metrics['production_fulfillment'][good].append((region_id, fulfill))
                    metrics['regional_stress'][region_id] = rm_metrics.get('stress_level', 0.0)
                except Exception as e:
                    self.logger.error(f"Fehler beim Abrufen der Performance-Metriken von Region {region_id}: {e}")
            else:
                self.logger.warning(f"RegionalManager für Region {region_id} hat keine 'get_performance_metrics' Methode.")


        # Identifiziere globale kritische Engpässe
        # (Logik aus S3M._collect_performance_metrics übernehmen)
        for resource, regional_utils in metrics['resource_utilization'].items():
            if not regional_utils: continue # Leere Liste überspringen

            util_values = [u for _, u in regional_utils]
            if not util_values: continue # Keine gültigen Werte überspringen

            max_util = max(util_values)
            min_util = min(util_values)

            # Definition von "kritisch": z.B. hohe Auslastung in mind. einer Region
            # oder starkes Ungleichgewicht
            if max_util > 0.95: # Hohe Spitzenauslastung
                if resource not in metrics['critical_shortages']: metrics['critical_shortages'].append(resource)
            elif len(regional_utils) > 1:
                if max_util > 0.9 and min_util < 0.5 and (max_util - min_util) > 0.35 : # Starkes Ungleichgewicht (Schwellwerte ggf. anpassen)
                    if resource not in metrics['critical_shortages']: metrics['critical_shortages'].append(resource)

        self.logger.debug(f"Performance-Metriken gesammelt. Kritische Ressourcen: {metrics['critical_shortages']}")
        return metrics


    def _requires_adjustments(self, metrics: Dict[str, Any]) -> bool:
        """Entscheidet, ob Anpassungen aufgrund der Metriken notwendig sind."""
        # Kriterium 1: Kritische Ressourcenknappheiten
        if metrics.get('critical_shortages'):
            self.logger.debug("Anpassung erforderlich: Kritische Ressourcenknappheiten erkannt.")
            return True

        # Kriterium 2: Starke regionale Ungleichgewichte bei Ressourcennutzung
        for resource, regional_data in metrics.get('resource_utilization', {}).items():
            if len(regional_data) > 1:
                utils = [util for _, util in regional_data]
                # Prüfe auf signifikante Spanne und hohe/niedrige Werte
                if max(utils) > 0.9 and min(utils) < 0.6 and (max(utils) - min(utils)) > 0.4:
                    self.logger.debug(f"Anpassung erforderlich: Hohe Imbalance bei Nutzung von {resource}.")
                    return True

        # Kriterium 3: Signifikante Abweichungen bei der Produktionserfüllung
        for good, regional_data in metrics.get('production_fulfillment', {}).items():
            for _, fulfillment in regional_data:
                 # Prüfe auf starke Unter- oder Übererfüllung
                 if fulfillment < 0.75 or fulfillment > 1.25:
                     self.logger.debug(f"Anpassung erforderlich: Signifikante Planabweichung für {good} (Wert: {fulfillment:.2f}).")
                     return True

        self.logger.debug("Keine dringenden Anpassungen aus Metriken erforderlich.")
        return False # Keine dringenden Anpassungen nötig


    def _calculate_adjustments(self, metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Berechnet konkrete Anpassungsvorschläge basierend auf den Metriken.
        """
        adjustments: Dict[str, Dict[str, Any]] = {
            'resource_redistribution': {}, # {resource: {from_region: x, to_region: y, amount: z}}
            'priority_adjustments': {},    # {good: new_priority}
            'target_adjustments': {}       # {good: new_target_factor} - Faktor statt absolutem Wert
        }

        # 1. Ressourcenumverteilung bei kritischen Engpässen / Ungleichgewichten
        adjustments['resource_redistribution'] = self._calculate_redistribution_adjustments(metrics)

        # 2. Prioritätsanpassungen bei Unter-/Übererfüllung
        adjustments['priority_adjustments'] = self._calculate_priority_adjustments(metrics)

        # 3. Zielanpassungsfaktoren (für S4 vorschlagen)
        adjustments['target_adjustments'] = self._calculate_target_adjustment_factors(metrics)

        self.logger.debug(f"Anpassungen berechnet: "
                          f"{len(adjustments['resource_redistribution'])} Umverteilungen, "
                          f"{len(adjustments['priority_adjustments'])} Prio-Änderungen, "
                          f"{len(adjustments['target_adjustments'])} Ziel-Faktor-Vorschläge.")
        return adjustments


    def _calculate_redistribution_adjustments(self, metrics: Dict[str, Any]) -> Dict[str, Dict]:
        """Hilfsfunktion zur Berechnung von Ressourcenumverteilungen."""
        redistributions = {}
        utilization_data = metrics.get('resource_utilization', {})

        for resource in metrics.get('critical_shortages', []):
            # Hole Nutzungsdaten für diese Ressource
            util_tuples = utilization_data.get(resource, [])
            if len(util_tuples) > 1: # Nur sinnvoll bei mehr als einer Region
                # Sortiere nach Nutzung (niedrig -> hoch)
                sorted_utils = sorted(util_tuples, key=lambda x: x[1])
                low_region, low_util = sorted_utils[0]
                high_region, high_util = sorted_utils[-1]

                # Nur umverteilen, wenn signifikantes Delta und niedrige Region "Spielraum" hat
                # und hohe Region Bedarf signalisiert (>80% ist guter Indikator)
                if high_util > 0.8 and low_util < 0.7 and (high_util - low_util > 0.2):
                    # Frage verfügbare Menge von niedrig ausgelasteter Region ab (via S3M)
                    low_rm = self.s3_manager.regional_managers.get(low_region)
                    if low_rm and hasattr(low_rm, 'get_resource_amount'):
                         try:
                             available_amount = float(low_rm.get_resource_amount(resource))
                         except Exception:
                             available_amount = 0.0
                         # Menge zum Umverteilen: z.B. 10% des Bestands der niedrigen Region,
                         # skaliert mit der Nicht-Auslastung
                         transfer_amount = available_amount * 0.1 * (1.0 - low_util)

                         if transfer_amount > 1e-3: # Nur signifikante Mengen
                             redistributions[resource] = {
                                 'from_region': low_region,
                                 'to_region': high_region,
                                 'amount': transfer_amount
                             }
                             self.logger.debug(f"  Redistribution für {resource}: Plane Transfer von {transfer_amount:.2f} aus R{low_region} (Util: {low_util:.1%}) nach R{high_region} (Util: {high_util:.1%}).")
                    else:
                         self.logger.warning(f"Konnte RM für Region {low_region} nicht finden oder get_resource_amount fehlt für Redistribution.")

        return redistributions


    def _calculate_priority_adjustments(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Hilfsfunktion zur Berechnung von Prioritätsanpassungen."""
        priority_adjustments = {}
        fulfillment_data = metrics.get('production_fulfillment', {})

        for good, regional_data in fulfillment_data.items():
            if not regional_data:
                continue

            avg_fulfillment = sum(f for _, f in regional_data) / len(regional_data)
            max_f = max(f for _, f in regional_data)
            min_f = min(f for _, f in regional_data)

            current_priority = self.s3_manager.production_priorities.get(good, 1.0)
            new_priority = current_priority

            if avg_fulfillment < 0.85:
                increase = (1.0 / max(0.1, avg_fulfillment) - 1.0) * 0.5 * (1.0 - self.stability_factor)
                new_priority = min(3.0, current_priority * (1.0 + increase))
            elif avg_fulfillment > 1.15:
                decrease = (1.0 - 1.0 / avg_fulfillment) * 0.3 * (1.0 - self.stability_factor)
                new_priority = max(0.1, current_priority * (1.0 - decrease))
            elif max_f - min_f > 0.3:
                spread = max_f - min_f
                adjust = 0.05 * (spread - 0.3)
                if max_f > 1.0:
                    new_priority = max(0.1, current_priority * (1.0 - adjust))
                else:
                    new_priority = min(3.0, current_priority * (1.0 + adjust))

            if abs(new_priority - current_priority) > 0.01:
                priority_adjustments[good] = new_priority
                self.logger.debug(
                    f"  Prio-Anpassung für {good}: {current_priority:.2f} -> {new_priority:.2f} (Avg Fulfillment: {avg_fulfillment:.2f})."
                )

        return priority_adjustments


    def _calculate_target_adjustment_factors(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Hilfsfunktion zur Berechnung von Zielanpassungsfaktoren (für S4)."""
        target_adjustments = {}
        fulfillment_data = metrics.get('production_fulfillment', {})

        for good, regional_data in fulfillment_data.items():
            if not regional_data: continue

            avg_fulfillment = sum(f for _, f in regional_data) / len(regional_data)

            # Nur anpassen, wenn Ziel signifikant verfehlt wird
            if abs(avg_fulfillment - 1.0) > 0.1: # 10% Abweichung
                # Faktor vorschlagen: 1/Erfüllung, gedämpft
                factor = 1.0 / max(0.1, avg_fulfillment)
                # Dämpfung anwenden
                adjusted_factor = 1.0 + (factor - 1.0) * (1.0 - self.stability_factor)
                # Nur sinnvolle Faktoren speichern (z.B. nicht extrem hoch/niedrig)
                if 0.5 < adjusted_factor < 2.0:
                     target_adjustments[good] = adjusted_factor
                     self.logger.debug(f"  Ziel-Faktor für {good}: Schlage {adjusted_factor:.2f} vor (Avg Fulfillment: {avg_fulfillment:.2f}).")

        return target_adjustments

    # ---------------------------------------------------------------
    # Private Methode zur Delegation der Anwendung
    # ---------------------------------------------------------------

    def _apply_adjustments(self, adjustments: Dict[str, Dict[str, Any]]) -> int:
        """
        Delegiert die Anwendung der berechneten Anpassungen an den System3Manager.

        Args:
            adjustments: Das Dictionary mit den berechneten Anpassungen.

        Returns:
            Die Anzahl der erfolgreich *initiierten* Anpassungen.
        """
        applied_count = 0

        # 1. Ressourcenumverteilung anwenden
        # Rufe Methode im S3 Manager auf, um Transfer auszuführen
        if hasattr(self.s3_manager, 'execute_inter_regional_transfer'):
            for resource, data in adjustments.get('resource_redistribution', {}).items():
                if data.get('amount', 0.0) > 1e-6:
                    success = self.s3_manager.execute_inter_regional_transfer(
                        from_region=data['from_region'],
                        to_region=data['to_region'],
                        resource=resource,
                        amount=data['amount']
                    )
                    if success:
                        self.logger.info(f"  Feedback Angewendet: Transfer von {data['amount']:.2f} {resource} initiiert.")
                        applied_count += 1
                    else:
                        self.logger.warning(f"  Feedback Problem: Transfer von {resource} via S3 Manager fehlgeschlagen.")
        else:
            if adjustments.get('resource_redistribution'):
                self.logger.error("S3Manager fehlt Methode 'execute_inter_regional_transfer'. Umverteilung nicht möglich.")

        # 2. Prioritätsanpassungen anwenden
        priority_updates = adjustments.get('priority_adjustments', {})
        if priority_updates:
            if hasattr(self.s3_manager, 'apply_priority_updates'):
                updated_count = self.s3_manager.apply_priority_updates(priority_updates)
                applied_count += updated_count
                if updated_count > 0:
                    self.logger.info(f"  Feedback Angewendet: {updated_count} Prioritäten aktualisiert.")
            else:
                self.logger.error("S3Manager fehlt Methode 'apply_priority_updates'.")

        # 3. Zielanpassungsfaktoren an S4 senden
        target_factors = adjustments.get('target_adjustments', {})
        if target_factors:
            if hasattr(self.s3_manager, 'send_target_factor_feedback'):
                feedback_sent = self.s3_manager.send_target_factor_feedback(target_factors)
                if feedback_sent:
                    applied_count += 1 # Zählen das Senden als Erfolg
                    self.logger.info("  Feedback Angewendet: Zielanpassungsfaktoren an S4 gesendet.")
            else:
                self.logger.error("S3Manager fehlt Methode 'send_target_factor_feedback'.")

        return applied_count