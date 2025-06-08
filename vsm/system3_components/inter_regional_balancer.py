# Impaler/vsm/system3_components/inter_regional_balancer.py
"""
Komponente für den interregionalen Ressourcenausgleich in System 3.

Berechnet optimale (via SciPy, falls verfügbar) oder heuristische
Ressourcentransfers zwischen Regionen und initiiert diese über
die entsprechenden RegionalSystem3Manager-Instanzen.
"""

import logging
import numpy as np
import math  # Wird für Logarithmus im Scipy Objective benötigt
import random
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union, TYPE_CHECKING

# Verwende die SciPy-Variable und Funktion aus dem Haupt-System3-Modul,
# damit Tests diese leichter patchen können.
try:
    from ..system3 import minimize, SCIPY_AVAILABLE
    from scipy.optimize import OptimizeResult
except Exception:  # pragma: no cover - falls system3 den Import nicht bereitstellt
    try:
        from scipy.optimize import minimize, OptimizeResult
        SCIPY_AVAILABLE = True
    except ImportError:  # pragma: no cover - SciPy nicht verfügbar
        SCIPY_AVAILABLE = False
        minimize = None  # type: ignore

if TYPE_CHECKING:
    # Zirkulären Import zur Laufzeit vermeiden, aber Type Hinting ermöglichen
    from ..system3 import System3Manager, RegionalSystem3Manager

class InterRegionalResourceBalancer:
    """
    Berechnet und initiiert Ressourcentransfers zwischen Regionen, um
    Knappheiten und Überschüsse auszugleichen.
    """

    def __init__(self, s3_manager: 'System3Manager'):
        """
        Initialisiert den InterRegionalResourceBalancer.

        Args:
            s3_manager: Eine Referenz auf die übergeordnete System3Manager-Instanz.
        """
        self.s3_manager: 'System3Manager' = s3_manager
        self.logger: logging.Logger = s3_manager.logger.getChild('InterRegionalBalancer')
        # Zugriff auf die System3-Konfiguration
        self.config: Dict[str, Any] = getattr(s3_manager.config, "system3_params", {})

        # Prüfe, ob fortgeschrittene Optimierung genutzt werden soll und kann
        self.enable_advanced_optimization: bool = self.config.get("enable_advanced_optimization", True) and SCIPY_AVAILABLE
        if self.config.get("enable_advanced_optimization", True) and not SCIPY_AVAILABLE:
            self.logger.warning("SciPy nicht gefunden. Fortgeschrittene Optimierung für Ressourcenausgleich ist deaktiviert.")

        self.logger.debug(f"InterRegionalResourceBalancer initialisiert (Advanced Opt: {self.enable_advanced_optimization})")


    def balance_resources(self) -> List[Dict[str, Any]]:
        """
        Hauptmethode zum Ausführen des interregionalen Ressourcenausgleichs.

        Sammelt Bedarfe und Überschüsse von allen Regionen, berechnet die
        notwendigen Transfers (optimiert oder proportional) und weist die
        RegionalSystem3Manager an, die Transfers durchzuführen.

        Returns:
            Eine Liste von Log-Einträgen über die versuchten und durchgeführten Transfers.
            Format: [{"resource": str, "transfers_calculated": int, "transfers_applied": int}]
        """
        # Kein Ausgleich nötig bei 0 oder 1 Region
        if len(self.s3_manager.regions) <= 1:
            self.logger.debug("Weniger als zwei Regionen vorhanden, kein interregionaler Ausgleich erforderlich.")
            return []

        self.logger.info("Starte inter-regionalen Ressourcenausgleich...")

        # 1. Sammle Bedarf, Überschuss und Priorität pro Region
        regional_needs: Dict[str, Dict[Union[int, str], float]] = defaultdict(dict)
        regional_surpluses: Dict[str, Dict[Union[int, str], float]] = defaultdict(dict)
        regional_priorities: Dict[Union[int, str], float] = {}

        for region_id, rm in self.s3_manager.regional_managers.items():
            # Diese Methoden müssen im RegionalSystem3Manager implementiert sein!
            if not all(hasattr(rm, method) for method in ['calculate_resource_needs', 'calculate_resource_surpluses', 'calculate_regional_priority']):
                 self.logger.error(f"RegionalSystem3Manager für Region {region_id} fehlen benötigte Methoden für den Ressourcenausgleich.")
                 continue # Überspringe diese Region

            needs = rm.calculate_resource_needs()
            surpluses = rm.calculate_resource_surpluses()
            regional_priorities[region_id] = rm.calculate_regional_priority()

            for resource, amount in needs.items():
                if amount > 1e-6: regional_needs[resource][region_id] = amount # Nur signifikanten Bedarf
            for resource, amount in surpluses.items():
                if amount > 1e-6: regional_surpluses[resource][region_id] = amount # Nur signifikanten Überschuss

        # 2. Führe Ausgleich für jede relevante Ressource durch
        transfer_log: List[Dict[str, Any]] = []
        # Finde alle Ressourcen, für die es Bedarf UND Überschuss gibt
        resources_to_balance = set(regional_needs.keys()) & set(regional_surpluses.keys())

        for resource in resources_to_balance:
            needy_regions = regional_needs[resource]
            supplier_regions = regional_surpluses[resource]

            # Prüfe, ob überhaupt Bedarf oder Überschuss für diese Ressource existiert
            if not needy_regions or not supplier_regions:
                continue

            total_needed = sum(needy_regions.values())
            total_surplus = sum(supplier_regions.values())

            if total_needed <= 1e-6 or total_surplus <= 1e-6:
                continue # Nichts zu tun für diese Ressource

            amount_to_transfer = min(total_needed, total_surplus)

            # 3. Berechne die spezifischen Transfers
            calculated_transfers = self._calculate_transfers(
                resource, needy_regions, supplier_regions, regional_priorities, amount_to_transfer
            )

            # 4. Wende die berechneten Transfers an
            applied_count = 0
            if calculated_transfers:
                 self.logger.debug(f"Berechnete Transfers für {resource}: {len(calculated_transfers)}")
                 for from_region, to_region, amount in calculated_transfers:
                     # Nutze die Hilfsmethode im S3 Manager (oder hier implementiert)
                     if hasattr(self.s3_manager, 'execute_inter_regional_transfer'):
                         success = self.s3_manager.execute_inter_regional_transfer(from_region, to_region, resource, amount)
                         if success:
                             applied_count += 1
                     else:
                         # Fallback: Direkter Aufruf (wenn execute_inter_regional_transfer nicht in S3M existiert)
                         success = self._execute_single_transfer(from_region, to_region, resource, amount)
                         if success: applied_count +=1

                 # Logge das Ergebnis für diese Ressource
                 transfer_log.append({
                     "resource": resource,
                     "transfers_calculated": len(calculated_transfers),
                     "transfers_applied": applied_count
                 })

        # Abschließendes Logging
        if transfer_log:
            summary = ", ".join([f"{item['resource']}: {item['transfers_applied']}/{item['transfers_calculated']}" for item in transfer_log])
            self.logger.info(f"Inter-regionale Transfers durchgeführt für: {summary}")
        else:
            self.logger.info("Keine inter-regionalen Transfers durchgeführt (kein Bedarf/Überschuss oder Fehler).")

        return transfer_log


    def _execute_single_transfer(self, from_region_id: Union[int, str], to_region_id: Union[int, str], resource: str, amount: float) -> bool:
         """
         Führt einen einzelnen Transfer direkt über die RegionalManager aus.
         Dies ist ein Fallback, falls die Methode nicht im S3Manager existiert.
         """
         from_rm = self.s3_manager.regional_managers.get(from_region_id)
         to_rm = self.s3_manager.regional_managers.get(to_region_id)

         if not from_rm or not to_rm:
              self.logger.error(f"Transfer fehlgeschlagen: Region(en) {from_region_id} oder {to_region_id} nicht gefunden.")
              return False

         # Prüfe, ob die RM die Methoden haben
         if not hasattr(from_rm, 'transfer_resource') or not hasattr(to_rm, 'receive_resource'):
              self.logger.error(f"Transfer fehlgeschlagen: Benötigte Methoden 'transfer_resource' oder 'receive_resource' fehlen bei RMs.")
              return False

         # Die transfer_resource Methode im RM prüft, ob genug vorhanden ist.
         success = from_rm.transfer_resource(resource, amount, to_region_id)
         if success:
             to_rm.receive_resource(resource, amount)
             # Optional: Tracking im S3 Manager aktualisieren (falls S3M die Logik nicht hat)
             if hasattr(self.s3_manager, 'resource_flows'):
                 self.s3_manager.resource_flows[resource][f"{from_region_id}->{to_region_id}"] = amount
             self.logger.info(f"Transfer ERFOLGREICH: {amount:.2f} {resource} von R{from_region_id} nach R{to_region_id}.")
             return True
         else:
             # RM hat den Transfer abgelehnt (vermutlich nicht genug Bestand)
             self.logger.warning(f"Transfer FEHLGESCHLAGEN (RM): {amount:.2f} {resource} von R{from_region_id} nach R{to_region_id}.")
             return False


    def _calculate_transfers(self,
        resource: str,
        needy_regions: Dict[Union[int, str], float],
        supplier_regions: Dict[Union[int, str], float],
        regional_priorities: Dict[Union[int, str], float],
        total_transfer: float
    ) -> List[Tuple[Union[int, str], Union[int, str], float]]:
        """
        Wählt die Berechnungsmethode (Optimierung oder Proportional)
        und gibt die Liste der Transfers zurück.
        """
        if total_transfer <= 1e-6: return []

        if self.enable_advanced_optimization:
            try:
                transfers = self._optimize_transfers_scipy(
                    resource, needy_regions, supplier_regions, regional_priorities, total_transfer
                )
                self.logger.debug(f"SciPy-Optimierung für {resource} genutzt.")
                return transfers
            except Exception as e:
                self.logger.warning(f"SciPy-Optimierung für {resource} fehlgeschlagen: {e}. Nutze proportionalen Fallback.", exc_info=False)
                # Expliziter Aufruf des Fallbacks
                return self._proportional_transfers(
                    resource, needy_regions, supplier_regions, regional_priorities, total_transfer
                )
        else:
            self.logger.debug(f"Nutze proportionalen Fallback für {resource} (Optimierung deaktiviert).")
            return self._proportional_transfers(
                resource, needy_regions, supplier_regions, regional_priorities, total_transfer
            )


    # ---------------------------------------------------------------
    # Implementierung der Berechnungsmethoden
    # (Hier wird der Code aus dem ursprünglichen system3.py eingefügt und angepasst)
    # ---------------------------------------------------------------

    def _optimize_transfers_scipy(self, resource: str, needy: Dict, suppliers: Dict, priorities: Dict, total_transfer: float) -> List[Tuple]:
        """
        Berechnet Transfers mittels Scipy-Optimierung (Minimierung der Kosten/Maximierung des Nutzens).
        Angepasste Version von S3M._optimize_inter_regional_transfers_scipy.
        """
        from .. import system3
        if not system3.SCIPY_AVAILABLE:
             self.logger.error("Versuch, SciPy zu nutzen, obwohl nicht verfügbar. Nutze Fallback.")
             # Expliziter Aufruf des Fallbacks
             return self._proportional_transfers(resource, needy, suppliers, priorities, total_transfer)

        # --- Setup für Scipy ---
        needy_ids = list(needy.keys())
        supplier_ids = list(suppliers.keys())
        num_needy = len(needy_ids)
        num_suppliers = len(supplier_ids)

        if num_needy == 0 or num_suppliers == 0:
            return [] # Nichts zu optimieren

        # Entscheidungsvariablen: x[i, j] = Menge von supplier i zu needy j
        num_vars = num_suppliers * num_needy
        var_indices: Dict[Tuple[Union[int, str], Union[int, str]], int] = {}
        idx = 0
        for i, s_id in enumerate(supplier_ids):
            for j, n_id in enumerate(needy_ids):
                var_indices[(s_id, n_id)] = idx
                idx += 1

        # Zielfunktion: Minimiere (negative Priorität * Erfüllung) + Transportkosten
        # Zugriff auf Transportkosten über s3_manager.model.get_transport_cost benötigt
        transport_weight = self.config.get("transport_cost_weight_interregional", 0.1) # Gewichtung der Transportkosten
        def objective(x: np.ndarray) -> float:
            cost = 0.0
            fulfillment = defaultdict(float) # Erfüllung pro needy region

            for i, s_id in enumerate(supplier_ids):
                for j, n_id in enumerate(needy_ids):
                    idx = var_indices[(s_id, n_id)]
                    amount = x[idx]
                    fulfillment[n_id] += amount

                    # Transportkosten (optional, erfordert Methode im Model)
                    if transport_weight > 0 and hasattr(self.s3_manager.model, 'get_transport_cost'):
                        try:
                            transport_cost_per_unit = self.s3_manager.model.get_transport_cost(s_id, n_id, resource)
                            cost += amount * transport_cost_per_unit * transport_weight
                        except Exception as tc_error:
                             self.logger.warning(f"Fehler bei Abruf Transportkosten ({s_id}->{n_id}, {resource}): {tc_error}")
                             # Ohne Transportkosten weiterrechnen? Oder abbrechen? Hier: weiter ohne.
                             pass

            # Nutzen durch Erfüllung (gewichtet mit Priorität)
            utility = 0.0
            for j, n_id in enumerate(needy_ids):
                 # Log-Nutzen für konkave Funktion (abnehmender Grenznutzen der Erfüllung)
                 # Sichert auch, dass Erfüllung > 0 bevorzugt wird.
                 ratio = min(fulfillment[n_id] / max(1e-9, needy[n_id]), 1.0) # Verhältnis der Erfüllung (max 1.0)
                 utility += priorities.get(n_id, 1.0) * math.log(1.0 + 9.0 * ratio) # Skaliert auf log(1) bis log(10)

            return cost - utility # Minimiere Kosten - Nutzen (d.h. maximiere Nutzen - Kosten)

        # Constraints
        constraints = []
        # 1. Supplier können nicht mehr liefern als verfügbar (Überschuss)
        for i, s_id in enumerate(supplier_ids):
            cons = {'type': 'ineq', 'fun': lambda x, i=i, s_id=s_id: suppliers[s_id] - sum(x[var_indices[(s_id, n_id)]] for j, n_id in enumerate(needy_ids))}
            constraints.append(cons)

        # 2. Needy Regionen erhalten maximal ihren Bedarf
        for j, n_id in enumerate(needy_ids):
            cons = {'type': 'ineq', 'fun': lambda x, j=j, n_id=n_id: needy[n_id] - sum(x[var_indices[(s_id, n_id)]] for i, s_id in enumerate(supplier_ids))}
            constraints.append(cons)

        # Bounds für Variablen (>= 0)
        bounds = [(0, None)] * num_vars

        # Initial guess (z.B. proportional oder null)
        x0 = np.zeros(num_vars)
        # Hier könnte man den proportionalen Fallback als Startwert berechnen
        # initial_prop = self._proportional_transfers(...)
        # ... x0 füllen ...

        # --- Optimierung ---
        try:
            result: OptimizeResult = system3.minimize(
                objective, x0, method='SLSQP', bounds=bounds, constraints=constraints
            )
        except Exception as opt_error:
             self.logger.error(f"Fehler während Scipy minimize für Ressource {resource}: {opt_error}", exc_info=True)
             # Fallback bei Optimierungsfehler
             return self._proportional_transfers(resource, needy, suppliers, priorities, total_transfer)

        # --- Ergebnis auswerten ---
        transfers = []
        if result.success:
            self.logger.info(f"Inter-regionale Optimierung für {resource} erfolgreich. Zielfunktionswert: {-result.fun:.3f}") # -fun weil wir minimiert haben
            x_opt = result.x
            for i, s_id in enumerate(supplier_ids):
                for j, n_id in enumerate(needy_ids):
                    idx = var_indices[(s_id, n_id)]
                    amount = x_opt[idx]
                    if amount > 1e-6: # Nur signifikante Transfers berücksichtigen
                        transfers.append((s_id, n_id, amount))
        else:
             self.logger.warning(f"Inter-regionale Optimierung für {resource} fehlgeschlagen: {result.message}. Nutze proportionalen Fallback.")
             # Expliziter Aufruf des Fallbacks
             transfers = self._proportional_transfers(resource, needy, suppliers, priorities, total_transfer)

        return transfers


    def _proportional_transfers(self,
        resource: str,
        needy: Dict[Union[int, str], float],
        suppliers: Dict[Union[int, str], float],
        priorities: Dict[Union[int, str], float],
        total_transfer: float
    ) -> List[Tuple[Union[int, str], Union[int, str], float]]:
        """
        Berechnet proportionale Transfers basierend auf Bedarf und Priorität.
        Angepasste Version von S3M._proportional_inter_regional_transfers.
        """
        transfers = []
        remaining_suppliers = dict(suppliers) # Kopie, da wir den Bestand reduzieren
        remaining_need = dict(needy) # Kopie des Bedarfs

        if total_transfer <= 1e-6 or not needy or not suppliers:
             return []

        # Sortiere Needy nach Priorität * Bedarf (höchster Bedarf/Prio zuerst)
        needy_sorted_ids = sorted(
            needy.keys(),
            key=lambda n_id: priorities.get(n_id, 1.0) * needy[n_id],
            reverse=True
        )

        # Iteriere durch die bedürftigen Regionen
        for needy_id in needy_sorted_ids:
            still_needed = remaining_need[needy_id]
            if still_needed <= 1e-6: continue # Bedarf schon gedeckt

            # Iteriere durch verfügbare Supplier (optional sortiert, z.B. nach niedrigster Prio oder größtem Überschuss?)
            # Hier einfache Iteration durch Dictionary-Reihenfolge (kann verbessert werden)
            supplier_ids_shuffled = list(remaining_suppliers.keys())
            random.shuffle(supplier_ids_shuffled) # Fügt etwas Zufall hinzu

            for supplier_id in supplier_ids_shuffled:
                available = remaining_suppliers[supplier_id]
                if available <= 1e-6: continue

                # Bestimme Transfermenge
                transfer = min(still_needed, available)
                transfers.append((supplier_id, needy_id, transfer))

                # Aktualisiere verbleibende Mengen
                remaining_suppliers[supplier_id] -= transfer
                still_needed -= transfer
                remaining_need[needy_id] -= transfer # Aktualisiere auch Bedarf (wichtig?)

                if remaining_suppliers[supplier_id] <= 1e-6:
                     # Entferne leere Supplier nicht direkt während Iteration,
                     # setze nur auf 0 oder markiere sie
                     pass

                if still_needed <= 1e-6:
                     break # Bedarf dieser Region gedeckt, gehe zur nächsten bedürftigen Region

        # Bereinige Transfers von sehr kleinen Mengen (numerische Artefakte)
        final_transfers = [(f, t, a) for f, t, a in transfers if a > 1e-6]

        # Optional: Prüfe, ob Gesamttransfer `total_transfer` entspricht (sollte ungefähr passen)
        allocated_sum = sum(a for _,_,a in final_transfers)
        if abs(allocated_sum - total_transfer) > 1e-3 and total_transfer > 0:
             self.logger.warning(f"Proportionale Allokation für {resource}: Summe ({allocated_sum:.2f}) weicht leicht von Ziel ({total_transfer:.2f}) ab.")
             # Ggf. Skalierung der Transfers auf `total_transfer`?

        return final_transfers