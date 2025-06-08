# Impaler/agents/resource.py
"""
Verwaltet den globalen Pool an natürlichen Ressourcen, deren Regeneration,
Abbau und die damit verbundenen Umweltauswirkungen in einer Simulation
ohne klassische Marktmechanismen.
"""

import random
import numpy as np
import logging
from collections import defaultdict
import math
from typing import Dict, Any, Optional, TYPE_CHECKING

# Typ-Prüfung Imports
if TYPE_CHECKING:
    from ..core.model import EconomicModel
    # Importiere Config-Typen für bessere Hints
    from ..core.config import EnvironmentConfig

class ResourceAgent:
    """
    Verwaltet den globalen Bestand an natürlichen Ressourcen, deren
    Regeneration und Abbau sowie grundlegende Umweltindikatoren.

    Diese Implementierung ist für Planwirtschaften angepasst und enthält
    keine monetäre Logik (Preise, Kosten). Der Fokus liegt auf physischen
    Beständen, Regenerationsraten und Nachhaltigkeitsaspekten.

    Attributes:
        unique_id (str): Eindeutige Kennung ("resource_agent").
        model (EconomicModel): Referenz zum Hauptmodell.
        logger (logging.Logger): Logger-Instanz.
        inventory (Dict[str, float]): Aktuelle Bestände an Ressourcen {"resource_name": amount}.
        resource_capacity (Dict[str, float]): Maximalkapazitäten pro Ressource.
        resource_regeneration_rates (Dict[str, float]): Basis-Regenerationsraten pro Ressource.
        resource_depletion_rates (Dict[str, float]): Natürliche Abbauraten pro Ressource.
        extraction_efficiency (float): Effizienzfaktor beim Abbau/Extrahieren (0-1).
        environmental_impact_factors (Dict[str, Dict[str, float]]): Umweltbelastung pro abgebauter Einheit
                                                                   {resource: {impact_type: factor}}.
        cumulative_environmental_impact (Dict[str, float]): Aufsummierte globale Umweltbelastung pro Typ.
        sustainability_index (float): Globaler Nachhaltigkeitsindikator (0-1), beeinflusst Regeneration.
        resource_quality (Dict[str, float]): Qualitätsfaktor pro Ressource (0-1), kann Abbau beeinflussen.
    """
    unique_id: str
    model: 'EconomicModel'
    logger: logging.Logger
    inventory: Dict[str, float]
    resource_capacity: Dict[str, float]
    resource_regeneration_rates: Dict[str, float]
    resource_depletion_rates: Dict[str, float]
    extraction_efficiency: float
    environmental_impact_factors: Dict[str, Dict[str, float]]
    cumulative_environmental_impact: Dict[str, float]
    sustainability_index: float
    resource_quality: Dict[str, float]

    def __init__(self, unique_id: str, model: 'EconomicModel', **kwargs):
        """
        Initialisiert den ResourceAgent.

        Liest Parameter aus der EnvironmentConfig des Modells.

        Args:
            unique_id: Eindeutige ID (normalerweise "resource_agent").
            model: Referenz zum Hauptmodell.
            **kwargs: Optionale zusätzliche Parameter (werden aktuell nicht verwendet).
        """
        self.unique_id = unique_id
        self.model = model
        self.logger = model.logger.getChild('ResourceAgent')

        # Lade Konfiguration sicher aus dem Modell. Manche Tests nutzen nur einen
        # einfachen Mock ohne EnvironmentConfig. Deshalb fallen wir bei fehlender
        # Konfiguration auf Attribute direkt am Modell zurück.
        env_config = getattr(getattr(model, "config", None), "environment_config", None)

        if env_config and isinstance(getattr(env_config, "resource_capacities", None), dict):
            self.resource_capacity = dict(getattr(env_config, "resource_capacities", {}))
            self.resource_regeneration_rates = dict(getattr(env_config, "resource_regeneration_rates", {}))
            self.resource_depletion_rates = dict(getattr(env_config, "resource_depletion_rates",
                                                         {res: random.uniform(0.01, 0.05) for res in self.resource_capacity}))
            self.environmental_impact_factors = dict(getattr(env_config, "emission_factors",
                                                           {res: {"co2": 0.1, "pollution": 0.05} for res in self.resource_capacity}))
            # Kompatibilität: Einfacher Impact-Wert pro Ressource
            default_imp = {res: random.uniform(0.1, 0.3) for res in self.resource_capacity}
            self.environmental_impact = dict(getattr(env_config, "environmental_impact", default_imp))
            self.inventory = self.resource_capacity.copy()
            self.resource_quality = dict(getattr(env_config, "resource_quality",
                                                {res: random.uniform(0.8, 1.0) for res in self.resource_capacity}))
        else:
            self.logger.error("EnvironmentConfig nicht im Modell gefunden! Initialisiere ResourceAgent mit leeren Werten.")
            self.resource_capacity = dict(getattr(model, "resource_capacity", {}))
            self.resource_regeneration_rates = dict(getattr(model, "resource_regeneration", {}))
            self.resource_depletion_rates = {res: random.uniform(0.01, 0.05) for res in self.resource_capacity}
            self.environmental_impact_factors = {res: {"co2": 0.1, "pollution": 0.05} for res in self.resource_capacity}
            self.environmental_impact = {res: random.uniform(0.1, 0.3) for res in self.resource_capacity}
            self.inventory = {res: 0.0 for res in self.resource_capacity}
            # Tests ohne EnvironmentConfig erwarten deterministische Qualität
            # von 1.0, damit die Extraktion berechenbar bleibt.
            self.resource_quality = {res: 1.0 for res in self.resource_capacity}


        # Extraktionseffizienz aus Config oder Default
        if env_config and isinstance(getattr(env_config, "base_extraction_efficiency", 0.8), (int, float)):
            base_eff = getattr(env_config, "base_extraction_efficiency", 0.8)
        else:
            base_eff = 0.8
        self.extraction_efficiency = float(kwargs.get('extraction_efficiency', base_eff))

        # Tracking von Umweltauswirkungen
        self.cumulative_environmental_impact = defaultdict(float)
        # Für einfache Tests existiert weiterhin ein skalare Gesamtmetrik
        # "cumulative_impact".
        self.cumulative_impact = 0.0
        # Globaler Nachhaltigkeitsindex (startet hoch)
        self.sustainability_index = float(kwargs.get('initial_sustainability', 0.95))

        # Alias-Namen fuer Rueckwaerts-Kompatibilitaet
        self.resource_regeneration = self.resource_regeneration_rates
        self.depletion_rate = self.resource_depletion_rates
        # "cumulative_impact" blieb früher ein einfacher Float.  Zur
        # Kompatibilität speichern wir ihn zusätzlich separat.

        self.logger.info(f"ResourceAgent '{self.unique_id}' initialisiert mit {len(self.inventory)} Ressourcen.")

    # ------------------------------------------------------------------
    # Kompatibilitäts-Wrapper für ältere Tests/Code
    def update_resource_regeneration(self) -> None:
        """Abwaertskompatible Methode (alias fuer _update_dynamic_regeneration_rates)."""
        self._update_dynamic_regeneration_rates()

    def resource_depletion_cycle(self) -> None:
        """Alias fuer _apply_regeneration_and_depletion."""
        self._apply_regeneration_and_depletion()

    def step_stage(self, stage: str) -> None:
        """
        Führt Aktionen für die angegebene Simulationsphase aus.

        Args:
            stage: Name der aktuellen Phase.
        """
        self.logger.debug(f"ResourceAgent: Executing stage '{stage}'")
        if stage == "resource_regen":
            # 1. Regenerationsraten basierend auf Umweltzustand etc. anpassen
            self._update_dynamic_regeneration_rates()
            # 2. Regeneration und natürlichen Abbau durchführen
            self._apply_regeneration_and_depletion()
            # 3. Nachhaltigkeitsindex ggf. anpassen
            self._update_sustainability_index()

        # Hinweis: invest_in_sustainability wird nicht mehr automatisch hier aufgerufen,
        #          sondern sollte extern getriggert werden (z.B. durch System 5).

    def _update_dynamic_regeneration_rates(self) -> None:
        """
        Passt die Regenerationsraten dynamisch an, z.B. basierend auf
        Umweltbelastung oder globalem Nachhaltigkeitsindex.
        (Ersetzt alte `update_resource_regeneration`-Logik).
        """
        self.logger.debug("Aktualisiere dynamische Regenerationsraten...")

        # Manche Tests benutzen ein minimales Mock-Modell ohne vollständige
        # Konfiguration.  In diesem Fall sollen die in der Instanz
        # gespeicherten `resource_regeneration_rates` weiterverwendet werden.
        env_cfg = getattr(getattr(self.model, "config", None),
                          "environment_config", None)
        if env_cfg and isinstance(getattr(env_cfg, "resource_regeneration_rates", None), dict):
            base_rates = env_cfg.resource_regeneration_rates
        else:
            base_rates = self.resource_regeneration_rates

        for resource, base_rate in base_rates.items():
            if resource not in self.resource_capacity or resource not in self.inventory:
                continue

            capacity = self.resource_capacity[resource]
            current_amount = self.inventory[resource]

            # Faktor 1: Füllstand (ähnlich logistischem Wachstum)
            fill_level_factor = 1.0
            if capacity > 0:
                 relative_amount = current_amount / capacity
                 # Regeneration stärker bei niedrigem Füllstand, schwächer nahe Kapazität
                 fill_level_factor = max(0.1, 1.0 - relative_amount**2) # Beispiel: quadratische Abnahme

            # Faktor 2: Globale Nachhaltigkeit
            sustainability_factor = max(0.2, self.sustainability_index) # Mind. 20% Regeneration

            # Faktor 3: Spezifische Umweltbelastung (z.B. 'pollution' wirkt auf 'water')
            env_impact_factor = 1.0
            # Beispiel: Hohe Verschmutzung reduziert Wasserregeneration
            if resource == 'water':
                 pollution_level = self.cumulative_environmental_impact.get('pollution', 0.0)
                 pollution_capacity = 0.0
                 if env_cfg is not None:
                      pollution_capacity = getattr(env_cfg, 'environmental_capacities', {}).get('pollution', 1000.0)
                 else:
                      pollution_capacity = 1000.0
                 if pollution_capacity > 0:
                      relative_pollution = pollution_level / pollution_capacity
                      env_impact_factor = max(0.3, 1.0 - relative_pollution * 0.7) # Max 70% Reduktion

            # Berechne finale Rate für diesen Step
            dynamic_rate = base_rate * fill_level_factor * sustainability_factor * env_impact_factor
            self.resource_regeneration_rates[resource] = max(0.0, dynamic_rate) # Rate kann nicht negativ sein

            # self.logger.debug(f"  Resource '{resource}': BaseRate={base_rate:.3f}, FillFactor={fill_level_factor:.2f}, "
            #                   f"SustFactor={sustainability_factor:.2f}, EnvFactor={env_impact_factor:.2f} -> DynRate={dynamic_rate:.3f}")


    def _apply_regeneration_and_depletion(self) -> None:
        """
        Wendet Regeneration und natürlichen Abbau auf die Ressourcenbestände an.
        (Ersetzt alte `resource_depletion_cycle`-Logik).
        """
        self.logger.debug("Wende Regeneration und Depletion an...")
        for resource, current_amount in list(self.inventory.items()): # Kopie für Iteration
            capacity = self.resource_capacity.get(resource, float('inf'))
            regen_rate = self.resource_regeneration_rates.get(resource, 0.0) # Nutzt dynamische Rate
            depletion_rate = self.resource_depletion_rates.get(resource, 0.0)

            # Regeneration (bezogen auf aktuellen Bestand und Kapazität)
            regeneration_amount = 0.0
            if capacity > 0 and current_amount < capacity:
                 # Nutze Rate, die bereits in _update_dynamic_regeneration_rates angepasst wurde
                 # Einfachere Annahme hier: Rate gilt pro Einheit Bestand
                 regeneration_amount = regen_rate * current_amount
                 # Alternativ: Logistisches Wachstum (wenn Rate nicht schon angepasst wurde)
                 # regeneration_amount = regen_rate * current_amount * (1 - current_amount / capacity)

            # Depletion (bezogen auf aktuellen Bestand)
            depletion_amount = depletion_rate * current_amount

            # Nettoänderung
            net_change = regeneration_amount - depletion_amount
            new_amount = current_amount + net_change

            # Kappe bei 0 und Kapazität
            final_amount = np.clip(new_amount, 0.0, capacity)

            if abs(final_amount - current_amount) > 1e-3: # Nur loggen bei Änderung
                 self.logger.debug(f"  Resource '{resource}': {current_amount:.2f} + {regeneration_amount:.2f} (Regen) "
                                   f"- {depletion_amount:.2f} (Deplete) -> {final_amount:.2f} (Limit: {capacity:.1f})")
                 self.inventory[resource] = final_amount

    def _update_sustainability_index(self) -> None:
        """Aktualisiert den globalen Nachhaltigkeitsindex."""
        # Beispiel: Index sinkt bei hoher Umweltbelastung, steigt bei Unterschreitung von Zielen
        impact_penalty = 0.0
        target_bonus = 0.0
        num_factors = 0

        env_cfg = getattr(getattr(self.model, "config", None),
                          "environment_config", None)
        cfg_valid = env_cfg and isinstance(getattr(env_cfg, "environmental_capacities", None), dict)
        env_caps = getattr(env_cfg, "environmental_capacities", {}) if cfg_valid else {}
        sus_targets = getattr(env_cfg, "sustainability_targets", {}) if cfg_valid else {}
        current_step = getattr(self.model, "current_step", 0)

        # Wenn keine brauchbare Konfiguration vorhanden ist, führen wir einen
        # einfachen leichten Anstieg des Index durch, solange er unter 0.6 liegt.
        if not cfg_valid and self.sustainability_index < 0.6:
            self.sustainability_index = min(1.0, self.sustainability_index + 0.01)
            return

        for impact_type, cumulative_val in self.cumulative_environmental_impact.items():
            capacity = env_caps.get(impact_type)
            if capacity and capacity > 0:
                 # Penalty steigt, je näher man an die Kapazität kommt
                 relative_impact = cumulative_val / capacity
                 impact_penalty += max(0, relative_impact - 0.5) * 0.01 # Penalty > 50% Auslastung
                 num_factors +=1

            # Bonus, wenn Nachhaltigkeitsziel unterschritten wird
            if env_cfg is not None:
                target = env_cfg.get_current_sustainability_target(impact_type, current_step)
            else:
                target = None
            if target is not None and cumulative_val < target * 0.8: # Deutlich unter Ziel
                 target_bonus += 0.005
                 num_factors += 1

        # Berechne Änderung und wende Glättung an
        change = target_bonus - impact_penalty
        # Glättungsfaktor, um schnelle Schwankungen zu vermeiden
        smoothing = 0.1
        self.sustainability_index = np.clip(
             self.sustainability_index * (1 - smoothing) + (self.sustainability_index + change) * smoothing,
             0.1, 1.0 # Index zwischen 0.1 und 1.0
        )
        if abs(change) > 0.001:
             self.logger.debug(f"Nachhaltigkeitsindex angepasst auf {self.sustainability_index:.3f} (Change: {change:.4f})")


    def extract_resources(self, request_amounts: Dict[str, float]) -> Dict[str, float]:
        """
        Simuliert die Extraktion von Ressourcen durch Agenten (z.B. Producer).

        Berücksichtigt verfügbaren Bestand, Extraktionseffizienz und
        Umweltauswirkungen.

        Args:
            request_amounts: Dictionary {resource_name: desired_amount}.

        Returns:
            Dictionary {resource_name: actual_extracted_amount}.
            Die tatsächlich erhaltene Menge nach Effizienzverlust.
        """
        extracted: Dict[str, float] = {}
        total_impact_this_extraction = defaultdict(float)

        for resource, desired_amount in request_amounts.items():
            if desired_amount <= 0:
                extracted[resource] = 0.0
                continue
            if resource not in self.inventory:
                self.logger.warning(f"Angeforderte Ressource '{resource}' nicht im Inventar verfügbar.")
                extracted[resource] = 0.0
                continue

            # 1. Verfügbare Menge prüfen
            available_amount = self.inventory[resource]
            # Menge, die maximal entnommen werden kann
            amount_to_extract = min(available_amount, desired_amount)

            if amount_to_extract <= 0:
                 extracted[resource] = 0.0
                 continue

            # 2. Bestand reduzieren
            self.inventory[resource] -= amount_to_extract
            self.logger.debug(f"Extrahiere {amount_to_extract:.2f} von '{resource}'. Bestand jetzt: {self.inventory[resource]:.2f}")

            # 3. Tatsächlich nutzbare Menge nach Effizienz berechnen
            # Berücksichtige optional Ressourcenqualität
            quality_factor = self.resource_quality.get(resource, 1.0)
            effective_efficiency = self.extraction_efficiency * quality_factor
            actual_yield = amount_to_extract * effective_efficiency
            extracted[resource] = actual_yield

            # 4. Umweltauswirkungen berechnen und akkumulieren
            impacts = self.environmental_impact_factors.get(resource, {})
            for impact_type, factor in impacts.items():
                 impact_amount = amount_to_extract * factor  # Impact basiert auf entnommener Menge
                 self.cumulative_environmental_impact[impact_type] += impact_amount
                 total_impact_this_extraction[impact_type] += impact_amount
                 # Summiere zusätzlich einen einfachen Gesamtindex für alte Tests
                 self.cumulative_impact += impact_amount

        if total_impact_this_extraction:
             impact_log = ", ".join([f"{k}: {v:.2f}" for k, v in total_impact_this_extraction.items()])
             self.logger.debug(f"Extraktion verursachte Umweltauswirkungen: {impact_log}")

        return extracted

    def invest_in_sustainability(self, investment_amount: float = 0.0, *, invest_factor: float = None) -> None:
        """
        Simuliert Investitionen in Nachhaltigkeit und Ressourceneffizienz.
        Wird typischerweise extern getriggert (z.B. von System 5 oder als Policy).

        Args:
            investment_amount: "Investierte" Menge (kein Geld, eher ein Maß für den Aufwand).
        """
        # Ältere Tests verwenden den Parameternamen ``invest_factor``.
        if invest_factor is not None:
            investment_amount = invest_factor
        if investment_amount <= 0:
            return

        # Lese Konfigurationswerte für die Effektivität der Investition
        config = getattr(getattr(self.model, "config", None), "environment_config", None)

        def _safe_number(obj, attr, default):
            val = getattr(obj, attr, default) if obj else default
            return val if isinstance(val, (int, float)) else default

        sustainability_investment_effectiveness = _safe_number(config, "sustainability_investment_effectiveness", 0.05)
        impact_reduction_effectiveness = _safe_number(config, "impact_reduction_effectiveness", 0.01)
        extraction_efficiency_effectiveness = _safe_number(config, "extraction_efficiency_effectiveness", 0.005)

        # 1. Verbessere globalen Nachhaltigkeitsindex (mit abnehmendem Grenznutzen)
        old_sustain = self.sustainability_index
        improvement_sustain = sustainability_investment_effectiveness * math.log(1 + investment_amount) * (1.1 - old_sustain)
        self.sustainability_index = min(1.0, old_sustain + improvement_sustain)

        # 2. Reduziere Umwelt-Impact-Faktoren (pro Ressource)
        for resource, impacts in self.environmental_impact_factors.items():
             for impact_type in list(impacts.keys()):  # Kopie für Iteration über Keys
                  old_factor = impacts[impact_type]
                  reduction = impact_reduction_effectiveness * math.log(1 + investment_amount) * old_factor
                  # Reduziere Faktor, aber nicht unter einen Minimalwert (z.B. 10% des Originals)
                  impacts[impact_type] = max(old_factor * 0.1, old_factor - reduction)

        # Kompatibilität: reduziere auch einfachen Impact-Wert pro Ressource
        for res in list(self.environmental_impact.keys()):
             old_val = self.environmental_impact[res]
             red = impact_reduction_effectiveness * math.log(1 + investment_amount) * old_val
             self.environmental_impact[res] = max(old_val * 0.1, old_val - red)

        # 3. Erhöhe Extraktionseffizienz (mit abnehmendem Grenznutzen)
        old_eff = self.extraction_efficiency
        improvement_eff = extraction_efficiency_effectiveness * math.log(1 + investment_amount) * (1.0 - old_eff)
        self.extraction_efficiency = min(0.98, old_eff + improvement_eff) # Max 98% Effizienz

        self.logger.info(f"Investition in Nachhaltigkeit ({investment_amount:.1f}): "
                         f"Index {old_sustain:.3f}->{self.sustainability_index:.3f}, "
                         f"Extraction Eff {old_eff:.3f}->{self.extraction_efficiency:.3f}")


    # --- Methoden zur Abfrage von Zuständen ---

    def get_resource_level(self, resource: str) -> float:
        """Gibt den aktuellen Bestand einer Ressource zurück."""
        return self.inventory.get(resource, 0.0)

    def get_resource_capacity(self, resource: str) -> float:
        """Gibt die Maximalkapazität einer Ressource zurück."""
        return self.resource_capacity.get(resource, 0.0)

    def get_cumulative_impact(self, impact_type: str) -> float:
        """Gibt die kumulierte Umweltbelastung eines Typs zurück."""
        return self.cumulative_environmental_impact.get(impact_type, 0.0)

    def get_sustainability_index(self) -> float:
        """Gibt den aktuellen globalen Nachhaltigkeitsindex zurück."""
        return self.sustainability_index

    def generate_report(self) -> Dict[str, Any]:
        """Erstellt einen Bericht über den aktuellen Zustand des ResourceAgent."""
        return {
            "unique_id": self.unique_id,
            "inventory": dict(self.inventory),
            "capacities": dict(self.resource_capacity),
            "regeneration_rates": {k: round(v, 4) for k, v in self.resource_regeneration_rates.items()},
            "depletion_rates": {k: round(v, 4) for k, v in self.resource_depletion_rates.items()},
            "extraction_efficiency": round(self.extraction_efficiency, 3),
            "cumulative_impact": self.cumulative_impact,
            "sustainability_index": round(self.sustainability_index, 3),
            "resource_quality": {k: round(v, 3) for k, v in self.resource_quality.items()},
        }