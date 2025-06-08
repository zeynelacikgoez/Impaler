# Impaler/utils/metrics.py
"""
Dienstprogrammfunktionen zur Berechnung verschiedener ökonomischer Metriken.

Dieses Modul stellt Funktionen zur Berechnung von Indikatoren bereit, die in der
Simulation zur Messung der Systemleistung, Ungleichheit, Effizienz und anderer
ökonomischer oder sozialer Aspekte verwendet werden können. Einige Metriken sind
eher für Marktwirtschaften relevant, können aber zu Vergleichszwecken oder für
Hybridmodelle nützlich sein.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set, TYPE_CHECKING

# Importiere mathematische Hilfsfunktionen (Pfad ggf. anpassen)
try:
    from .math_utils import gini_coefficient, theil_index, atkinson_index
except ImportError:
    # Fallback, falls math_utils nicht im selben Verzeichnis ist
    # Dies sollte durch korrekte Paketstruktur vermieden werden
    logging.warning("math_utils nicht gefunden, Ungleichheitsmetriken sind nicht verfügbar.")
    # Definiere Dummy-Funktionen, damit der Rest des Codes nicht bricht
    def gini_coefficient(values: List[float]) -> float: return 0.0
    def theil_index(values: List[float]) -> float: return 0.0
    def atkinson_index(values: List[float], epsilon: float = 0.5) -> float: return 0.0


# Logger für dieses Modul
logger = logging.getLogger(__name__)

# Konstante für numerische Stabilität
EPSILON = 1e-9

def calculate_price_index(current_prices: Dict[str, float],
                         base_prices: Dict[str, float],
                         weights: Optional[Dict[str, float]] = None) -> float:
    """
    Berechnet einen Preisindex (Inflationsmaß) mittels gewichteter Preise.

    Hinweis: In einer reinen Planwirtschaft ohne Preismechanismen ist diese Metrik
             weniger relevant, könnte aber für externe Güter oder Vergleichszwecke dienen.

    Args:
        current_prices: Dictionary mit aktuellen Preisen pro Gut {gut: preis}.
        base_prices: Dictionary mit Basis-/Referenzpreisen pro Gut {gut: preis}.
        weights: Optionales Dictionary mit Gewichtungen pro Gut {gut: gewicht}.
                 Wenn None, werden gleiche Gewichte verwendet.

    Returns:
        Preisindexwert (typischerweise >= 0). Ein Wert > 1.0 deutet auf Inflation hin.
        Gibt 1.0 zurück bei leeren Inputs oder Fehlern.
    """
    if not current_prices or not base_prices:
        logger.debug("Berechnung Preisindex: Leere Preislisten erhalten.")
        return 1.0

    # Verwende nur Güter, die in beiden Preislisten vorkommen
    common_goods = set(current_prices.keys()).intersection(base_prices.keys())
    if not common_goods:
        logger.debug("Berechnung Preisindex: Keine gemeinsamen Güter in Preislisten.")
        return 1.0

    # Gewichte bestimmen
    if weights is None:
        num_common = len(common_goods)
        weights = {good: 1.0 / num_common for good in common_goods}
        logger.debug(f"Berechnung Preisindex: Verwende gleiche Gewichte für {num_common} Güter.")
    else:
        # Normalisiere Gewichte, falls sie nicht 1 ergeben
        total_weight = sum(weights.get(good, 0) for good in common_goods)
        if total_weight <= EPSILON:
            logger.warning("Berechnung Preisindex: Gesamtgewichtung ist Null oder negativ.")
            return 1.0
        weights = {good: weights.get(good, 0) / total_weight for good in common_goods}

    # Berechne gewichteten Index
    index_sum = 0.0
    valid_weight_sum = 0.0 # Summe der tatsächlich verwendeten Gewichte

    for good in common_goods:
        current_p = current_prices.get(good)
        base_p = base_prices.get(good)
        weight = weights.get(good, 0)

        # Stelle sicher, dass Preise und Gewichtung gültig sind
        if current_p is not None and base_p is not None and base_p > EPSILON and weight > 0:
            try:
                price_ratio = current_p / base_p
                index_sum += weight * price_ratio
                valid_weight_sum += weight
            except ZeroDivisionError:
                 logger.warning(f"Berechnung Preisindex: Division durch Null bei Gut '{good}' (Basispreis={base_p}).")
                 continue # Überspringe dieses Gut
        elif weight > 0:
             logger.debug(f"Berechnung Preisindex: Überspringe Gut '{good}' wegen fehlender Preise oder Basispreis Null.")


    # Endgültigen Index berechnen
    if valid_weight_sum <= EPSILON:
        logger.warning("Berechnung Preisindex: Keine gültigen gewichteten Preise gefunden.")
        return 1.0

    # Wenn einige Güter aufgrund fehlender Daten übersprungen wurden,
    # ist die Summe der tatsächlich verwendeten Gewichte < 1. In diesem Fall
    # muss index_sum erneut normalisiert werden, um einen korrekten Index zu erhalten.
    price_index = index_sum / valid_weight_sum

    logger.debug(f"Berechneter Preisindex: {price_index:.4f}")
    return max(0.0, price_index) # Index sollte nicht negativ sein


def calculate_gdp(production_quantities: Dict[str, float],
                  prices: Dict[str, float],
                  intermediate_consumption: Optional[Dict[str, float]] = None,
                  government_spending: float = 0.0,
                  net_exports: float = 0.0) -> float:
    """
    Berechnet das Bruttoinlandsprodukt (BIP) - verschiedene Ansätze möglich.
    Hier: Produktionswert-Ansatz (Output - Vorleistungen).

    Hinweis: In einer Planwirtschaft ohne Marktpreise ist das BIP als Konzept
             schwierig anzuwenden. Man könnte stattdessen den Gesamtwert der
             produzierten Güter anhand von Planungs-"Bewertungen" oder
             Arbeitswert-Äquivalenten berechnen. Diese Funktion nutzt übergebene 'prices'.

    Args:
        production_quantities: Produzierte Mengen pro Gut {gut: menge}.
        prices: Preise oder Planwerte pro Gut {gut: preis/wert}.
        intermediate_consumption: Optional, Verbrauch von Gütern als Vorleistung {gut: menge}.
        government_spending: Ausgaben des "Staates" (falls modelliert).
        net_exports: Nettoexporte (falls modelliert).

    Returns:
        BIP-Wert oder äquivalenter Gesamtwert. Gibt 0.0 bei Fehlern zurück.
    """
    if not production_quantities or not prices:
        return 0.0

    # 1. Brutto-Produktionswert berechnen
    gross_production_value = 0.0
    for good, quantity in production_quantities.items():
        price = prices.get(good)
        if price is not None and quantity > 0:
            gross_production_value += quantity * price

    # 2. Wert der Vorleistungen abziehen
    intermediate_value = 0.0
    if intermediate_consumption:
        for good, quantity in intermediate_consumption.items():
            price = prices.get(good)
            if price is not None and quantity > 0:
                intermediate_value += quantity * price

    # BIP = Brutto Produktionswert - Vorleistungen (+ Gov Spending + Net Exports)
    # (Vereinfachte Version hier ohne Gov/Export)
    gdp = gross_production_value - intermediate_value
    # Falls andere Komponenten modelliert sind:
    # gdp += government_spending + net_exports

    logger.debug(f"BIP berechnet: {gdp:.2f} (ProdWert: {gross_production_value:.2f}, Vorleistung: {intermediate_value:.2f})")
    return max(0.0, gdp)


def calculate_unemployment_rate(employed_count: int, potential_workforce: int) -> float:
    """
    Berechnet die Arbeitslosenquote.

    Args:
        employed_count: Anzahl der beschäftigten Personen/Agenten.
        potential_workforce: Gesamtzahl der potenziell arbeitsfähigen Personen/Agenten.

    Returns:
        Arbeitslosenquote zwischen 0.0 und 1.0. Gibt 0.0 zurück, wenn workforce <= 0.
    """
    if potential_workforce <= 0:
        logger.debug("Berechnung Arbeitslosenquote: Keine potenzielle Workforce vorhanden.")
        return 0.0
    if employed_count < 0:
        logger.warning(f"Berechnung Arbeitslosenquote: Negativer employed_count ({employed_count}) erhalten.")
        employed_count = 0
    if employed_count > potential_workforce:
         logger.warning(f"Berechnung Arbeitslosenquote: Mehr Beschäftigte ({employed_count}) als Workforce ({potential_workforce}). Setze Rate auf 0.")
         return 0.0


    rate = 1.0 - (employed_count / potential_workforce)
    logger.debug(f"Arbeitslosenquote berechnet: {rate:.2%}")
    return rate


def calculate_distribution_metrics(values: List[float]) -> Dict[str, Optional[float]]:
    """
    Berechnet verschiedene Metriken zur Verteilung einer Größe (z.B. Output, Zufriedenheit).

    Nutzt Gini, Theil und Atkinson-Index (falls math_utils verfügbar).

    Args:
        values: Liste von Werten (z.B. Output pro Produzent, Zufriedenheit pro Konsument).
                Negative Werte werden ignoriert oder verursachen Fehler bei manchen Indizes.

    Returns:
        Dictionary mit Ungleichheitsmetriken {'gini': float, 'theil': float, 'atkinson_05': float}
        oder leeres Dict bei ungültigen Eingaben. Werte können None sein, wenn Berechnung fehlschlägt.
    """
    metrics: Dict[str, Optional[float]] = {
        "gini": None,
        "theil": None,
        "atkinson_05": None, # Atkinson mit epsilon=0.5
        "mean": None,
        "median": None,
        "std_dev": None,
        "min": None,
        "max": None
    }
    if not values:
        logger.debug("Berechnung Verteilungsmetriken: Leere Werteliste.")
        return metrics

    # Konvertiere zu numpy array und filtere ungültige Werte (NaN, Inf)
    try:
        valid_values = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    except Exception as e:
         logger.error(f"Fehler beim Konvertieren der Werte für Verteilungsmetriken: {e}")
         return metrics

    if len(valid_values) == 0:
        logger.debug("Berechnung Verteilungsmetriken: Keine gültigen Werte nach Filterung.")
        return metrics

    # Grundlegende Statistiken
    metrics["mean"] = float(np.mean(valid_values))
    metrics["median"] = float(np.median(valid_values))
    metrics["std_dev"] = float(np.std(valid_values))
    metrics["min"] = float(np.min(valid_values))
    metrics["max"] = float(np.max(valid_values))


    # Berechne Ungleichheitsindizes (nur für nicht-negative Werte)
    non_negative_values = valid_values[valid_values >= 0]
    if len(non_negative_values) > 0:
        try:
            metrics["gini"] = float(gini_coefficient(non_negative_values.tolist()))
        except Exception as e:
            logger.warning(f"Fehler bei Gini-Berechnung: {e}")
        try:
             # Theil erfordert positive Werte
             positive_values = non_negative_values[non_negative_values > EPSILON]
             if len(positive_values) > 0:
                  metrics["theil"] = float(theil_index(positive_values.tolist()))
        except Exception as e:
            logger.warning(f"Fehler bei Theil-Berechnung: {e}")
        try:
             # Atkinson erfordert positive Werte
             positive_values = non_negative_values[non_negative_values > EPSILON]
             if len(positive_values) > 0:
                  metrics["atkinson_05"] = float(atkinson_index(positive_values.tolist(), epsilon=0.5))
        except Exception as e:
            logger.warning(f"Fehler bei Atkinson-Berechnung: {e}")

    logger.debug(f"Verteilungsmetriken berechnet: Gini={metrics['gini']:.3f} (falls berechnet)")
    return metrics


def calculate_plan_fulfillment(planned: Dict[str, float], actual: Dict[str, float]) -> Dict[str, Any]:
    """
    Berechnet Metriken zur Planerfüllung.

    Args:
        planned: Geplante Mengen pro Gut/Ziel {key: planned_value}.
        actual: Tatsächlich erreichte Mengen pro Gut/Ziel {key: actual_value}.

    Returns:
        Dictionary mit Planerfüllungsmetriken:
        { 'overall_fulfillment': float (0-1+),
          'fulfillment_by_key': {key: ratio (0-1+)},
          'total_planned': float,
          'total_actual': float }
    """
    if not planned: # Kein Plan, keine Erfüllung messbar (oder 100% wenn auch actual leer?)
        return {
            'overall_fulfillment': 1.0 if not actual else 0.0,
            'fulfillment_by_key': {},
            'total_planned': 0.0,
            'total_actual': sum(actual.values())
        }

    fulfillment_by_key: Dict[str, float] = {}
    weighted_fulfillment_sum = 0.0
    total_planned_value = 0.0
    total_actual_value = sum(actual.values()) # Gesamter Ist-Wert

    for key, planned_value in planned.items():
        actual_value = actual.get(key, 0.0)
        total_planned_value += planned_value

        if planned_value > EPSILON:
            ratio = actual_value / planned_value
            fulfillment_by_key[key] = ratio
            # Gewichteter Durchschnitt basierend auf Planwert
            weighted_fulfillment_sum += ratio * planned_value
        elif actual_value > EPSILON:
             fulfillment_by_key[key] = float('inf') # Unendlich, wenn Plan 0 war, aber Ist > 0
             # Wie soll dies in die Gesamtmetrik einfließen? Hier ignoriert.
        else:
             fulfillment_by_key[key] = 1.0 # Plan 0, Ist 0 -> 100% erfüllt

    # Berechne Gesamt-Erfüllungsgrad (gewichtet nach Planwert)
    overall_fulfillment = weighted_fulfillment_sum / total_planned_value if total_planned_value > EPSILON else 1.0

    logger.debug(f"Planerfüllung berechnet: Overall={overall_fulfillment:.2%}")
    return {
        'overall_fulfillment': overall_fulfillment,
        'fulfillment_by_key': fulfillment_by_key,
        'total_planned': total_planned_value,
        'total_actual': total_actual_value
    }


# --- Wrapper-Funktionen für spezifische Anwendungsfälle ---

def calculate_producer_metrics(producers: List[Any]) -> Dict[str, Optional[float]]:
    """
    Berechnet aggregierte Metriken für alle aktiven Produzenten.

    Args:
        producers: Liste von ProducerAgent-Instanzen.

    Returns:
        Dictionary mit aggregierten Producer-Metriken.
    """
    active_producers = [p for p in producers if not getattr(p, "bankrupt", False)]
    if not active_producers:
        logger.debug("Berechnung Producer-Metriken: Keine aktiven Producer gefunden.")
        return {"producer_metrics_error": 0.0} # Signalisiert Fehler/keine Daten

    metrics = {}

    # Kapazitätsauslastung
    total_output = sum(getattr(p, 'total_output_this_step', 0.0) for p in active_producers)
    total_capacity = sum(getattr(p, 'productive_capacity', 0.0) for p in active_producers)
    metrics["avg_capacity_utilization"] = total_output / total_capacity if total_capacity > EPSILON else 0.0

    # Produktions-Ungleichheit (Gini des Outputs)
    outputs = [getattr(p, 'total_output_this_step', 0.0) for p in active_producers]
    metrics["output_gini"] = gini_coefficient(outputs)

    # Technologie-Level
    tech_levels = [getattr(p, 'tech_level', 1.0) for p in active_producers]
    metrics["avg_tech_level"] = float(np.mean(tech_levels))
    metrics["tech_level_gini"] = gini_coefficient(tech_levels)

    # Durchschnittliche Effizienz (könnte komplexer sein, gewichtet nach Output?)
    avg_efficiencies = [getattr(p, 'calculate_average_efficiency', lambda: 1.0)() for p in active_producers]
    metrics["avg_producer_efficiency"] = float(np.mean(avg_efficiencies))

    logger.debug(f"Producer-Metriken berechnet: AvgUtil={metrics['avg_capacity_utilization']:.1%}, OutputGini={metrics['output_gini']:.3f}")
    return metrics

def calculate_consumer_metrics(consumers: List[Any]) -> Dict[str, Optional[float]]:
    """
    Berechnet aggregierte Metriken für alle aktiven Konsumenten.

    Args:
        consumers: Liste von ConsumerAgent-Instanzen.

    Returns:
        Dictionary mit aggregierten Konsumenten-Metriken.
    """
    active_consumers = [c for c in consumers if not getattr(c, "insolvent", False)]
    if not active_consumers:
        logger.debug("Berechnung Consumer-Metriken: Keine aktiven Consumer gefunden.")
        return {"consumer_metrics_error": 0.0}

    metrics = {}

    # Zufriedenheit
    satisfactions = [getattr(c, 'satisfaction', 0.0) for c in active_consumers]
    metrics["avg_satisfaction"] = float(np.mean(satisfactions))
    metrics["satisfaction_gini"] = gini_coefficient(satisfactions)
    metrics["min_satisfaction"] = float(min(satisfactions))
    metrics["max_satisfaction"] = float(max(satisfactions))

    # Durchschnittlicher Erfüllungsgrad (über alle Güter und Konsumenten)
    all_fulfillments = []
    for c in active_consumers:
         all_fulfillments.extend(getattr(c, 'fulfillment_ratios', {}).values())
    metrics["avg_need_fulfillment"] = float(np.mean(all_fulfillments)) if all_fulfillments else 1.0

    logger.debug(f"Consumer-Metriken berechnet: AvgSat={metrics['avg_satisfaction']:.3f}, SatGini={metrics['satisfaction_gini']:.3f}")
    return metrics


# Hinweis: Funktionen wie calculate_market_efficiency, calculate_financial_system_metrics,
# calculate_price_volatility, calculate_labor_market_metrics (teilweise)
# wurden entfernt oder stark vereinfacht, da sie primär Markt- oder Finanzkonzepte
# abbilden, die im Kern von Impaler (Planwirtschaft) weniger relevant sind.
# calculate_resource_sustainability wurde ebenfalls entfernt, da dies eher
# im ResourceAgent oder EnvironmentConfig/Model selbst gehandhabt wird.