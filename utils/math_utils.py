# Impaler/utils/math_utils.py
"""
Mathematische Hilfsfunktionen für ökonomische Simulationen.

Dieses Modul stellt verschiedene mathematische Funktionen bereit, die in der
Simulation verwendet werden, z.B. für Ungleichheitsmetriken, Distanzberechnungen
und numerische Stabilität.
"""

import numpy as np
import math
import random
import logging
from typing import List, Union, Tuple, Optional

# Logger für dieses Modul
logger = logging.getLogger(__name__)

# Kleine Konstante für numerische Stabilität und Vergleiche
EPSILON = 1e-9

def sign_with_small_bias(value: float, threshold: float = EPSILON) -> int:
    """
    Gibt das Vorzeichen eines Wertes zurück, mit zufälligem Bias bei Werten nahe Null.

    Verhindert deterministisches Verhalten bei exaktem Nullwert.

    Args:
        value: Der zu prüfende Wert.
        threshold: Schwellwert, unterhalb dessen (absolut) randomisiert wird.

    Returns:
        1 oder -1.
    """
    if abs(value) < threshold:
        return random.choice([1, -1])
    elif value > 0:
        return 1
    else: # value < 0 (oder exakt 0, was durch threshold abgefangen wird)
        return -1

def gini_coefficient(values: Union[List[float], np.ndarray]) -> float:
    """
    Berechnet den Gini-Koeffizienten für eine Liste oder ein Array von Werten.

    Misst Ungleichheit, wobei 0 perfekte Gleichheit und 1 perfekte Ungleichheit darstellt.
    Ignoriert negative Werte und erfordert mindestens einen positiven Wert für eine sinnvolle Berechnung.

    Args:
        values: Liste oder NumPy-Array von numerischen Werten (z.B. Einkommen, Vermögen).

    Returns:
        Gini-Koeffizient zwischen 0.0 und 1.0. Gibt 0.0 zurück bei leeren oder nur Null-Inputs.
    """
    # Konvertiere zu NumPy Array und filtere nicht-endliche Werte
    try:
        vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    except Exception as e:
        logger.warning(f"Gini: Fehler bei Konvertierung zu Array: {e}. Gebe 0.0 zurück.")
        return 0.0

    # Filter nicht-negative Werte
    vals = vals[vals >= 0]

    n = len(vals)
    if n == 0:
        logger.debug("Gini: Leere oder nur ungültige Werte nach Filterung.")
        return 0.0

    # Sortiere Werte
    sorted_vals = np.sort(vals)
    # Berechne kumulative Summe
    cum_vals = np.cumsum(sorted_vals, dtype=float)
    total_sum = cum_vals[-1]

    # Wenn die Summe aller Werte nahe Null ist, herrscht (fast) perfekte Gleichheit
    if total_sum < EPSILON:
        logger.debug("Gini: Gesamtsumme der Werte ist nahe Null.")
        return 0.0

    # Gini-Formel (vereinfacht): 1 - 2 * (Fläche unter Lorenz-Kurve)
    # Fläche unter Lorenz = sum(cum_vals) / (total_sum * n) - Anpassung für diskrete Werte
    # Alternative Formel (aus Wikipedia / NumPy Beispielen):
    # Gini = (Summe aller |xi - xj|) / (2 * n * Summe aller xi)
    # Effizientere Berechnung:
    index = np.arange(1, n + 1) # 1-based index
    gini = (np.sum((2 * index - n - 1) * sorted_vals)) / (n * total_sum)

    # Stelle sicher, dass das Ergebnis im gültigen Bereich liegt
    return float(np.clip(gini, 0.0, 1.0))


def theil_index(values: Union[List[float], np.ndarray]) -> float:
    """
    Berechnet den Theil-Index (Theil's T) für eine Liste oder ein Array von Werten.

    Ein Entropie-basiertes Ungleichheitsmaß. Erfordert positive Werte.
    Höhere Werte bedeuten höhere Ungleichheit.

    Args:
        values: Liste oder NumPy-Array von numerischen Werten.

    Returns:
        Theil-Index (>= 0). Gibt 0.0 zurück bei leeren/ungültigen Inputs oder wenn alle Werte gleich sind.
    """
    try:
        vals = np.asarray([v for v in values if v is not None and np.isfinite(v) and v > EPSILON], dtype=float)
    except Exception as e:
        logger.warning(f"Theil: Fehler bei Konvertierung zu Array: {e}. Gebe 0.0 zurück.")
        return 0.0

    n = len(vals)
    if n <= 1: # Ungleichheit nicht definiert für 0 oder 1 Wert
        logger.debug(f"Theil: Nicht genug positive Werte ({n}).")
        return 0.0

    mean_val = np.mean(vals)
    if mean_val < EPSILON: # Durchschnitt ist praktisch Null
        logger.debug("Theil: Durchschnittswert ist nahe Null.")
        # Wenn alle Werte sehr klein sind, ist Ungleichheit auch ~0
        return 0.0

    # Theil's T Formel: (1/n) * Summe [ (xi / mean) * log(xi / mean) ]
    # Vermeide log(0) oder Division durch Null (bereits durch Filterung von > EPSILON)
    with np.errstate(divide='ignore', invalid='ignore'): # Unterdrücke Warnungen für log(small number)
        ratios = vals / mean_val
        log_ratios = np.log(ratios)
        # Ersetze NaN oder Inf durch 0 (passiert wenn ratio -> 0, dann ist ratio*log(ratio) -> 0)
        log_ratios[~np.isfinite(log_ratios)] = 0.0
        theil_t = (1 / n) * np.sum(ratios * log_ratios)

    logger.debug(f"Theil-Index berechnet: {theil_t:.4f}")
    return float(max(0.0, theil_t)) # Theil sollte nicht negativ sein


def atkinson_index(values: Union[List[float], np.ndarray], epsilon: float = 0.5) -> float:
    """
    Berechnet den Atkinson-Index mit einem Ungleichheitsaversionsparameter epsilon.

    Misst Ungleichheit, wobei das Ergebnis als der Anteil des Gesamteinkommens/-vermögens
    interpretiert werden kann, der "verschwendet" wird, um die aktuelle Ungleichheit
    zu erreichen, im Vergleich zu einer perfekt gleichen Verteilung.
    Erfordert positive Werte und epsilon != 1.

    Args:
        values: Liste oder NumPy-Array von numerischen Werten.
        epsilon: Ungleichheitsaversionsparameter (typischerweise > 0 und != 1).
                 Höheres epsilon = höhere Aversion gegen Ungleichheit.

    Returns:
        Atkinson-Index (0 bis 1). Gibt 0.0 bei leeren/ungültigen Inputs oder epsilon=1 zurück.
    """
    if abs(epsilon - 1.0) < EPSILON:
        logger.warning("Atkinson-Index ist für epsilon=1 nicht standardmäßig definiert (erfordert geometrischen Mittelwert). Gebe 0.0 zurück.")
        # Alternativ: geometrischen Mittelwert implementieren
        return 0.0
    if epsilon < 0:
         logger.warning(f"Atkinson-Index: Epsilon ({epsilon}) sollte nicht-negativ sein. Ergebnis könnte uninterpretiert sein.")

    try:
        vals = np.asarray([v for v in values if v is not None and np.isfinite(v) and v > EPSILON], dtype=float)
    except Exception as e:
        logger.warning(f"Atkinson: Fehler bei Konvertierung zu Array: {e}. Gebe 0.0 zurück.")
        return 0.0

    n = len(vals)
    if n <= 1:
        logger.debug(f"Atkinson: Nicht genug positive Werte ({n}).")
        return 0.0

    mean_val = np.mean(vals)
    if mean_val < EPSILON:
        logger.debug("Atkinson: Durchschnittswert ist nahe Null.")
        return 0.0

    # Berechne Atkinson-Index: 1 - [ (1/n) * Summe[ (xi / mean)^(1-epsilon) ] ]^(1 / (1-epsilon))
    exponent = 1.0 - epsilon
    with np.errstate(divide='ignore', invalid='ignore', power='ignore'): # Potentielle Fehler abfangen
        term_inside_sum = np.power(vals / mean_val, exponent)
        # Prüfe auf ungültige Ergebnisse nach Potenzierung
        if not np.all(np.isfinite(term_inside_sum)):
             logger.warning("Atkinson: Ungültige Werte nach Potenzierung aufgetreten. Ergebnis könnte ungenau sein.")
             term_inside_sum = term_inside_sum[np.isfinite(term_inside_sum)]
             if len(term_inside_sum) == 0: return 0.0 # Keine gültigen Terme mehr

        mean_of_powers = np.mean(term_inside_sum)

        if mean_of_powers < 0 and abs(1.0 / exponent) % 1 != 0:
            # Verhindere Berechnung von Wurzeln aus negativen Zahlen, wenn Exponent nicht passend
            logger.warning(f"Atkinson: Kann Index nicht berechnen (negative Basis für gebrochenen Exponenten). Mean of Powers: {mean_of_powers}, Exponent: {1.0/exponent}")
            return 0.0 # Oder anderer Fehlerwert?

        try:
             # Potenzieren mit 1/(1-epsilon)
             ede = np.power(mean_of_powers, 1.0 / exponent)
             # Index = 1 - (EDE / Mean) -- EDE = Equally Distributed Equivalent income
             atkinson = 1.0 - ede # Da wir mit (xi / mean) gerechnet haben, ist ede bereits relativ zum Mean
        except (ValueError, OverflowError) as e:
             logger.warning(f"Atkinson: Numerischer Fehler bei Potenzierung: {e}")
             return 0.0

    # Stelle sicher, dass Ergebnis im gültigen Bereich ist
    result = float(np.clip(atkinson, 0.0, 1.0))
    logger.debug(f"Atkinson-Index (eps={epsilon:.2f}) berechnet: {result:.4f}")
    return result


def add_differential_privacy_noise(value: float, epsilon_dp: float, sensitivity: float = 1.0) -> float:
    """
    Fügt Laplace-Rauschen hinzu, um Differential Privacy zu gewährleisten.

    Args:
        value: Der Originalwert.
        epsilon_dp: Privacy-Budget (kleiner = mehr Privatsphäre, mehr Rauschen). Muss > 0 sein.
        sensitivity: Die maximale Änderung des Wertes durch die Daten einer einzelnen Entität.

    Returns:
        Der verrauschte Wert.
    """
    if epsilon_dp <= 0:
        logger.error("Differential Privacy Epsilon muss positiv sein.")
        # Was tun? Fehler werfen oder Originalwert zurückgeben? Hier: Originalwert
        return value
    if sensitivity <= 0:
         logger.warning("Differential Privacy Sensitivity sollte positiv sein.")
         sensitivity = 1.0 # Fallback

    scale = sensitivity / epsilon_dp
    noise = np.random.laplace(loc=0.0, scale=scale)
    return value + noise


def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    """
    Berechnet die euklidische Distanz (L2-Norm) zwischen zwei Vektoren.

    Args:
        u: Erster Vektor (NumPy Array).
        v: Zweiter Vektor (NumPy Array). Muss gleiche Länge wie u haben.

    Returns:
        Die euklidische Distanz (float, >= 0).
    """
    if u.shape != v.shape:
        raise ValueError(f"Vektoren für euklidische Distanz müssen gleiche Shape haben: {u.shape} vs {v.shape}")
    try:
        distance = np.linalg.norm(u - v)
        return float(distance)
    except Exception as e:
        logger.error(f"Fehler bei Berechnung der euklidischen Distanz: {e}")
        return float('inf') # Unendliche Distanz bei Fehler


def hyperbolic_distance(u: np.ndarray, v: np.ndarray, epsilon: float = EPSILON) -> float:
    """
    Berechnet die hyperbolische Distanz im Poincaré-Ball-Modell.

    Nimmt an, dass die Norm der Vektoren < 1 ist.

    Args:
        u: Erster Vektor (NumPy Array).
        v: Zweiter Vektor (NumPy Array).
        epsilon: Kleiner Wert zur numerischen Stabilisierung.

    Returns:
        Hyperbolische Distanz (float, >= 0).
    """
    if u.shape != v.shape:
        raise ValueError(f"Vektoren für hyperbolische Distanz müssen gleiche Shape haben: {u.shape} vs {v.shape}")

    try:
        norm_u_sq = np.dot(u, u)
        norm_v_sq = np.dot(v, v)
        diff_sq = np.dot(u - v, u - v)

        # Sicherstellen, dass Normen < 1 sind (für Poincaré Ball)
        if norm_u_sq >= 1.0 or norm_v_sq >= 1.0:
             logger.warning(f"Hyperbolische Distanz: Vektornormen sollten < 1 sein (u: {norm_u_sq:.3f}, v: {norm_v_sq:.3f}). Ergebnis könnte ungenau sein.")
             # Optional: Vektoren projizieren? Hier nicht.

        denominator = (1.0 - norm_u_sq) * (1.0 - norm_v_sq)
        # Numerische Stabilität: Nenner darf nicht zu klein/negativ werden
        if denominator < epsilon:
             logger.debug(f"Hyperbolische Distanz: Nenner nahe Null ({denominator:.2e}). Setze auf Epsilon.")
             denominator = epsilon

        # Argument für arccosh
        # arcosh(x) ist nur für x >= 1 definiert
        argument = 1.0 + 2.0 * (diff_sq / denominator)

        # Sicherstellen, dass Argument >= 1 ist (kann durch Fließkommafehler < 1 werden)
        argument = max(1.0, argument)

        distance = np.arccosh(argument)
        return float(distance)

    except Exception as e:
        logger.error(f"Fehler bei Berechnung der hyperbolischen Distanz: {e}")
        return float('inf')

def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """
    Berechnet die Softmax-Funktion für einen Vektor, optional mit Temperatur.

    Args:
        x: Eingabe-Vektor (NumPy Array).
        temp: Temperaturparameter (> 0). Höhere Temp -> gleichmäßigere Verteilung,
              Temp -> 0 -> nähert sich argmax an.

    Returns:
        NumPy Array mit Wahrscheinlichkeitsverteilung (Summe ist 1).
    """
    if temp <= 0:
        logger.warning(f"Softmax Temperatur muss positiv sein, war {temp}. Setze auf 1.0.")
        temp = 1.0
    if x.size == 0:
        return np.array([]) # Leeres Array zurückgeben

    try:
        # Stabilisierung: Subtrahiere Maximum für numerische Stabilität
        x_stable = x / temp
        x_stable = x_stable - np.max(x_stable)
        exp_x = np.exp(x_stable)
        sum_exp_x = np.sum(exp_x)

        if sum_exp_x < EPSILON:
             logger.warning("Softmax: Summe der Exponenten ist nahe Null. Gebe gleichmäßige Verteilung zurück.")
             return np.ones_like(x) / x.size # Gleichverteilung als Fallback

        return exp_x / sum_exp_x
    except Exception as e:
         logger.error(f"Fehler bei Softmax-Berechnung: {e}")
         # Fallback: Gleichverteilung
         return np.ones_like(x) / x.size


def clip_norm(x: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    """
    Skaliert einen Vektor neu, sodass seine L2-Norm maximal 'max_norm' beträgt.

    Args:
        x: Eingabe-Vektor (NumPy Array).
        max_norm: Die maximal erlaubte L2-Norm (> 0).

    Returns:
        Der ggf. neu skalierte Vektor.
    """
    if max_norm <= 0:
        logger.error(f"clip_norm: max_norm muss positiv sein, war {max_norm}.")
        return np.zeros_like(x) # Nullvektor als Fallback

    norm = np.linalg.norm(x)
    if norm > max_norm:
        return x * (max_norm / (norm + EPSILON)) # Füge EPSILON hinzu, falls norm=0 sein könnte
    else:
        return x


def moving_average(values: List[float], window: int) -> List[float]:
    """
    Berechnet den gleitenden Durchschnitt einer Zeitreihe.

    Args:
        values: Liste von Zahlenwerten.
        window: Größe des Fensters für den Durchschnitt. Muss positiv sein.

    Returns:
        Liste mit den gleitenden Durchschnittswerten. Die Liste ist kürzer
        als die Eingabe (`len(values) - window + 1`). Gibt leere Liste zurück bei Fehlern.
    """
    if window <= 0:
        logger.error(f"moving_average: Fenstergröße muss positiv sein, war {window}.")
        return []
    if not values or len(values) < window:
        logger.debug(f"moving_average: Nicht genug Werte ({len(values)}) für Fenstergröße {window}.")
        return []

    try:
        # Nutze np.convolve für effiziente Berechnung
        weights = np.ones(window) / window
        ma = np.convolve(values, weights, mode='valid')
        return ma.tolist()
    except Exception as e:
        logger.error(f"Fehler bei Berechnung des gleitenden Durchschnitts: {e}")
        return []