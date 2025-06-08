# governance/__init__.py

"""
Das 'governance'-Paket bündelt alle Klassen und Module, die
staatliche oder regierungsspezifische Funktionen abbilden. Dazu
gehören fiskalische, sozialpolitische und weitere
Meta-Governance-Aspekte.

In diesem Paket befinden sich:
- government.py (GovernmentAgent und zugehörige Intelligence)
- ggf. weitere Module (z.B. für kommunale Verwaltung, regulatorische Behörden usw.)
"""

from .government import GovernmentAgent, GovernmentIntelligence

__all__ = [
    "GovernmentAgent",
    "GovernmentIntelligence",
]
