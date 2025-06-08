# ml/__init__.py

"""
Das ml-Paket enthält verschiedene Reinforcement-Learning-Implementierungen, 
wie tabellarisches Q-Learning, Deep Q-Learning und QMIX für Multi-Agent RL.
Durch den Import in dieser __init__.py-Datei wird das Paket 
einfacher zugänglich gemacht.
"""

from .q_learning import QLearningAgent, DeepQLearningAgent

# Falls QMIX separat angefragt/benutzt wird, hier ebenfalls importieren:
try:
    from .qmix import QMIXNetwork, QMIXAgent
except ImportError:
    # Falls qmix.py nicht vorhanden oder andere Abhängigkeiten fehlen, ignorieren
    QMIXNetwork = None
    QMIXAgent = None

__all__ = [
    "QLearningAgent",
    "DeepQLearningAgent",
    "QMIXNetwork",
    "QMIXAgent"
]
