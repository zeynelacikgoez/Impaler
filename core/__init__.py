"""
Core module for the economic simulation framework.

This module contains the central components of the economic simulation:
- EconomicModel: Main simulation model that coordinates all agents and systems
- StageManager: Flexible stage execution and management
- CrisisManager: Advanced crisis simulation and management
- DataCollector: Modular data collection and analysis
- SimulationConfig: Configuration system for customizing model parameters
"""

from .model import EconomicModel
from .stagemanager import StageManager
from .crisismanager import CrisisManager
from .datacollector import DataCollector
from .config import (
    SimulationConfig,
    ConsumerAdaptationParams,
    ProducerParamsConfig,
    EconomicModelConfig,
    IOConfig,
    InnovationEffect,
)

# Version information
__version__ = "0.2.0"

# Default configuration
default_config = SimulationConfig().to_dict()

# Simulation stages in recommended execution order
# Note: The new StageManager allows dynamic registration of stages,
# but we keep this list for backward compatibility
SIMULATION_STAGES = [
    "resource_regen",              # Umwelt regeneriert sich
    "need_estimation",             # Konsumentenbedarfe werden aktualisiert
    "state_estimation",            # AEKF schätzt den globalen Zustand
    "system5_policy",              # S5 setzt übergeordnete Ziele
    "system4_strategic_planning_admm",  # S4 führt ADMM aus
    "broadcast_plan_and_prices",   # Plan und Preise an Agenten
    "system3_coordination",        # Regionale Vorgaben setzen
    "system2_coordination",        # Inter-regionale Koordination
    "local_rl_execution",          # RL-Agenten wählen Aktionen
    "production_execution",        # Producer führen Aktionen aus
    "consumption",                 # Konsumenten verbrauchen Güter
    "environmental_impact",        # Umweltauswirkungen berechnen
    "crisis_management",           # Krisen werden gehandhabt
    "welfare_assessment",          # Wohlfahrt und Gini
    "bookkeeping"                  # Datensammlung
]

# Export main classes
__all__ = [
    'EconomicModel',
    'StageManager',
    'CrisisManager',
    'DataCollector',
    'SimulationConfig',
    'ConsumerAdaptationParams',
    'ProducerParamsConfig',
    'EconomicModelConfig',
    'IOConfig',
    'InnovationEffect',
    'default_config',
    'SIMULATION_STAGES'
]
