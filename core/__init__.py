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
    "resource_regen",              # Resource agent regeneration
    "state_estimation",            # Estimate global/agent state
    "need_estimation",             # Update consumer needs
    "infrastructure_development",  # Update infrastructure capacities
    "system5_policy",              # High-level policy adjustments
    "system4_strategic_planning_admm",  # Long-term planning with ADMM
    "system4_tactical_planning",   # Medium-term planning
    "system3_aggregation",         # Aggregate regional data
    "system2_coordination",        # Inter-regional coordination
    "system3_feedback",            # Feedback to regions
    "broadcast_plan_and_prices",   # Share global plan and price signals
    "system1_operations",          # Operational execution preparation
    "local_production_planning",   # Producer-level planning
    "local_rl_execution",          # Local RL-driven decisions
    "admm_update",                 # ADMM constraint resolution
    "production_execution",        # Actual production execution
    "consumption",                 # Consumer consumption
    "environmental_impact",        # Calculate environmental impacts
    "technology_progress",         # Update technology levels
    "crisis_management",           # Handle active crises
    "welfare_assessment",          # Calculate welfare metrics
    "learning_adaptation",         # ML-based adaptation
    "vsm_reconfiguration",         # Dynamic VSM reconfiguration
    "bookkeeping"                  # Final data collection
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
