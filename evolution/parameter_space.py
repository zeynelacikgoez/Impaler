from typing import Dict, Any

# Definiert den Suchraum fuer jedes Parameter.
# Format: (typ, [moegliche Werte] oder (min, max))
# 'typ' kann 'int', 'float', 'categorical' oder 'bool' sein.
# 'path' ist der Punkt-getrennte Pfad zum Parameter in SimulationConfig.

PARAMETER_SPACE: Dict[str, Dict[str, Any]] = {
    # VSM-Parameter (Systemdynamik)
    "s3_autonomy_level": {
        "path": "system3_params.autonomy_level",
        "type": "float",
        "range": (0.2, 0.8),
    },
    "s4_admm_rho": {
        "path": "admm_config.rho",
        "type": "float",
        "range": (0.05, 5.0),
    },
    "s2_fairness_threshold": {
        "path": "system2_params.fairness_threshold",
        "type": "float",
        "range": (0.1, 0.5),
    },
    "s3_conflict_strategy": {
        "path": "system3_params.initial_conflict_strategy",
        "type": "categorical",
        "values": ["proportional", "priority_based", "bargaining"],
    },

    # Governance-Parameter (Werte & Ziele)
    "fairness_weight": {
        "path": "planning_priorities.fairness_weight",
        "type": "float",
        "range": (0.5, 3.0),
    },
    "co2_weight": {
        "path": "planning_priorities.co2_weight",
        "type": "float",
        "range": (0.1, 10.0),
    },
    "resilience_weight": {
        "path": "planning_priorities.resilience_weight",
        "type": "float",
        "range": (0.2, 2.5),
    },

    # Agenten-Parameter (Verhaltensweisen)
    "consumer_lifestyle_adapt_interval": {
        "path": "agent_populations.default_consumers.params.lifestyle_adapt_interval",
        "type": "int",
        "range": (5, 20),
    },
    "producer_enable_rl": {
        "path": "agent_populations.default_producers.params.enable_rl",
        "type": "bool",
    },
    "producer_depreciation_rate": {
        "path": "agent_populations.default_producers.params.depreciation_rate",
        "type": "float",
        "range": (0.01, 0.05),
    },
    "producer_compliance_level": {
        "path": "agent_populations.default_producers.params.compliance_level",
        "type": "float",
        "range": (0.7, 1.0),
        "description": "Wie stark ein Produzent versucht, Planziele exakt zu erfüllen."
    },
    "producer_hoarding_propensity": {
        "path": "agent_populations.default_producers.params.hoarding_propensity",
        "type": "float",
        "range": (0.0, 0.4),
        "description": "Die Neigung eines Produzenten, ungenutzte Ressourcen zu horten."
    },

    # Strukturelle Parameter (Zusammensetzung der Wirtschaft)
    "producer_count_ratio": {
        "path": "structure.producer_consumer_ratio",
        "type": "float",
        "range": (0.1, 0.5),  # 10-50% der Gesamtpopulation sind Producer
    },
    "transport_capacity_factor": {
        "path": "structure.transport_capacity_factor",
        "type": "float",
        "range": (0.5, 2.0),  # Skaliert die Basis-Transportkapazitaet
    },

    # --- NEUE GENE FÜR PLANWIRTSCHAFTLICHE STEUERUNG ---
    "underproduction_penalty": {
        "path": "planning_priorities.planwirtschaft_params.underproduction_penalty",
        "type": "float",
        "range": (0.5, 10.0),
        "description": "Globaler Straffaktor für zu wenig Produktion.",
    },
    "overproduction_penalty": {
        "path": "planning_priorities.planwirtschaft_params.overproduction_penalty",
        "type": "float",
        "range": (0.0, 0.5),
        "description": "Globaler Straffaktor für zu viel Produktion.",
    },
    "inventory_cost": {
        "path": "planning_priorities.planwirtschaft_params.inventory_cost",
        "type": "float",
        "range": (0.01, 0.3),
        "description": "Kosten für das Halten von ungenutztem Inventar.",
    },
    "priority_sector_demand_factor": {
        "path": "planning_priorities.planwirtschaft_params.priority_sector_demand_factor",
        "type": "float",
        "range": (1.0, 1.5),
        "description": "Bedarfs-Boost für priorisierte Sektoren.",
    },

    # --- NEUES KOMPLEXES GEN ---
    "base_co2_penalty": {
        "path": "planning_priorities.planwirtschaft_params.co2_penalties",
        "type": "float",
        "range": (1.0, 50.0),
        "is_complex": True,
        "description": "Basiswert für die CO2-Bepreisung, der skaliert wird.",
    },
}


from pydantic import BaseModel
from impaler.core.config import SimulationConfig, create_default_config, AgentPopulationConfig


def _set_nested_value(obj: Any, path: str, value: Any) -> None:
    """Set a nested attribute or dictionary entry specified by a dot path."""
    parts = path.split('.')
    current = obj
    for part in parts[:-1]:
        if isinstance(current, BaseModel):
            if getattr(current, part, None) is None:
                setattr(current, part, {})
            current = getattr(current, part)
        elif isinstance(current, dict):
            current = current.setdefault(part, {})
        else:
            raise ValueError(f"Cannot traverse part '{part}' in path '{path}'.")
    last = parts[-1]
    if isinstance(current, BaseModel):
        setattr(current, last, value)
    elif isinstance(current, dict):
        current[last] = value
    else:
        raise ValueError(f"Cannot set part '{last}' in path '{path}'.")


def generate_config_from_genome(genome: Dict[str, Any]) -> SimulationConfig:
    """Create a SimulationConfig from a genome dictionary."""
    config = create_default_config()

    # Apply direct parameters
    for gene, val in genome.items():
        spec = PARAMETER_SPACE.get(gene)
        if not spec or spec.get("is_complex"):
            continue
        if gene in {"producer_count_ratio", "transport_capacity_factor"}:
            continue
        _set_nested_value(config, spec["path"], val)

    # Handle virtual parameters
    if "producer_count_ratio" in genome:
        ratio = genome["producer_count_ratio"]
        pops = config.agent_populations
        prod_pop = pops.get("default_producers")
        cons_pop = pops.get("default_consumers")
        if isinstance(prod_pop, AgentPopulationConfig) and isinstance(cons_pop, AgentPopulationConfig):
            total = prod_pop.count + cons_pop.count
            prod_count = max(1, int(total * ratio))
            cons_count = max(0, total - prod_count)
            prod_pop.count = prod_count
            cons_pop.count = cons_count

    if "transport_capacity_factor" in genome:
        factor = genome["transport_capacity_factor"]
        regions = config.regional_config.regions
        new_cap: Dict[str, Dict[str, float]] = {}
        for r1 in regions:
            new_cap[r1] = {}
            for r2 in regions:
                if r1 == r2:
                    continue
                new_cap[r1][r2] = 100.0 * factor
        config.regional_config.transport_capacity = new_cap

    # --- Spezialbehandlung für komplexe Gene ---
    if "base_co2_penalty" in genome:
        base_value = genome["base_co2_penalty"]
        co2_penalties_dict: Dict[str, float] = {}
        for good in config.goods:
            co2_penalties_dict[good] = base_value
        spec = PARAMETER_SPACE["base_co2_penalty"]
        _set_nested_value(config, spec["path"], co2_penalties_dict)

    return config

