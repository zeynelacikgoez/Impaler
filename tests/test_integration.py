# tests/test_integration.py
"""
Integrationstests für das Impaler-Framework:
- Testen das Zusammenspiel verschiedener Module (EconomicModel, Agents, usw.)
- Keine tiefen Unit-Tests einzelner Klassen, sondern Fokus auf den Gesamtablauf.
"""

import pytest
import math # For math.isclose if needed for float comparisons
from collections import defaultdict

# Passe die Importpfade an dein Projekt an, falls sie anders lauten
from impaler.core.config import SimulationConfig, AgentPopulationConfig, RegionalConfig, DemandConfig, ConsumerAdaptationParams, ScenarioConfig, SpecificAgentConfig
from impaler.core.model import EconomicModel
from impaler.agents.consumer import ConsumerAgent # Needed for type checking in model setup


@pytest.fixture
def basic_sim_config():
    """A very basic simulation config for integration tests."""
    return SimulationConfig(
        agent_populations={
            "producers_pop": AgentPopulationConfig(
                agent_class="ProducerAgent", 
                count=1,
                params={"production_lines": [{"output_good": "A", "base_efficiency": 1.0, "input_requirements": []}]}
            )
        },
        regional_config=RegionalConfig(regions=["Region1"]),
        simulation_steps=3
    )

def test_small_simulation_run(basic_sim_config):
    """
    Führt eine minimale Simulation mit wenigen Agenten und wenigen Schritten durch
    und prüft, ob sie ohne Fehler läuft und das Ergebnis plausibel ist.
    """
    model = EconomicModel(config=basic_sim_config)

    for _ in range(basic_sim_config.simulation_steps):
        model.step()

    summary = model.get_summary_statistics()
    assert summary.get("last_total_output", summary.get("total_output", -1)) >= 0, "Total Output sollte nicht negativ sein."


def test_medium_simulation_with_admm(basic_sim_config):
    """
    Testet eine mittelgroße Simulation mit ADMM- und VSM-Komponenten aktiv,
    um zu prüfen, ob das Modell korrekt durchläuft und Werte plausibel bleiben.
    """
    # Modify basic_sim_config for this test
    basic_sim_config.agent_populations["producers_pop"].count = 5
    basic_sim_config.regional_config.regions = ["Region1", "Region2"]
    basic_sim_config.simulation_steps = 10
    basic_sim_config.vsm_on = True
    basic_sim_config.admm_on = True
    
    model = EconomicModel(config=basic_sim_config)

    for _ in range(model.config.simulation_steps):
        model.step()

    summary = model.get_summary_statistics()
    gini_val = summary.get("production_gini", summary.get("welfare", {}).get("production_gini", 0.0))
    assert 0.0 <= gini_val <= 1.0, "Gini-Koeffizient außerhalb [0,1] ist unwahrscheinlich."


def test_small_run_no_exceptions(basic_sim_config):
    """
    'Smoke Test': Testet, ob bei wenigen Schritten und Agenten
    keinerlei Exceptions auftreten (stabiler Grundlauf).
    """
    model = EconomicModel(config=basic_sim_config) # Uses the 1 producer, 1 region, 3 steps config

    for _ in range(model.config.simulation_steps):
        model.step()
    assert True


# --- Tests for Regional Demand Aggregation ---

@pytest.fixture
def demand_aggregation_config():
    """
    SimulationConfig tailored for testing regional demand aggregation.
    Defines consumers in different regions with specific needs.
    """
    # Define agent populations for consumers
    consumer_populations = {
        "consumers_region_A": AgentPopulationConfig(
            agent_class="ConsumerAgent",
            count=2,
            region_distribution={"region_A": 2}, # Assign all to region_A
            params={ # Params applied to each agent in this population
                "base_needs": {"food": 10, "water": 5}, # ConsumerA1, ConsumerA2
                "demographic_type": "test_type_A" 
            }
        ),
        "consumers_region_B": AgentPopulationConfig(
            agent_class="ConsumerAgent",
            count=1,
            region_distribution={"region_B": 1}, # Assign all to region_B
            params={
                "base_needs": {"food": 8, "tools": 15}, # ConsumerB1
                "demographic_type": "test_type_B"
            }
        )
    }
    # Minimal producer to make model happy if it expects some
    producer_populations = {
         "dummy_producers": AgentPopulationConfig(agent_class="ProducerAgent", count=0)
    }
    
    # Combine populations
    agent_populations_combined = {**consumer_populations, **producer_populations}


    return SimulationConfig(
        simulation_name="TestDemandAggregation",
        simulation_steps=1, # Only need to run one step (or just the stage)
        agent_populations=agent_populations_combined,
        specific_agents=[], # No specific agents for this test
        regional_config=RegionalConfig(regions=["region_A", "region_B"]),
        demand_config=DemandConfig(), # Use default demand config
        scenario_config=ScenarioConfig(),
        stages=["need_estimation"] # Focus only on this stage
    )

class TestRegionalDemandAggregation:

    @pytest.mark.parametrize("parallel_execution_flag", [False, True])
    def test_regional_and_global_demand_aggregation(self, demand_aggregation_config, parallel_execution_flag):
        """
        Tests if consumer demands are correctly aggregated globally and regionally,
        for both sequential and parallel execution modes.
        """
        config = demand_aggregation_config
        config.parallel_execution = parallel_execution_flag
        if parallel_execution_flag:
            config.num_workers = 2 # Ensure workers if parallel

        model = EconomicModel(config=config)
        
        # Ensure consumers are created as expected
        assert len(model.consumers) == 3
        consumers_in_A = [c for c in model.consumers if c.region_id == "region_A"]
        consumers_in_B = [c for c in model.consumers if c.region_id == "region_B"]
        assert len(consumers_in_A) == 2
        assert len(consumers_in_B) == 1

        # Manually set/verify needs for clarity, overriding potential demographic factors for this test
        # Consumer agents' base_needs are set via params in AgentPopulationConfig
        # We assume the __init__ of ConsumerAgent correctly sets self.base_needs from these params.
        # If demographic_factors significantly alter these, this test might need adjustment or
        # a very basic demographic_type. For now, assume "test_type_A/B" have no factors.
        
        # Override needs after agent creation for precise control in this test
        for consumer in model.consumers:
            if consumer.unique_id.startswith("consumers_region_A"): #IDs are like "consumers_region_A_0"
                consumer.base_needs = {"food": 10, "water": 5} # For both consumers in region_A
            elif consumer.unique_id.startswith("consumers_region_B"):
                consumer.base_needs = {"food": 8, "tools": 15}


        # Execute the need_estimation stage
        # model.stage_manager.run_stages(parallel=model.parallel_execution) # This would run all defined stages
        model.stage_need_estimation() # More direct for this test

        # --- Assertions ---
        agg_demand = model.aggregated_consumer_demand
        reg_demand = model.regional_consumer_demand

        # Global Demand
        # Food: (10+10) from region_A + 8 from region_B = 28
        # Water: (5+5) from region_A = 10
        # Tools: 15 from region_B = 15
        assert math.isclose(agg_demand.get("food", 0.0), 28.0)
        assert math.isclose(agg_demand.get("water", 0.0), 10.0)
        assert math.isclose(agg_demand.get("tools", 0.0), 15.0)
        assert len(agg_demand) == 3 # Only these 3 goods

        # Regional Demand - Region A
        # Food: 10 + 10 = 20
        # Water: 5 + 5 = 10
        assert "region_A" in reg_demand
        assert math.isclose(reg_demand["region_A"].get("food", 0.0), 20.0)
        assert math.isclose(reg_demand["region_A"].get("water", 0.0), 10.0)
        assert reg_demand["region_A"].get("tools", 0.0) == 0.0 # No tools in region_A
        assert len(reg_demand["region_A"]) == 2 # Food, Water

        # Regional Demand - Region B
        # Food: 8
        # Tools: 15
        assert "region_B" in reg_demand
        assert math.isclose(reg_demand["region_B"].get("food", 0.0), 8.0)
        assert reg_demand["region_B"].get("water", 0.0) == 0.0 # No water in region_B
        assert math.isclose(reg_demand["region_B"].get("tools", 0.0), 15.0)
        assert len(reg_demand["region_B"]) == 2 # Food, Tools

        # Check if S4 planner received the update (optional, if S4 is part of the minimal setup)
        if model.system4planner and hasattr(model.system4planner, 'update_demand_forecast'):
            # This part depends on S4's internal state, could be mocked or checked if simple
            # For now, this test focuses on model.aggregated_consumer_demand and model.regional_consumer_demand
            pass


# General Recommendation for Performance Benchmark Tests:
# It is highly recommended to add performance benchmark tests for critical,
# computationally intensive sections of the simulation. This helps in:
#   - Identifying performance bottlenecks.
#   - Tracking performance regressions or improvements across code changes.
#   - Evaluating the impact of different optimization strategies.
#
# Tools like `pytest-benchmark` or the standard library's `timeit` can be used.
#
# Key areas to consider for benchmarking:
# 1. ProducerAgent.local_subproblem_admm:
#    - Benchmark with varying numbers of goods, production lines, and constraints.
#    - Compare performance of different SciPy solver methods if applicable.
#
# 2. DataCollector methods:
#    - If data collection, especially with high frequency or high detail_level,
#      is suspected to be a bottleneck, benchmark methods like `_collect_agent_data`
#      and `export_data` with varying numbers of agents and history length.
#
# 3. Overall simulation step time:
#    - Benchmark `EconomicModel.step()` with different configurations:
#      - Varying number of agents (producers, consumers).
#      - ADMM enabled vs. disabled.
#      - Parallel execution enabled vs. disabled.
#      - Different VSM system complexities or feature flags.
#
# 4. Specific stage functions in EconomicModel:
#    - Stages like `stage_need_estimation`, `stage_production_execution`,
#      `stage_consumption` which involve loops over many agents, especially
#      when parallelized, should be benchmarked to understand scaling.
#
# Example using pytest-benchmark (conceptual):
#
# import pytest
#
# def my_critical_function(param1, param2):
#     # ... some computation ...
#     pass
#
# @pytest.fixture
# def setup_params_for_benchmark(): # Example fixture
#     # Setup complex parameters here if needed
#     return "param_value1", "param_value2"
#
# @pytest.mark.benchmark(group="critical_funcs")
# def test_my_critical_function_performance(benchmark, setup_params_for_benchmark):
#     param1, param2 = setup_params_for_benchmark
#     benchmark(my_critical_function, param1, param2)
#
# This requires installing pytest-benchmark and structuring tests accordingly.
# Consider creating a separate test file (e.g., tests/test_performance.py) for these.

