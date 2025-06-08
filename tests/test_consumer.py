# tests/test_consumer.py

import pytest
import math
import numpy as np
from unittest.mock import MagicMock, patch

from impaler.agents.consumer import ConsumerAgent, NeedPriority # Import Enum
from impaler.core.config import SimulationConfig, DemandConfig, ConsumerAdaptationParams, ScenarioConfig
from collections import defaultdict


@pytest.fixture
def dummy_model_config():
    """ Provides a basic SimulationConfig with a DemandConfig. """
    cfg = SimulationConfig()
    # ``ConsumerAdaptationParams`` ist selbst ein Pydantic-Modell.  In der
    # aktuellen Implementierung erwartet ``DemandConfig`` fuer dieses Feld ein
    # Dictionary.  Die Tests wurden daher angepasst, um ``model_dump`` zu
    # verwenden.
    cfg.demand_config = DemandConfig(
        consumer_adaptation_params=ConsumerAdaptationParams().model_dump(),
        seasonal_factors={}
    )
    # ``ScenarioConfig`` enthaelt keine ``external_events`` mehr; wir verwenden
    # die Defaults.
    cfg.scenario_config = ScenarioConfig()
    return cfg

@pytest.fixture
def dummy_model(dummy_model_config):
    """
    Erzeugt ein Mock-Objekt für das 'model', das ProducerAgent normalerweise erwartet.
    Du kannst hier beliebige Felder hinzufügen, die für Tests relevant sind.
    """
    model = MagicMock()
    model.config = dummy_model_config # Use the more complete config
    model.logger = MagicMock()
    model.current_step = 0
    return model


# Keep existing tests if they are still relevant
class TestConsumerAgentOriginal: # Renamed to avoid conflict if needed

    def test_consumer_initialization(self, dummy_model):
        """
        Testet, ob ConsumerAgent korrekt initialisiert wird,
        wenn keine base_needs und preference_weights übergeben werden.
        """
        consumer = ConsumerAgent(
            unique_id="cons1",
            model=dummy_model, 
            region_id=0 
        )

        assert consumer.unique_id == "cons1"
        assert consumer.region_id == 0
        # Default-Werte laut consumer.py (wenn create_consumer_need_profile("standard") das liefert)
        # This depends on create_consumer_need_profile, so making it more general
        assert isinstance(consumer.base_needs, dict)
        assert isinstance(consumer.preference_weights, dict)
        assert consumer.satisfaction == 0.5 # Initial default


    def test_consumer_with_custom_needs_and_preferences(self, dummy_model):
        """
        Testet die Initialisierung mit benutzerdefinierten
        needs und preferences.
        """
        custom_needs = {"food": 5.0, "water": 2.0}
        custom_prefs = {"food": 0.7, "water": 0.3}
        consumer = ConsumerAgent(
            unique_id="cons2",
            model=dummy_model,
            region_id=1,
            base_needs=custom_needs,
            preference_weights=custom_prefs
        )

        assert consumer.base_needs == custom_needs
        assert consumer.preference_weights == custom_prefs


    def test_receive_allocation(self, dummy_model):
        """
        Testet, ob receive_allocation() die Mengen
        korrekt ins consumption_received bucht.
        """
        consumer = ConsumerAgent("cons3", dummy_model, region_id=1, base_needs={"food":1}) # Base needs for pref norm
        allocation = {"food": 1.0, "water": 0.5}

        assert consumer.consumption_received["food"] == 0.0
        assert consumer.consumption_received["water"] == 0.0

        consumer.receive_allocation(allocation)

        assert consumer.consumption_received["food"] == 1.0
        assert consumer.consumption_received["water"] == 0.5


    def test_evaluate_consumption(self, dummy_model):
        """
        Testet, ob evaluate_consumption() die Satisfaction korrekt
        berechnet und consumption_actual (nicht received) für Historie nutzt.
        """
        consumer = ConsumerAgent("cons4", dummy_model, region_id=1, base_needs={"food": 2.0})
        consumer.consumption_actual = defaultdict(float, {"food": 2.0}) # Simulate consumption happened

        consumer.evaluate_consumption()

        assert consumer.satisfaction > 0 # Exact value depends on utility function
        # consumption_actual is used for history, not reset here.
        # consumption_received is reset in step_stage after evaluation.
        assert consumer.consumption_actual["food"] == 2.0 


    def test_step_stage_consumption_and_evaluation(self, dummy_model):
        """
        Testet den Aufruf von step_stage("consumption_and_evaluation"),
        ob es evaluate_consumption() auslöst und received zurücksetzt.
        """
        consumer = ConsumerAgent("cons5", dummy_model, region_id=1, base_needs={"food":2.0})
        consumer.consumption_received["food"] = 2.0  # Simuliere Zuteilung

        consumer.step_stage("consumption") # Or "consumption_and_evaluation"

        assert consumer.satisfaction > 0
        # consumption_received should be reset by reset_consumption_received called in step_stage
        assert consumer.consumption_received["food"] == 0.0 


    def test_get_summary_statistics(self, dummy_model):
        """
        Testet, ob get_summary_statistics() ein sinnvolles
        Dictionary zurückgibt, z.B. 'current_satisfaction'.
        """
        consumer = ConsumerAgent("cons6", dummy_model, region_id=1, base_needs={"food":1})
        consumer.satisfaction = 0.75

        stats = consumer.get_summary_statistics()
        assert isinstance(stats, dict)
        assert stats["agent_id"] == "cons6"
        assert stats["current_satisfaction"] == 0.75
        assert "average_satisfaction_hist" in stats # Changed from average_satisfaction
        assert "region_id" in stats # Changed from region


class TestConsumerAgentSubstitutionCache:

    @pytest.fixture
    def consumer_for_cache_tests(self, dummy_model):
        """Consumer with a defined set of substitution pairs."""
        base_needs = {"food": 10, "drink": 5, "snack": 3, "treat": 2}
        substitution_pairs = {
            ("food", "alt_food"): 0.8, # 1 alt_food replaces 0.8 food
            ("food", "super_alt_food"): 0.9, # Better substitute
            ("drink", "alt_drink"): 0.7,
            ("snack", "alt_snack_bad"): 0.3,
            ("snack", "alt_snack_good"): 0.6,
        }
        # Ensure all goods in substitution pairs are in base_needs for preference normalization
        all_goods_in_needs = set(base_needs.keys())
        for orig, sub in substitution_pairs.keys():
            all_goods_in_needs.add(orig)
            all_goods_in_needs.add(sub)
        
        final_base_needs = {good: base_needs.get(good, 1.0) for good in all_goods_in_needs}

        agent = ConsumerAgent(
            unique_id="cache_tester",
            model=dummy_model,
            region_id=1,
            base_needs=final_base_needs,
            substitution_pairs=substitution_pairs,
            preference_weights={g: 1/len(final_base_needs) for g in final_base_needs} # Equal weights
        )
        return agent

    def test_prepared_substitutes_initialization(self, consumer_for_cache_tests):
        agent = consumer_for_cache_tests
        cache = agent._prepared_substitutes

        assert "food" in cache
        assert cache["food"] == [("super_alt_food", 0.9), ("alt_food", 0.8)] # Sorted by factor desc

        assert "drink" in cache
        assert cache["drink"] == [("alt_drink", 0.7)]

        assert "snack" in cache
        assert cache["snack"] == [("alt_snack_good", 0.6), ("alt_snack_bad", 0.3)]

        assert "treat" not in cache # No substitutes defined for treat

    def test_prepared_substitutes_update_after_learning_factor_increase(self, consumer_for_cache_tests):
        agent = consumer_for_cache_tests
        
        # Initial state for "food"
        assert agent._prepared_substitutes["food"] == [("super_alt_food", 0.9), ("alt_food", 0.8)]

        # Learn that "alt_food" is a better substitute for "food"
        # Increase factor of ("food", "alt_food") from 0.8 to 0.95
        # Need multiple successful substitutions for a significant change
        original_factor = agent.substitution_pairs[("food", "alt_food")]
        target_factor = 0.95
        
        # Simulate learning that increases the factor
        with patch.object(agent, '_update_prepared_substitutes', wraps=agent._update_prepared_substitutes) as mock_update_cache:
            while agent.substitution_pairs[("food", "alt_food")] < target_factor:
                agent._learn_from_substitution("food", "alt_food", success=True)
                if agent.substitution_pairs[("food", "alt_food")] <= original_factor: # Ensure it's increasing
                    agent.substitution_pairs[("food", "alt_food")] = original_factor * (1.0 + agent.substitution_learning_rate * 10) # Boost for test
                original_factor = agent.substitution_pairs[("food", "alt_food")] # Update for next check
            
            assert mock_update_cache.called # Ensure cache update was triggered by learning

        # Verify the cache is updated and re-sorted
        assert agent._prepared_substitutes["food"][0][0] == "alt_food" # Should be the best now
        assert math.isclose(agent._prepared_substitutes["food"][0][1], agent.substitution_pairs[("food", "alt_food")])
        assert agent._prepared_substitutes["food"][1][0] == "super_alt_food"
        assert math.isclose(agent._prepared_substitutes["food"][1][1], 0.9)


    def test_prepared_substitutes_update_after_learning_factor_decrease(self, consumer_for_cache_tests):
        agent = consumer_for_cache_tests
        # Initial state for "snack"
        assert agent._prepared_substitutes["snack"] == [("alt_snack_good", 0.6), ("alt_snack_bad", 0.3)]
        
        # Learn that "alt_snack_good" is a worse substitute for "snack"
        # Decrease factor of ("snack", "alt_snack_good") from 0.6 by repeatedly failing
        with patch.object(agent, '_update_prepared_substitutes', wraps=agent._update_prepared_substitutes) as mock_update_cache:
            for _ in range(5): # Assuming 5 failures trigger factor reduction
                agent._learn_from_substitution("snack", "alt_snack_good", success=False)
            assert mock_update_cache.call_count >= 1 # Cache update should be triggered

        # Verify the cache is updated and re-sorted (or factor changed)
        # Original factor was 0.6. After 5 failures, it becomes 0.6 * 0.98 = 0.588
        assert agent._prepared_substitutes["snack"][0][0] == "alt_snack_good" # Still first if not below alt_snack_bad
        assert math.isclose(agent._prepared_substitutes["snack"][0][1], 0.6 * 0.98) 
        assert agent._prepared_substitutes["snack"][1][0] == "alt_snack_bad"
        assert math.isclose(agent._prepared_substitutes["snack"][1][1], 0.3)


    def test_apply_substitutions_uses_updated_cache(self, consumer_for_cache_tests, dummy_model):
        agent = consumer_for_cache_tests
        agent.base_needs = {"food": 10.0, "alt_food": 1.0, "super_alt_food": 1.0} # Need 10 food
        agent.preference_weights = {"food":0.5, "alt_food":0.25, "super_alt_food":0.25} # Re-normalize
        agent._normalize_preferences()
        
        # Initial state: super_alt_food (0.9) is better than alt_food (0.8)
        assert agent._prepared_substitutes["food"] == [("super_alt_food", 0.9), ("alt_food", 0.8)]

        # Scenario 1: Default preference
        agent.consumption_received = defaultdict(float, {"alt_food": 200.0, "super_alt_food": 200.0, "food": 0.0})
        agent.apply_substitutions()
        # Expected: uses super_alt_food. 10 food needed / 0.9 factor = 11.11 super_alt_food
        assert math.isclose(agent.consumption_actual["super_alt_food"], 200.0 - (10.0 / 0.9))
        assert math.isclose(agent.consumption_actual["alt_food"], 200.0) # Unchanged
        assert math.isclose(agent.consumption_actual["food"], 10.0) # Fully substituted

        # Scenario 2: Learn that alt_food becomes much better
        original_factor_alt_food = agent.substitution_pairs[("food", "alt_food")]
        with patch.object(agent, '_update_prepared_substitutes', wraps=agent._update_prepared_substitutes) as mock_update_cache:
            # Drastically increase alt_food's factor to ensure it's preferred
            agent.substitution_pairs[("food", "alt_food")] = 0.95 
            agent._learn_from_substitution("food", "alt_food", success=True) # This call will also trigger _update_prepared_substitutes
            assert mock_update_cache.called
        
        # Cache should now have alt_food as the best substitute
        assert agent._prepared_substitutes["food"][0] == ("alt_food", agent.substitution_pairs[("food", "alt_food")])
        assert agent._prepared_substitutes["food"][1] == ("super_alt_food", 0.9)
        
        # Reset consumption and apply again
        agent.consumption_received = defaultdict(float, {"alt_food": 200.0, "super_alt_food": 200.0, "food": 0.0})
        agent.consumption_actual.clear() # Clear previous actuals
        agent.apply_substitutions()
        
        # Expected: now uses alt_food because its factor is 0.95
        # 10 food needed / 0.95 factor = 10.526 alt_food
        assert math.isclose(agent.consumption_actual["alt_food"], 200.0 - (10.0 / agent.substitution_pairs[("food", "alt_food")]))
        assert math.isclose(agent.consumption_actual["super_alt_food"], 200.0) # Unchanged
        assert math.isclose(agent.consumption_actual["food"], 10.0) # Fully substituted

    def test_prepared_substitutes_handles_no_substitutes_good(self, consumer_for_cache_tests):
        agent = consumer_for_cache_tests
        # 'treat' has no substitutes defined in the fixture
        assert "treat" not in agent._prepared_substitutes
        
        # Ensure apply_substitutions doesn't break for 'treat'
        agent.base_needs = {"treat": 5.0}
        agent.consumption_received = defaultdict(float, {"treat": 2.0})
        agent.apply_substitutions()
        assert math.isclose(agent.consumption_actual["treat"], 2.0) # No change as no subs

    def test_prepared_substitutes_handles_empty_substitution_pairs(self, dummy_model):
        agent = ConsumerAgent(
            unique_id="no_sub_agent",
            model=dummy_model,
            region_id=1,
            base_needs={"food": 10},
            substitution_pairs={} # Explicitly empty
        )
        assert agent._prepared_substitutes == {}

        # Ensure apply_substitutions works
        agent.consumption_received = defaultdict(float, {"food": 5.0})
        agent.apply_substitutions()
        assert math.isclose(agent.consumption_actual["food"], 5.0)


class TestLearningSubstitution:
    def test_apply_substitution_learning(self, dummy_model):
        agent = ConsumerAgent(
            unique_id="learn_sub",
            model=dummy_model,
            region_id=1,
            base_needs={"A": 1.0, "B": 1.0},
            preference_weights={"A": 0.5, "B": 0.5},
        )

        agent.last_order = np.array([10.0, 0.0])
        planned = np.array([10.0, 0.0])
        received = np.array([5.0, 0.0])
        prices = np.array([2.0, 1.0])
        demand = agent._apply_substitution(planned, received, prices, agent.sub_cfg)

        # Substitution matrix should learn to replace A with B
        assert agent.S[0, 1] > 0
        # Demand should shift some quantity to good B
        assert demand[1] > 0

