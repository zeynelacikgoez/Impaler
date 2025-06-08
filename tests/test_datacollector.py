import pytest
import random
from unittest.mock import MagicMock

# Assuming DataCollector and EconomicModel are accessible for import
from impaler.core.datacollector import DataCollector
from impaler.core.model import EconomicModel
from impaler.core.config import SimulationConfig

class MockAgent:
    """A simple mock agent for testing data collection."""
    def __init__(self, unique_id: str):
        self.unique_id = unique_id

    def generate_report(self):
        return {"id": self.unique_id, "data_point": random.randint(1, 100)}

@pytest.fixture
def mock_economic_model():
    """Provides a mocked EconomicModel with configurable agent lists."""
    model = MagicMock(spec=EconomicModel)
    model.config = SimulationConfig() # Attach a default config
    model.current_step = 0
    # Ensure all agent lists that DataCollector iterates over are present
    model.producers = []
    model.consumers = []
    model.infrastructure_agents = []
    model.logger = MagicMock()
    return model

class TestDataCollectorSampling:
    """Tests the agent_sampling_rate feature of the DataCollector."""

    def test_collector_initialization_clips_sampling_rate(self):
        """Tests that agent_sampling_rate is clipped to [0, 1] during init."""
        # Rate > 1.0
        dc1 = DataCollector(config={"agent_sampling_rate": 1.5})
        assert dc1.agent_sampling_rate == 1.0

        # Rate < 0.0
        dc2 = DataCollector(config={"agent_sampling_rate": -0.5})
        assert dc2.agent_sampling_rate == 0.0

        # Valid rate
        dc3 = DataCollector(config={"agent_sampling_rate": 0.7})
        assert dc3.agent_sampling_rate == 0.7

    def test_collect_agent_data_sampling_full(self, mock_economic_model):
        """
        Tests full data collection when agent_sampling_rate is 1.0.
        """
        # 1. Setup
        dc = DataCollector(config={"agent_sampling_rate": 1.0, "store_agent_data": True})
        mock_agents = [MockAgent(f"agent_{i}") for i in range(10)]
        mock_economic_model.producers = mock_agents  # Put all agents in one list for simplicity

        # 2. Action
        dc._collect_agent_data(mock_economic_model)

        # 3. Expected
        assert len(dc.agent_data) == 10
        for agent in mock_agents:
            assert agent.unique_id in dc.agent_data
            assert len(dc.agent_data[agent.unique_id]) == 1

    def test_collect_agent_data_sampling_none(self, mock_economic_model):
        """
        Tests that at least one agent is sampled when rate is 0.0 (if agents exist).
        """
        # 1. Setup
        dc = DataCollector(config={"agent_sampling_rate": 0.0, "store_agent_data": True})
        mock_agents = [MockAgent(f"agent_{i}") for i in range(10)]
        mock_economic_model.producers = mock_agents

        # 2. Action
        dc._collect_agent_data(mock_economic_model)

        # 3. Expected
        # The logic is `max(1, int(len * rate))`, so for rate=0.0 it becomes `max(1, 0)` -> 1
        assert len(dc.agent_data) == 1

    def test_collect_agent_data_sampling_half_is_deterministic(self, mock_economic_model):
        """
        Tests partial sampling (e.g., 0.5) is deterministic with a seed.
        """
        # 1. Setup
        dc = DataCollector(config={"agent_sampling_rate": 0.5, "store_agent_data": True})
        mock_agents = [MockAgent(f"agent_{i}") for i in range(10)]
        mock_economic_model.producers = mock_agents

        # 2. Action
        random.seed(42) # Set seed for predictable `random.sample`
        dc._collect_agent_data(mock_economic_model)

        # 3. Expected
        # With rate=0.5, we expect int(10 * 0.5) = 5 agents
        assert len(dc.agent_data) == 5
        # Determine the exact agents that should have been sampled with this seed
        random.seed(42)
        expected_agent_ids = {agent.unique_id for agent in random.sample(mock_agents, 5)}
        assert set(dc.agent_data.keys()) == expected_agent_ids

    def test_collect_agent_data_sampling_empty_agent_list(self, mock_economic_model):
        """
        Tests that no errors occur and no data is collected if agent lists are empty.
        """
        # 1. Setup
        dc = DataCollector(config={"agent_sampling_rate": 0.5, "store_agent_data": True})
        # mock_economic_model has empty agent lists by default

        # 2. Action
        dc._collect_agent_data(mock_economic_model)

        # 3. Expected
        assert len(dc.agent_data) == 0

    def test_collect_agent_data_store_agent_data_is_false(self, mock_economic_model):
        """
        Tests that no agent data is collected if store_agent_data is False,
        by checking the public `collect_data` method.
        """
        # 1. Setup
        # Set collection frequency to 1 to ensure it runs
        dc = DataCollector(config={"store_agent_data": False, "collection_frequency": 1})
        mock_agents = [MockAgent(f"agent_{i}") for i in range(10)]
        mock_economic_model.producers = mock_agents
        # Mock the model's current_step
        mock_economic_model.current_step = 0

        # 2. Action
        dc.collect_data(mock_economic_model) # Call the public method

        # 3. Expected
        # The guard in `collect_data` should prevent `_collect_agent_data` from being called
        assert len(dc.agent_data) == 0
        # Model data should still be collected
        assert len(dc.model_data) == 1