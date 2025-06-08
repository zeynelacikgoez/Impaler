import pytest

pytest.importorskip("pytest_benchmark")

from impaler.core.config import create_default_config
from impaler.core.model import EconomicModel

@pytest.mark.benchmark(group="simulation_step")
def test_simple_simulation_step(benchmark):
    cfg = create_default_config(simulation_steps=1, performance_profile="fast_prototype")
    model = EconomicModel(cfg)
    benchmark(model.step)
