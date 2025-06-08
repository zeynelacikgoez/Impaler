import cProfile
import pstats
from impaler.core.config import create_default_config
from impaler.core.model import EconomicModel


def run():
    cfg = create_default_config(simulation_steps=25)
    model = EconomicModel(cfg)
    model.run_simulation()


if __name__ == "__main__":
    cProfile.run("run()", "impaler_profile.prof")
    p = pstats.Stats("impaler_profile.prof")
    p.strip_dirs().sort_stats("cumulative").print_stats(30)
