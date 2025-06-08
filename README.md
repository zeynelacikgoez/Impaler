# Impaler

**Impaler** is a highly advanced, agent-based simulation platform designed to generate and evaluate novel planning and economic models. It focuses on exploring complex, non-market-based coordination mechanisms using concepts from the **Viable System Model (VSM)**, Multi-Agent Reinforcement Learning (MARL), and evolutionary algorithms.

## Core Concepts

Impaler integrates several cutting-edge paradigms to enable deep insights into complex economic systems:

* **Agent-Based Simulation:** At its core, the system simulates autonomous agents (`Producer`, `Consumer`, `Resource`) with dynamic needs, learning capabilities, and complex production processes.

* **Viable System Model (VSM):** The simulation is structured around the principles of VSM—a cybernetic management model. Different system layers (S2, S3, S4, S5) take on specific roles, from operational coordination to strategic policy-making.

* **Dynamic Crisis Simulation:** The `CrisisManager` can simulate planned or random crises—such as resource shortages or natural disasters—to test the system’s resilience and adaptability.

* **Evolutionary Model Search:** Impaler leverages genetic algorithms to explore the vast parameter space of economic models. By running simulations in parallel with different "genomes" (configurations) and evaluating them with a fitness function (`welfare`, `sustainability`, `stability`), the system can evolve more robust and efficient model configurations.

* **Multi-Agent Reinforcement Learning (MARL):** Agents, especially the `ProducerAgent`, can operate in RL mode to learn optimal behavioral strategies (e.g., for investment decisions). The framework uses **Ray RLlib** and provides a **PettingZoo**-compatible environment for standardized training.

* **Flexible Pydantic Configuration:** The entire simulation is governed by a strongly-typed, validated configuration structure (`core/config.py`), allowing for easy customization and reproducibility of experiments.

## Architecture Overview

The project follows a modular architecture:

* **/core:** Contains the central `EconomicModel`, `StageManager`, `CrisisManager`, `DataCollector`, and the Pydantic-based `SimulationConfig`.
* **/agents:** Defines the behavior logic for economic agents (`ProducerAgent`, `ConsumerAgent`, `ResourceAgent`).
* **/vsm:** Implements the different layers of the Viable System Model (`System2Coordinator`, `System3Manager`, `System4Planner`).
* **/governance:** Houses System 5 (`GovernmentAgent`), responsible for overarching policy and goal-setting.
* **/evolution:** Provides tools for evolutionary model search, including the fitness function and parameter space definition.
* **/marl:** Contains the codebase for training Multi-Agent Reinforcement Learning models, including PettingZoo environments (`envs`) and training scripts.
* **/utils:** Offers mathematical and economic utility functions.

## Installation

### Prerequisites

* Python 3.7 or higher
* Some features (e.g., optimization) benefit from optional packages like SciPy.

### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/zeynelacikgoez/impaler.git
   cd Impaler-main
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   All required packages—including `ray[rllib]`, `pettingzoo`, `pydantic`, and `numpy`—are listed in the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Impaler can be run in several modes, depending on the research objective.

### 1. Running a Single Simulation

This mode is useful for testing and analyzing a specific economic model using a fixed configuration.

```python
# Example script: run_single_sim.py

from impaler.core.model import EconomicModel
from impaler.core.config import create_default_config

if __name__ == "__main__":
    # Create and optionally customize the default config
    config = create_default_config()
    config.simulation_steps = 50
    config.logging_config.log_level = "INFO"
    config.agent_populations["default_producers"].count = 10
    config.agent_populations["default_consumers"].count = 40
    
    # Initialize and run the model
    model = EconomicModel(config=config)
    print("Starting single simulation...")
    model.run_simulation()
    
    # Save results
    model.save_results(path="single_run_results.json", format="json")
    print("Simulation completed. Results saved to 'single_run_results.json'.")
```

### 2. Evolutionary Model Search

This is the most powerful mode, used to find robust and high-performing model parameters. The `run_evolution.py` script executes many simulations in parallel and evolves the best configurations over several generations.

**Start the evolutionary search:**

```bash
python run_evolution.py
```

* Genetic algorithm settings (population size, mutation rate, etc.) are specified in `run_evolution.py`.
* The searchable parameter space (`PARAMETER_SPACE`) is defined in `evolution/parameter_space.py`.
* Model fitness is evaluated by `evolution/fitness.py`.
* Results from the best genomes of each generation are saved in the `evolution_results/` directory.

### 3. Training RL Agents (MARL)

The `marl` directory is set up for training agents using Reinforcement Learning. It utilizes Ray RLlib and a custom PettingZoo environment.

**Start a training run (e.g., using QMIX):**

```bash
python marl/train.py --config marl/config/qmix.yaml
```

* Add `--wandb` to enable logging with [Weights & Biases](https://wandb.ai/) (requires `wandb login`).
* The RL algorithm configuration file (e.g., `qmix.yaml`) controls hyperparameters like learning rate, network architecture, and environment settings.
* The reward function is weighted via the `marl/config/reward.yml` configuration.

## Configuration

The simulation is fully controlled via the Pydantic-based `SimulationConfig` in `core/config.py`. This enables robust, validated, and easily extensible configuration through Python dictionaries or JSON files.

* **Structure:** The config is organized into logical blocks such as `regional_config`, `environment_config`, `planning_priorities`, and `agent_populations`.
* **Customization:** You can generate a default config with `create_default_config()` and modify it programmatically, or load a complete config from a dictionary or JSON.
* **Agent Definition:** Agents can be defined as populations (`agent_populations`) with randomized parameters or as specific instances (`specific_agents`).

## Testing

The project includes a comprehensive test suite using `pytest`.

**Run all tests:**

```bash
pytest
```

This executes all unit and integration tests in the `tests/` directory to ensure the correctness of core components.

## License

This project is licensed under the MIT License. For more details, refer to the `LICENSE` file.