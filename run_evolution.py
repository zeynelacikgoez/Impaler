import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Dict, Any, List

from impaler.core.model import EconomicModel
from impaler.evolution.parameter_space import PARAMETER_SPACE, generate_config_from_genome
from impaler.evolution.fitness import calculate_fitness
from impaler.evolution.utils import (
    create_initial_population,
    select_parents,
    crossover,
    mutate,
)

# --- GA configuration ---
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
NUM_PARENTS_MATING = 20
MUTATION_RATE = 0.1
SIMULATION_STEPS_PER_RUN = 200

FITNESS_WEIGHTS = {
    "welfare": 1.5,
    "sustainability": 1.0,
    "stability": 0.5,
    "resilience": 0.75,
}

# Directory where intermediate results are stored
RESULTS_DIR = "evolution_results"



def run_simulation_for_genome(genome_id: int, genome: Dict[str, Any]) -> Tuple[int, Dict[str, Any], float, Dict[str, Any] | None]:
    """Run a single simulation for a genome and return its fitness.

    Parameters
    ----------
    genome_id:
        Index of the genome in the population.
    genome:
        Dictionary representing the genome.
    """
    try:
        print(f"Starte Simulation f\u00fcr Genom {genome_id}...")
        config = generate_config_from_genome(genome)
        config.simulation_steps = SIMULATION_STEPS_PER_RUN
        model = EconomicModel(config=config)
        model.run_simulation()
        results = model.get_results()
        fitness = calculate_fitness(results, FITNESS_WEIGHTS)
        print(f"Genom {genome_id} beendet. Fitness: {fitness:.4f}")
        return genome_id, genome, fitness, results
    except Exception as exc:  # noqa: BLE001
        print(f"!!! FEHLER bei Genom {genome_id}: {exc}")
        return genome_id, genome, -1000.0, None


def main() -> None:
    print("Starte evolution\u00e4re Suche nach optimalen Wirtschaftsmodellen...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    population: List[Dict[str, Any]] = create_initial_population(POPULATION_SIZE, PARAMETER_SPACE)
    max_workers = os.cpu_count() or 1

    population_with_fitness: List[Tuple[int, Dict[str, Any], float, Dict[str, Any] | None]] = []
    for generation in range(NUM_GENERATIONS):
        print(f"\n===== Generation {generation + 1}/{NUM_GENERATIONS} =====")
        population_with_fitness.clear()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_simulation_for_genome, i, genome): (i, genome)
                for i, genome in enumerate(population)
            }
            for future in futures:
                try:
                    population_with_fitness.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    idx, genome = futures[future]
                    print(f"Worker crashed for genome {idx}: {exc}")
                    population_with_fitness.append((idx, genome, -1000.0, None))

        population_with_fitness.sort(key=lambda x: x[2], reverse=True)
        best_genome_this_gen = population_with_fitness[0]
        print(f"Beste Fitness in Generation {generation + 1}: {best_genome_this_gen[2]:.4f}")
        result_path = os.path.join(RESULTS_DIR, f"best_genome_gen_{generation + 1}.json")
        with open(result_path, "w") as fh:
            json.dump({"genome": best_genome_this_gen[1], "fitness": best_genome_this_gen[2]}, fh, indent=2)

        num_elites = POPULATION_SIZE // 10
        elites = [g[1] for g in population_with_fitness[:num_elites]]
        next_population: List[Dict[str, Any]] = list(elites)

        num_offspring = POPULATION_SIZE - num_elites
        for _ in range(num_offspring):
            parent1, parent2 = select_parents(population_with_fitness, NUM_PARENTS_MATING)
            child = crossover(parent1, parent2)
            child = mutate(child, PARAMETER_SPACE, MUTATION_RATE)
            next_population.append(child)

        population = next_population

    print("\nEvolution abgeschlossen!")
    best_overall = population_with_fitness[0]
    print(f"Bestes gefundenes Genom (Fitness: {best_overall[2]:.4f}):")
    print(json.dumps(best_overall[1], indent=2))
    result_path = os.path.join(RESULTS_DIR, "best_genome_overall.json")
    with open(result_path, "w") as fh:
        json.dump({"genome": best_overall[1], "fitness": best_overall[2]}, fh, indent=2)


if __name__ == "__main__":
    main()
