"""Utility functions for the genetic algorithm."""
from __future__ import annotations

import random
from typing import Dict, Any, Iterable, List, Tuple


def _random_value(spec: Dict[str, Any]) -> Any:
    """Generate a random value for a gene specification."""
    if spec["type"] == "float":
        lo, hi = spec["range"]
        return random.uniform(lo, hi)
    if spec["type"] == "int":
        lo, hi = spec["range"]
        return random.randint(int(lo), int(hi))
    if spec["type"] == "categorical":
        return random.choice(spec["values"])
    if spec["type"] == "bool":
        return random.choice([True, False])
    raise ValueError(f"Unknown type {spec['type']}")


def generate_random_genome(parameter_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Create a random genome from the given parameter space."""
    genome = {}
    for gene, spec in parameter_space.items():
        genome[gene] = _random_value(spec)
    return genome


def create_initial_population(size: int, parameter_space: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate the initial population of genomes."""
    return [generate_random_genome(parameter_space) for _ in range(size)]


def select_parents(population_with_fitness: Iterable[Tuple[int, Dict[str, Any], float, Any]], num_parents_mating: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Select two parents from the population for crossover."""
    sorted_pop = sorted(population_with_fitness, key=lambda x: x[2], reverse=True)
    candidates = sorted_pop[:num_parents_mating]
    weights = [max(fit, 0.0) + 1e-6 for _, _, fit, _ in candidates]
    genomes = [g for _, g, _, _ in candidates]
    parent1 = random.choices(genomes, weights=weights, k=1)[0]
    parent2 = random.choices(genomes, weights=weights, k=1)[0]
    return parent1, parent2


def crossover(parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
    """Combine two genomes to produce an offspring genome."""
    child = {}
    genes = set(parent1) | set(parent2)
    for gene in genes:
        if gene in parent1 and gene in parent2:
            child[gene] = random.choice([parent1[gene], parent2[gene]])
        elif gene in parent1:
            child[gene] = parent1[gene]
        else:
            child[gene] = parent2[gene]
    return child


def mutate(genome: Dict[str, Any], parameter_space: Dict[str, Dict[str, Any]], mutation_rate: float) -> Dict[str, Any]:
    """Mutate a genome with the given mutation rate."""
    mutated = dict(genome)
    for gene, spec in parameter_space.items():
        if random.random() < mutation_rate:
            mutated[gene] = _random_value(spec)
    return mutated

