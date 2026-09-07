import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _initialize_population(
    n_features: int,
    population_size: int,
):
    """
    Create the initial GA population.

    Each chromosome is a binary vector:
        1 -> feature selected
        0 -> feature ignored
    """

    population = []

    for _ in range(population_size):

        chromosome = np.random.choice(
            [0, 1],
            size=n_features,
            p=[0.4, 0.6],
        ).tolist()

        # Never allow an empty chromosome.
        if sum(chromosome) == 0:
            chromosome[
                np.random.randint(0, n_features)
            ] = 1

        population.append(chromosome)

    return population


def _fitness(
    chromosome,
    X,
    y,
    model,
    cv,
    alpha=0.02,
):
    """
    Calculate chromosome fitness.

    Fitness consists of:

        classification error
        +
        feature-count penalty

    Lower fitness is better.
    """

    # No selected feature
    if sum(chromosome) == 0:
        return 1.0

    mask = np.asarray(
        chromosome,
        dtype=bool,
    )

    X_selected = X.loc[:, mask]

    estimator = clone(model)

    scores = cross_val_score(
        estimator,
        X_selected,
        y,
        cv=cv,
        scoring="accuracy",
    )

    error = 1.0 - np.mean(scores)

    feature_penalty = (
        alpha
        * np.sum(chromosome)
        / len(chromosome)
    )

    return error + feature_penalty


def _tournament_selection(
    population,
    scores,
    tournament_size=3,
):
    """
    Select one chromosome using tournament selection.
    """

    tournament_size = min(
        tournament_size,
        len(population),
    )

    indices = np.random.choice(
        len(population),
        tournament_size,
        replace=False,
    )

    winner = indices[
        np.argmin(
            [scores[i] for i in indices]
        )
    ]

    return population[winner].copy()


def _crossover(
    parent1,
    parent2,
    crossover_rate=0.7,
):
    """
    Single-point crossover.
    """

    if len(parent1) < 2:
        return parent1.copy(), parent2.copy()

    if np.random.rand() >= crossover_rate:
        return parent1.copy(), parent2.copy()

    point = np.random.randint(
        1,
        len(parent1),
    )

    child1 = (
        parent1[:point]
        + parent2[point:]
    )

    child2 = (
        parent2[:point]
        + parent1[point:]
    )

    return child1, child2


def _mutate(
    chromosome,
    mutation_rate,
):
    """
    Flip chromosome genes according to mutation rate.
    """

    mutated = chromosome.copy()

    for i in range(len(mutated)):

        if np.random.rand() < mutation_rate:
            mutated[i] = 1 - mutated[i]

    # Never allow an empty chromosome.
    if sum(mutated) == 0:
        mutated[
            np.random.randint(
                0,
                len(mutated),
            )
        ] = 1

    return mutated


def ga_feature_selection(
    X,
    y,
    model,
    output_path=None,
    population_size=30,
    generations=50,
    crossover_rate=0.7,
    mutation_rate=0.1,
    alpha=0.02,
    random_state=42,
):
    """
    Genetic Algorithm feature selection.

    Parameters
    ----------
    X : pandas.DataFrame
        Training features.

    y : pandas.Series
        Training target.

    model :
        sklearn-compatible estimator.

    output_path : str or Path, optional
        Path for convergence graph.

    population_size : int
        Number of chromosomes.

    generations : int
        Maximum number of generations.

    crossover_rate : float
        Probability of crossover.

    mutation_rate : float
        Initial mutation probability.

    alpha : float
        Feature-count penalty.

    random_state : int
        Random seed.

    Returns
    -------
    selected_features : list[str]
        Names of selected features.
    """

    if X.empty:
        raise ValueError(
            "Cannot perform GA feature selection "
            "on an empty dataset."
        )

    if X.shape[1] == 1:
        return [X.columns[0]]

    if population_size < 2:
        raise ValueError(
            "population_size must be at least 2."
        )

    if generations < 1:
        raise ValueError(
            "generations must be at least 1."
        )

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    np.random.seed(random_state)

    # --------------------------------------------------
    # Cross Validation
    # --------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    # --------------------------------------------------
    # Initial Population
    # --------------------------------------------------

    population = _initialize_population(
        n_features=X.shape[1],
        population_size=population_size,
    )

    best_solution = None
    best_fitness = np.inf

    best_curve = []
    average_curve = []

    # --------------------------------------------------
    # GA Loop
    # --------------------------------------------------

    for generation in range(generations):

        scores = [
            _fitness(
                chromosome,
                X,
                y,
                model,
                cv,
                alpha=alpha,
            )
            for chromosome in population
        ]

        generation_best_index = int(
            np.argmin(scores)
        )

        generation_best = scores[
            generation_best_index
        ]

        generation_average = float(
            np.mean(scores)
        )

        # ----------------------------------------------
        # Global Best
        # ----------------------------------------------

        if generation_best < best_fitness:

            best_fitness = generation_best

            best_solution = population[
                generation_best_index
            ].copy()

        best_curve.append(
            best_fitness
        )

        average_curve.append(
            generation_average
        )

        print(
            f"GA Generation {generation + 1:3d} | "
            f"Best: {generation_best:.6f} | "
            f"Global: {best_fitness:.6f} | "
            f"Average: {generation_average:.6f}"
        )

        # ----------------------------------------------
        # Elitism
        # ----------------------------------------------

        elite_count = max(
            1,
            population_size // 5,
        )

        elite_indices = np.argsort(
            scores
        )[:elite_count]

        new_population = [
            population[i].copy()
            for i in elite_indices
        ]

        # ----------------------------------------------
        # Adaptive Mutation
        # ----------------------------------------------

        current_mutation_rate = max(
            0.02,
            mutation_rate
            * (
                1
                - generation / generations
            ),
        )

        # ----------------------------------------------
        # Generate Offspring
        # ----------------------------------------------

        while len(new_population) < population_size:

            parent1 = _tournament_selection(
                population,
                scores,
            )

            parent2 = _tournament_selection(
                population,
                scores,
            )

            child1, child2 = _crossover(
                parent1,
                parent2,
                crossover_rate=crossover_rate,
            )

            child1 = _mutate(
                child1,
                current_mutation_rate,
            )

            child2 = _mutate(
                child2,
                current_mutation_rate,
            )

            new_population.append(
                child1
            )

            if len(new_population) < population_size:
                new_population.append(
                    child2
                )

        population = new_population

    # --------------------------------------------------
    # Safety Check
    # --------------------------------------------------

    if best_solution is None:
        raise RuntimeError(
            "GA failed to find a feature subset."
        )

    # --------------------------------------------------
    # Selected Features
    # --------------------------------------------------

    selected_features = X.columns[
        np.asarray(
            best_solution,
            dtype=bool,
        )
    ].tolist()

    print("=" * 60)
    print("GA Feature Selection Completed")
    print(
        f"Selected Features: {selected_features}"
    )
    print(
        f"Selected: "
        f"{len(selected_features)} / {X.shape[1]}"
    )
    print(
        f"Best Fitness: {best_fitness:.6f}"
    )
    print("=" * 60)

    # --------------------------------------------------
    # Convergence Plot
    # --------------------------------------------------

    if output_path is not None:
        _save_convergence_plot(
            best_curve,
            average_curve,
            output_path,
        )

    return selected_features


def _save_convergence_plot(
    best_curve,
    average_curve,
    output_path,
):
    """
    Save GA convergence graph.

    Matplotlib is imported lazily so that the core
    algorithm remains usable without plotting.
    """

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(10, 6)
    )

    iterations = range(
        1,
        len(best_curve) + 1,
    )

    plt.plot(
        iterations,
        best_curve,
        label="Best Fitness",
    )

    plt.plot(
        iterations,
        average_curve,
        label="Average Fitness",
        linestyle="--",
    )

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(
        "GA Feature Selection Convergence"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()