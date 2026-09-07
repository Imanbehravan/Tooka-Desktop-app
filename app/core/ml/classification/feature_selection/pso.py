import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score


def pso_feature_selection(
    X,
    y,
    model,
    n_particles=20,
    iterations=30,
    inertia=0.7,
    cognitive=1.5,
    social=1.5,
    alpha=0.02,
    random_state=42,
):
    """
    Particle Swarm Optimization (PSO) for feature selection.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target values.

    model : sklearn estimator
        Model used to evaluate feature subsets.

    n_particles : int, default=20
        Number of particles in the swarm.

    iterations : int, default=30
        Number of optimization iterations.

    inertia : float, default=0.7
        Inertia weight.

    cognitive : float, default=1.5
        Cognitive coefficient.

    social : float, default=1.5
        Social coefficient.

    alpha : float, default=0.02
        Penalty applied to the number of selected features.

    random_state : int, default=42
        Random seed.

    Returns
    -------
    list
        Names of selected features.
    """

    if X.shape[1] == 0:
        raise ValueError("No features available for PSO.")

    if X.shape[1] == 1:
        return list(X.columns)

    rng = np.random.default_rng(random_state)

    n_features = X.shape[1]

    # ---------------------------------------------------------
    # Evaluation Function
    # ---------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    def evaluate_particle(position):
        """
        Evaluate one particle.

        position:
            Binary vector representing selected features.
        """

        selected_indices = np.where(position == 1)[0]

        # Prevent empty feature subsets
        if len(selected_indices) == 0:
            return 0.0

        X_selected = X.iloc[:, selected_indices]

        estimator = clone(model)

        scores = cross_val_score(
            estimator,
            X_selected,
            y,
            cv=cv,
            scoring="accuracy",
        )

        accuracy = scores.mean()

        feature_ratio = (
            len(selected_indices) / n_features
        )

        fitness = accuracy - alpha * feature_ratio

        return fitness

    # ---------------------------------------------------------
    # Initial Population
    # ---------------------------------------------------------

    positions = rng.integers(
        0,
        2,
        size=(n_particles, n_features),
    )

    # Make sure every particle selects at least one feature
    for i in range(n_particles):
        if not np.any(positions[i]):
            index = rng.integers(0, n_features)
            positions[i, index] = 1

    # ---------------------------------------------------------
    # Initial Velocities
    # ---------------------------------------------------------

    velocities = rng.uniform(
        -1,
        1,
        size=(n_particles, n_features),
    )

    # ---------------------------------------------------------
    # Personal Best
    # ---------------------------------------------------------

    personal_best_positions = positions.copy()

    personal_best_scores = np.array(
        [
            evaluate_particle(position)
            for position in positions
        ]
    )

    # ---------------------------------------------------------
    # Global Best
    # ---------------------------------------------------------

    best_particle_index = np.argmax(
        personal_best_scores
    )

    global_best_position = (
        personal_best_positions[
            best_particle_index
        ].copy()
    )

    global_best_score = (
        personal_best_scores[
            best_particle_index
        ]
    )

    # ---------------------------------------------------------
    # Optimization Loop
    # ---------------------------------------------------------

    for _ in range(iterations):

        for i in range(n_particles):

            r1 = rng.random(n_features)
            r2 = rng.random(n_features)

            # -------------------------------------------------
            # Velocity Update
            # -------------------------------------------------

            velocities[i] = (
                inertia * velocities[i]
                + cognitive
                * r1
                * (
                    personal_best_positions[i]
                    - positions[i]
                )
                + social
                * r2
                * (
                    global_best_position
                    - positions[i]
                )
            )

            # -------------------------------------------------
            # Sigmoid Transformation
            # -------------------------------------------------

            sigmoid = 1 / (
                1 + np.exp(-velocities[i])
            )

            # -------------------------------------------------
            # Position Update
            # -------------------------------------------------

            positions[i] = (
                rng.random(n_features)
                < sigmoid
            ).astype(int)

            # Prevent empty subset
            if not np.any(positions[i]):
                index = rng.integers(
                    0,
                    n_features,
                )

                positions[i, index] = 1

            # -------------------------------------------------
            # Evaluate
            # -------------------------------------------------

            score = evaluate_particle(
                positions[i]
            )

            # -------------------------------------------------
            # Update Personal Best
            # -------------------------------------------------

            if score > personal_best_scores[i]:

                personal_best_scores[i] = score

                personal_best_positions[i] = (
                    positions[i].copy()
                )

            # -------------------------------------------------
            # Update Global Best
            # -------------------------------------------------

            if score > global_best_score:

                global_best_score = score

                global_best_position = (
                    positions[i].copy()
                )

    # ---------------------------------------------------------
    # Selected Features
    # ---------------------------------------------------------

    selected_indices = np.where(
        global_best_position == 1
    )[0]

    selected_features = list(
        X.columns[selected_indices]
    )

    return selected_features