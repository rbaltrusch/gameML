# -*- coding: utf-8 -*-
"""Machine learning algorithm for the game snake, visualised with pygame."""

from src.entity import Entities, init_squares
from src.model import Model, Moves
from src.simulation import Simulation, VisualSimulation, encode_squares_2d
from src.tf_model import TfModel
from src.util import Position

# pylint: disable=missing-class-docstring, missing-docstring, no-member, invalid-name

SCREEN_DIMENSIONS = Position(500, 500)
SQUARE_SIZE = 50
MAX_MOVES_WITHOUT_EATING = 15
WEIGHT_DIVERGENCE = 0.05
NUMBER_OF_MODELS = 400
MODELS_SELECTED_PER_GENERATION = 200
NEW_MODELS_PER_GENERATION = 50
CLONES = 1
GENERATIONS = 10
LAYERS = 3
LAYER_SIZE = 100
POINTS_FOR_SURVIVING = 1
POINTS_FOR_EATING = 100
SEED = 1


def main_tf():
    width = SCREEN_DIMENSIONS.x // SQUARE_SIZE
    height = SCREEN_DIMENSIONS.y // SQUARE_SIZE
    squares = init_squares(width, height, square_size=SQUARE_SIZE)
    squares_dict = {square.grid_position: square for square in squares}

    simulation = Simulation(
        squares,
        squares_dict,
        POINTS_FOR_SURVIVING,
        POINTS_FOR_EATING,
        MAX_MOVES_WITHOUT_EATING,
        SEED,
        encoder=encode_squares_2d,
    )
    tf_model = TfModel(
        num_actions=len(Moves), input_size=(width, height, len(Entities))
    )
    tf_model.init_models()

    try:
        tf_model.train_model(simulation)
    except KeyboardInterrupt:
        pass

    tf_model.save(filepath="tf_model.json")
    simulation.reset()
    visual_simulation = VisualSimulation(SCREEN_DIMENSIONS, simulation)
    visual_simulation.run(model=tf_model)


def main():
    """Main function"""
    if not NUMBER_OF_MODELS or not GENERATIONS:
        return

    def init_model():
        return Model(
            inputs=len(squares) * len(Entities),
            outputs=len(Moves),
            layers=LAYERS,
            layer_size=LAYER_SIZE,
        )

    squares = init_squares(
        width=SCREEN_DIMENSIONS.x // SQUARE_SIZE,
        height=SCREEN_DIMENSIONS.y // SQUARE_SIZE,
        square_size=SQUARE_SIZE,
    )
    squares_dict = {square.grid_position: square for square in squares}

    models = [init_model() for _ in range(NUMBER_OF_MODELS)]

    simulation = Simulation(
        squares,
        squares_dict,
        POINTS_FOR_SURVIVING,
        POINTS_FOR_EATING,
        MAX_MOVES_WITHOUT_EATING,
        SEED,
    )

    for i in range(GENERATIONS):
        for model in models:
            model.rating = simulation.run(model)

        models = sorted(models, key=lambda x: x.rating, reverse=True)[
            :MODELS_SELECTED_PER_GENERATION
        ]

        models.extend(
            [
                model.clone(weight_divergence=WEIGHT_DIVERGENCE)
                for model in models
                for _ in range(CLONES)
            ]
        )

        # to avoid population degeneration on a local optimum
        models.extend(init_model() for _ in range(NEW_MODELS_PER_GENERATION))
        print(f"Step {i}: Best model rating", models[0].rating)

    # show trained seed, then with a random seed
    input("Press ENTER-key to start visual simulation...")
    for seed in [SEED, None]:
        simulation.seed = seed
        VisualSimulation(SCREEN_DIMENSIONS, simulation).run(models[0])


if __name__ == "__main__":
    main_tf()
