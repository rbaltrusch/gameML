# -*- coding: utf-8 -*-
"""Machine learning algorithm for the game snake, visualised with pygame."""

import random

import numpy

from src.entity import Entities, init_squares
from src.model import Model, Moves
from src.simulation import Simulation, VisualSimulation
from src.util import Position

# pylint: disable=missing-class-docstring, missing-docstring, no-member, invalid-name

SCREEN_DIMENSIONS = Position(500, 500)
SQUARE_SIZE = 50
MAX_MOVES_WITHOUT_EATING = 15
WEIGHT_DIVERGENCE = 0.01
NUMBER_OF_MODELS = 100
MODELS_SELECTED_PER_GENERATION = 50
CLONES = 1
GENERATIONS = 10
LAYERS = 3
LAYER_SIZE = 100
POINTS_FOR_SURVIVING = 1
POINTS_FOR_EATING = 20


def back_propagation(models, squares, squares_dict):
    """Back propagation draft"""
    for model in models:
        layer = random.choice(model.layers)
        values = list(zip(*[random.choices(list(range(x)), k=5) for x in layer.shape]))
        previous_values = [layer[x, y] for x, y in values]
        for x, y in values:
            layer[x, y] *= numpy.random.rand() * 2 - 1
        rating = Simulation(POINTS_FOR_SURVIVING, POINTS_FOR_EATING).run(
            model, squares, squares_dict, MAX_MOVES_WITHOUT_EATING
        )
        if rating < model.rating:
            for (x, y), previous_value in zip(values, previous_values):
                layer[x, y] = previous_value


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

    simulation = Simulation(POINTS_FOR_SURVIVING, POINTS_FOR_EATING)
    for i in range(GENERATIONS):
        for model in models:
            model.rating = simulation.run(
                model, squares, squares_dict, MAX_MOVES_WITHOUT_EATING
            )

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
        print(f"Step {i}: Best model rating", models[0].rating)

    for _ in range(5):
        VisualSimulation(
            SCREEN_DIMENSIONS, POINTS_FOR_SURVIVING, POINTS_FOR_EATING
        ).run(models[0], squares, squares_dict, MAX_MOVES_WITHOUT_EATING)


if __name__ == "__main__":
    main()
