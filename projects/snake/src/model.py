# -*- coding: utf-8 -*-
"""Contains the neural network implementation"""

import enum
from typing import List

import numpy

from src.util import Position

# pylint: disable=missing-class-docstring, missing-docstring, no-member, invalid-name


class Moves(enum.Enum):
    LEFT, RIGHT, UP, DOWN = range(1, 5)


class Model:
    def __init__(self, inputs, outputs, layers, layer_size):
        self.rating = 0
        self.layers: List[numpy.ndarray] = (
            [numpy.random.normal(size=(inputs, layer_size))]
            + [numpy.random.normal(size=(layer_size, layer_size))] * layers
            + [numpy.random.normal(size=(layer_size, outputs))]
        )

    def compute(self, values: numpy.ndarray) -> Position:
        for layer in self.layers:
            values.shape = (1, layer.shape[0])
            values = numpy.dot(values, layer)
        moves = decode_move(values)
        return lookup_position(moves)

    def clone(self, weight_divergence: float):
        model = self.__class__(0, 0, 0, 0)
        model.layers = [
            layer + (numpy.random.normal(size=layer.shape) * 2 - 1) * weight_divergence
            for layer in self.layers
        ]
        return model


def decode_move(values: numpy.ndarray) -> Moves:
    move, _ = max(zip(Moves, values[0]), key=lambda x: x[1])
    return move


def lookup_position(move: Moves) -> Position:
    offsets = {
        Moves.LEFT: (-1, 0),
        Moves.RIGHT: (1, 0),
        Moves.UP: (0, -1),
        Moves.DOWN: (0, 1),
    }
    return Position(*offsets[move])
