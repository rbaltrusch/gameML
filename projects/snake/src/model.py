# -*- coding: utf-8 -*-
"""Contains the neural network implementation"""

import enum
from typing import List, Optional, Tuple

import numpy

from src.util import Position

# pylint: disable=missing-class-docstring, missing-docstring, no-member, invalid-name


def sigmoid(x: numpy.ndarray) -> numpy.ndarray:
    return 1 / (1 + (numpy.exp(-x)))


class Moves(enum.Enum):
    LEFT, RIGHT, UP, DOWN = range(1, 5)


class Model:
    def __init__(self, inputs, outputs, layers, layer_size):
        self.rating = 0
        self.layers: List[numpy.ndarray] = (
            self._construct_matrices(shape=(inputs, layer_size), num=1)
            + self._construct_matrices(shape=(layer_size, layer_size), num=layers)
            + self._construct_matrices(shape=(layer_size, outputs), num=1)
        )
        self.biases: List[Optional[numpy.ndarray]] = [
            None,
            *self._construct_matrices(shape=(1, layer_size), num=layers),
            None,
        ]

    @staticmethod
    def _construct_matrices(shape: Tuple[int, int], *, num: int) -> numpy.ndarray:
        return [numpy.random.normal(size=shape) for _ in range(num)]

    def compute(self, values: numpy.ndarray) -> Position:
        for (layer, bias) in zip(self.layers, self.biases):
            values.shape = (1, layer.shape[0])
            values = numpy.dot(values, layer)
            if bias is not None:
                values += bias
            values = sigmoid(values)
        moves = decode_move(values)
        return lookup_position(moves)

    def clone(self, weight_divergence: float):
        model = self.__class__(0, 0, 0, 0)
        model.layers = [
            layer + (numpy.random.normal(size=layer.shape) * 2 - 1) * weight_divergence
            for layer in self.layers
        ]
        model.biases = [
            bias + (numpy.random.normal(size=bias.shape) * 2 - 1) * weight_divergence
            if bias is not None
            else bias
            for bias in self.biases
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
