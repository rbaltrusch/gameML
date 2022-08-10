# -*- coding: utf-8 -*-
"""Contains the game logic for the snake game"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy
import pygame

from src.entity import Entities, Square, init_apple, init_square
from src.model import Model, NpArray
from src.util import Position

# pylint: disable=missing-class-docstring, missing-docstring, no-member, invalid-name


def encode_squares(squares: List[Square]) -> NpArray:
    return numpy.array([x for square in squares for x in encode_entity(square.entity)])


def encode_entity(entity: Entities) -> List[int]:
    return [int(i == entity.value.number) for i in range(5)]


@dataclass(frozen=True)
class SimulationStep:
    next_state: NpArray
    reward: float
    done: bool


@dataclass
class Simulation:  # pylint: disable=too-many-instance-attributes

    squares: List[Square]
    squares_dict: Dict[Position, Square]
    points_for_surviving: int
    points_for_eating: int
    max_moves_without_eating: int
    seed: Optional[int]

    def __post_init__(self):
        random.seed(self.seed)
        self.moves_since_eating = 0
        self.paused = False
        self.terminated = False
        self.score = 0
        self.initial_entities = [square.entity for square in self.squares]
        self.snake = [self._init_snake(self.squares_dict)]
        self.apple = init_apple(self.squares)

    def init(self) -> NpArray:
        """Initialises simulation"""
        self.__post_init__()
        return self._encode()

    def reset(self):
        """restore squares state"""
        for square, entity in zip(self.squares, self.initial_entities):
            square.entity = entity

    def run(self, model: Model) -> int:
        self.init()
        while not self.terminated:
            if not self.paused:
                move = model.compute(values=self._encode())
                self._update_game(move)
            self.render(self.squares)
        self.reset()
        return max(0, self.score)

    def step(self, move: Position) -> SimulationStep:
        previous_score = self.score
        self._update_game(move)
        reward = self.score - previous_score
        return SimulationStep(
            next_state=self._encode(), reward=reward, done=self.terminated
        )

    def _encode(self):
        return encode_squares(self.squares)

    def _update_game(self, move: Position):  # pylint: disable=too-many-arguments
        if not self.snake:
            self.terminated = True
            return

        pos = self.snake[0].grid_position
        new_square = self.squares_dict[pos.x + move.x, pos.y + move.y]  # type: ignore
        if new_square.entity in [Entities.WALL, Entities.SNAKE]:
            self.terminated = True
            return

        self.snake[0].entity = Entities.SNAKE
        if new_square.entity == Entities.EMPTY:
            self.moves_since_eating += 1
            normalized_distance = (
                math.dist(self.snake[0].grid_position, self.apple.grid_position)
                / len(self.squares) ** 0.5
            )
            self.score += self.points_for_surviving * (1 - normalized_distance)
            last_square = self.snake.pop()
            last_square.entity = Entities.EMPTY
        else:
            self.moves_since_eating = 0
            for square in self.squares:
                if square.entity == Entities.APPLE:
                    square.entity = Entities.EMPTY
                    break
            self.apple = init_apple(self.squares)
            self.score += self.points_for_eating

        if self.moves_since_eating > self.max_moves_without_eating:
            self.terminated = True
            return

        self.snake.insert(0, new_square)
        new_square.entity = Entities.SNAKE_HEAD

    def render(self, squares: List[Square]):
        pass

    @staticmethod
    def _init_snake(squares_dict: Dict[Position, Square]) -> Square:
        return init_square([squares_dict[3, 3]], Entities.SNAKE_HEAD)  # type: ignore


class VisualSimulation:
    def __init__(self, screen_size: Tuple[int, int], simulation: Simulation):
        self.simulation = simulation
        self.simulation.paused = True
        self.simulation.render = self.render
        pygame.init()
        self.screen: pygame.surface.Surface = (  # pylint: disable=c-extension-no-member
            pygame.display.set_mode(screen_size)
        )
        self.clock = pygame.time.Clock()

    def run(self, model: Model) -> int:
        return self.simulation.run(model)

    def render(self, squares: List[Square]):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self.simulation.paused = False

        self.screen.fill((0, 0, 0))
        for square in squares:
            square.render(self.screen)
        pygame.display.flip()

        self.clock.tick(5)
