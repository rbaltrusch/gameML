# -*- coding: utf-8 -*-
"""Contains the game logic for the snake game"""

import math
import random
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


class Simulation:
    def __init__(
        self, points_for_surviving: int, points_for_eating: int, seed: Optional[int]
    ):
        self.points_for_surviving = points_for_surviving
        self.points_for_eating = points_for_eating
        self.seed = seed
        self.moves_since_eating = 0
        self.paused = False
        self.terminated = False
        self.score = 0

    def run(
        self,
        model: Model,
        squares: List[Square],
        squares_dict: Dict[Position, Square],
        max_moves_without_eating: int,
    ) -> int:
        random.seed(self.seed)
        initial_entities = [square.entity for square in squares]
        snake = [self._init_snake(squares_dict)]

        over = False
        self.moves_since_eating = 0
        self.paused = False
        self.terminated = False
        self.score = 0
        apple = init_apple(squares)
        while not self.terminated and not over:
            if not self.paused:
                move = model.compute(values=encode_squares(squares))
                over, apple = self._update_game(
                    squares, squares_dict, snake, apple, move, max_moves_without_eating
                )
            self.render(squares)

        # restore squares state
        for square, entity in zip(squares, initial_entities):
            square.entity = entity
        return max(0, self.score)

    def _update_game(  # pylint: disable=too-many-arguments
        self,
        squares: List[Square],
        squares_dict: Dict[Position, Square],
        snake: List[Square],
        apple: Square,
        move: Position,
        max_moves_without_eating: int,
    ) -> Tuple[bool, Square]:
        if not snake:
            return (True, apple)

        pos = snake[0].grid_position
        new_square = squares_dict[pos.x + move.x, pos.y + move.y]  # type: ignore
        if new_square.entity in [Entities.WALL, Entities.SNAKE]:
            return (True, apple)

        snake[0].entity = Entities.SNAKE
        if new_square.entity == Entities.EMPTY:
            self.moves_since_eating += 1
            normalized_distance = (
                math.dist(snake[0].grid_position, apple.grid_position)
                / len(squares) ** 0.5
            )
            self.score += self.points_for_surviving * (1 - normalized_distance)
            last_square = snake.pop()
            last_square.entity = Entities.EMPTY
        else:
            self.moves_since_eating = 0
            for square in squares:
                if square.entity == Entities.APPLE:
                    square.entity = Entities.EMPTY
                    break
            apple = init_apple(squares)
            self.score += self.points_for_eating

        if self.moves_since_eating > max_moves_without_eating:
            return (True, apple)

        snake.insert(0, new_square)
        new_square.entity = Entities.SNAKE_HEAD
        return (False, apple)

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

    def run(
        self,
        model: Model,
        squares: List[Square],
        squares_dict: Dict[Position, Square],
        max_moves_without_eating: int,
    ) -> int:
        return self.simulation.run(
            model, squares, squares_dict, max_moves_without_eating
        )

    def render(self, squares: List[Square]):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self.simulation.paused = False

        self.screen.fill((0, 0, 0))
        for square in squares:
            square.render(self.screen)
        pygame.display.flip()

        self.clock.tick(5)
