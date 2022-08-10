# -*- coding: utf-8 -*-
"""Contains the game logic for the snake game"""

from typing import Dict, List

import numpy
import pygame

from src.entity import Entities, Square, init_apple, init_square
from src.model import Model
from src.util import Position

# pylint: disable=missing-class-docstring, missing-docstring, no-member, invalid-name


def encode_squares(squares: List[Square]) -> numpy.ndarray:
    return numpy.array([x for square in squares for x in encode_entity(square.entity)])


def encode_entity(entity: Entities) -> List[int]:
    return [int(i == entity.value.number) for i in range(5)]


class Simulation:
    def __init__(self, points_for_surviving: int, points_for_eating: int):
        self.points_for_surviving = points_for_surviving
        self.points_for_eating = points_for_eating
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
        initial_entities = [square.entity for square in squares]
        snake = [self.init_snake(squares_dict)]

        over = False
        self.moves_since_eating = 0
        self.paused = False
        self.terminated = False
        self.score = 0
        init_apple(squares)
        # init_square([squares_dict[1, 1]], Entities.APPLE)
        while not self.terminated and not over:
            if not self.paused:
                move = model.compute(values=encode_squares(squares))
                over = self.update_game(
                    squares, squares_dict, snake, move, max_moves_without_eating
                )
            self.render(squares)

        # restore squares state
        for square, entity in zip(squares, initial_entities):
            square.entity = entity
        return max(0, self.score)

    def update_game(  # pylint: disable=too-many-arguments
        self,
        squares: List[Square],
        squares_dict: Dict[Position, Square],
        snake: List[Square],
        move: Position,
        max_moves_without_eating: int,
    ) -> bool:
        if not snake:
            return True

        pos = snake[0].grid_position
        new_square = squares_dict[pos.x + move.x, pos.y + move.y]
        if new_square.entity in [Entities.WALL, Entities.SNAKE]:
            return True

        snake[0].entity = Entities.SNAKE
        if new_square.entity == Entities.EMPTY:
            self.moves_since_eating += 1
            last_square = snake.pop()
            last_square.entity = Entities.EMPTY
            self.score += self.points_for_surviving
        else:
            self.moves_since_eating = 0
            for square in squares:
                if square.entity == Entities.APPLE:
                    square.entity = Entities.EMPTY
                    break
            init_apple(squares)
            self.score += self.points_for_eating

        if self.moves_since_eating > max_moves_without_eating:
            return True

        snake.insert(0, new_square)
        new_square.entity = Entities.SNAKE_HEAD
        return (False, apple)

    def render(self, squares):
        pass

    @staticmethod
    def init_snake(squares_dict: Dict[Position, Square]) -> Square:
        return init_square([squares_dict[3, 3]], Entities.SNAKE_HEAD)


class VisualSimulation(Simulation):
    def __init__(self, screen_size, points_for_surviving: int, points_for_eating: int):
        super().__init__(points_for_surviving, points_for_eating)
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        self.clock = pygame.time.Clock()
        self.paused = True

    def render(self, squares):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.terminated = False
            elif event.type == pygame.KEYDOWN:
                self.paused = False

        self.screen.fill((0, 0, 0))
        for square in squares:
            square.render(self.screen)
        pygame.display.flip()

        self.clock.tick(5)
