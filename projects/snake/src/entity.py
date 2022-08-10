# -*- coding: utf-8 -*-
"""Contains entity position handling and entity rendering logic"""

import enum
import random
from dataclasses import dataclass
from typing import List

import pygame

from src.util import Position

# pylint: disable=missing-class-docstring, missing-docstring, no-member, invalid-name


class OutOfSpaceException(Exception):
    pass


@dataclass(frozen=True)
class Colour:
    red: int = 0
    green: int = 0
    blue: int = 0

    def __iter__(self):
        yield self.red
        yield self.green
        yield self.blue


@dataclass
class Entity:
    number: int
    colour: Colour = Colour()


class Entities(enum.Enum):
    APPLE = Entity(number=4, colour=Colour(red=255))
    EMPTY = Entity(number=3, colour=Colour(blue=50))
    SNAKE = Entity(number=2, colour=Colour(green=150))
    SNAKE_HEAD = Entity(number=1, colour=Colour(green=255, blue=50, red=50))
    WALL = Entity(number=0, colour=Colour(red=50, green=50, blue=50))


@dataclass
class Square:
    x: int
    y: int
    square_size: int
    entity: Entities = Entities.EMPTY

    def render(self, screen: pygame.Surface):
        width = 1
        pygame.draw.rect(
            screen,
            tuple(self.entity.value.colour),
            (*self.position, self.square_size, self.square_size),
            width,
        )

    @property
    def position(self) -> Position:
        return Position(self.x * self.square_size, self.y * self.square_size)

    @property
    def grid_position(self) -> Position:
        return Position(self.x, self.y)


def init_squares(width, height, square_size):
    squares = [Square(x, y, square_size) for x in range(width) for y in range(height)]

    # place wall at edges of map
    for square in squares:
        if not (
            0 < square.grid_position.x < width - 1
            and 0 < square.grid_position.y < height - 1
        ):
            square.entity = Entities.WALL
    return squares


def get_empty_squares(squares: List[Square]) -> List[Square]:
    return [x for x in squares if x.entity == Entities.EMPTY]


def init_square(squares: List[Square], entity: Entities) -> Square:
    empty_squares = get_empty_squares(squares)
    if not empty_squares:
        raise OutOfSpaceException
    square = random.choice(empty_squares)
    square.entity = entity
    return square


def init_apple(squares: List[Square]):
    return init_square(squares, Entities.APPLE)
