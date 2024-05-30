import numpy as np
from typing import Tuple, Optional, Union, List
from collections import deque
import random
from vector import Vector
from math import sqrt

class Vision:
    __slots__ = ('dist_to_wall', 'dist_to_apple', 'dist_to_self')
    def __init__(self, dist_to_wall: float, dist_to_apple: float, dist_to_self: float):
        self.dist_to_wall = dist_to_wall
        self.dist_to_apple = dist_to_apple
        self.dist_to_self = dist_to_self

class DrawableVision:
    __slots__ = ('wall_location', 'apple_location', 'self_location')
    def __init__(self, wall_location: Vector, apple_location: Optional[Vector] = None, self_location: Optional[Vector] = None):
        self.wall_location = wall_location
        self.apple_location = apple_location
        self.self_location = self_location

class Slope:
    def __init__(self, rise: int, run: int):
        self.rise = rise
        self.run = run

class SnakeGame:
    def __init__(self, xsize: int = 30, ysize: int = 30, scale: int = 15, max_steps: int = 40000, controller=None):
        self.grid = Vector(xsize, ysize)
        self.scale = scale
        self.max_steps = max_steps  # Maximum allowed steps
        self.current_step = 0  # Initialize step counter
        initial_direction = random.choice([Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0)])
        self.snake = Snake(game=self, initial_direction=initial_direction)
        self.food = Food(game=self)
        self.controller = controller  # Add this line to accept a controller

    def run(self):
        running = True
        while running:
            next_move = self.controller.update()
            if next_move:
                self.snake.v = next_move
                self.snake.move()
                self.current_step += 1  # Increment step counter
                if self.current_step >= self.max_steps:
                    running = False  # Terminate the game if max step count reached
                    print("Terminated: Max steps reached")
            if not self.snake.p.within(self.grid):
                running = False
                print("Terminated: Snake hit the wall")
            if self.snake.cross_own_tail:
                running = False
                print("Terminated: Snake crossed its own tail")
            if self.snake.p == self.food.p:
                self.snake.add_score()
                self.food = Food(game=self)
                print(f"Step: {self.current_step}, Position: {self.snake.p}")

class Food:
    def __init__(self, game: SnakeGame):
        self.game = game
        self.p = Vector.random_within(self.game.grid)

class Snake:
    def __init__(self, *, game: SnakeGame, initial_direction: Vector = None):
        self.game = game
        self.score = 0
        self.v = initial_direction if initial_direction else random.choice([Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0)])
        self.body = deque([Vector.random_within(self.game.grid)])
        self._vision_type = VISION_8
        self._vision: List[Vision] = [None] * len(self._vision_type)
        self._drawable_vision: List[DrawableVision] = [None] * len(self._vision_type)

    def move(self):
        self.p = self.p + self.v

    def distance_to_food(self):
        max_distance = sqrt(self.game.grid.x ** 2 + self.game.grid.y ** 2)
        distance = sqrt((self.game.food.p.x - self.p.x) ** 2 + (self.game.food.p.y - self.p.y) ** 2)
        return distance / max_distance

    def vision(self):
        vision = []
        for slope in self._vision_type:  # You can choose VISION_16 or VISION_4 based on your preference
            vision_info, _ = self.look_in_direction(slope)
            vision.append(vision_info.dist_to_wall)
            vision.append(vision_info.dist_to_apple)
            vision.append(vision_info.dist_to_self)
        return vision

    def look_in_direction(self, slope: Slope) -> Tuple[Vision, DrawableVision]:
        dist_to_wall = None
        dist_to_apple = np.inf
        dist_to_self = np.inf

        wall_location = None
        apple_location = None
        self_location = None

        position = self.body[0].copy()
        distance = 1.0
        total_distance = 0.0

        # Can't start by looking at yourself
        position.x += slope.run
        position.y += slope.rise
        total_distance += distance
        body_found = False  # Only need to find the first occurrence since it's the closest
        food_found = False  # Although there is only one food, stop looking once you find it

        # Keep going until the position is out of bounds
        while self._within_wall(position):
            if not body_found and self._is_body_location(position):
                dist_to_self = total_distance
                self_location = position.copy()
                body_found = True
            if not food_found and self._is_apple_location(position):
                dist_to_apple = total_distance
                apple_location = position.copy()
                food_found = True

            wall_location = position
            position.x += slope.run
            position.y += slope.rise
            total_distance += distance
        assert(total_distance != 0.0)

        dist_to_wall = 1.0 / total_distance
        dist_to_apple = 1.0 if dist_to_apple != np.inf else 0.0
        dist_to_self = 1.0 if dist_to_self != np.inf else 0.0

        vision = Vision(dist_to_wall, dist_to_apple, dist_to_self)
        drawable_vision = DrawableVision(wall_location, apple_location, self_location)
        return (vision, drawable_vision)

    def _within_wall(self, position: Vector) -> bool:
        return position.x >= 0 and position.y >= 0 and position.x < self.game.grid.x and position.y < self.game.grid.y

    def _is_body_location(self, position: Vector) -> bool:
        return position in self.body

    def _is_apple_location(self, position: Vector) -> bool:
        return position == self.game.food.p

    def add_score(self):
        self.score += 1
        tail = self.body.pop()
        self.body.append(tail)
        self.body.append(tail)

    @property
    def cross_own_tail(self):
        try:
            self.body.index(self.p, 1)
            return True
        except ValueError:
            return False

    @property
    def p(self):
        return self.body[0]

    @p.setter
    def p(self, value):
        self.body.appendleft(value)
        self.body.pop()

VISION_16 = (
    Slope(-1, 0), Slope(-2, 1),  Slope(-1, 1),  Slope(-1, 2),
    Slope(0, 1),  Slope(1, 2),   Slope(1, 1),   Slope(2, 1),
    Slope(1, 0),  Slope(2, -1),  Slope(1, -1),  Slope(1, -2),
    Slope(0, -1), Slope(-1, -2), Slope(-1, -1), Slope(-2, -1)
)

VISION_8 = tuple([VISION_16[i] for i in range(len(VISION_16)) if i % 2 == 0])
VISION_4 = tuple([VISION_16[i] for i in range(len(VISION_16)) if i % 4 == 0])
