from collections import deque
from ga_controller import GAController
from vector import Vector
import random
from math import sqrt


class SnakeGame:
    def __init__(self, xsize: int=30, ysize: int=30, scale: int=15, max_steps: int=2000, controller=None):
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
            if not self.snake.p.within(self.grid):
                running = False
            if self.snake.cross_own_tail:
                running = False
            if self.snake.p == self.food.p:
                self.snake.add_score()
                self.food = Food(game=self)
        #print(f'{message} ... Score: {self.snake.score}')

class Food:
    def __init__(self, game: SnakeGame):
        self.game = game
        self.p = Vector.random_within(self.game.grid)

class Snake:
    def __init__(self, *, game: SnakeGame, initial_direction: Vector = None):
        self.game = game
        self.score = 0
        if initial_direction:
            self.v = initial_direction
        else:
            self.v = random.choice([Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0)])  # Randomize initial direction
        self.body = deque()
        self.body.append(Vector.random_within(self.game.grid))


    def move(self):
        self.p = self.p + self.v

    def distance_to_food(self):
        # Calculate Euclidean distance between snake's head and food
        return sqrt((self.game.food.p.x - self.p.x) ** 2 + (self.game.food.p.y - self.p.y) ** 2)

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

    def add_score(self):
        self.score += 1
        tail = self.body.pop()
        self.body.append(tail)
        self.body.append(tail)

    def debug(self):
        print('===')
        for i in self.body:
            print(str(i))
