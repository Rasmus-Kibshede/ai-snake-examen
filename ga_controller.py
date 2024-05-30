import random
from vector import Vector
import pygame
from game_controller import GameController
import numpy as np


class GAController(GameController):
    def __init__(self, game, model, display=True):
        self.display = display
        self.game = game
        self.model = model
        self.action_space = (Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0))
        self.current_direction = self.action_space[0]  # Initialize with any direction (e.g., up)
        self.screen = None
        self.direction_history = []
        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode((game.grid.x * game.scale, game.grid.y * game.scale))
            self.clock = pygame.time.Clock()
            self.color_snake_head = (0, 255, 0)
            self.color_food = (255, 0, 0)

    def __del__(self):
        if self.display and pygame.get_init():
            pygame.quit()

    def update(self) -> Vector:
        # observation space
        dn = self.game.snake.p.y / self.game.grid.y  # Normalize by grid height
        de = (self.game.grid.x - self.game.snake.p.x) / self.game.grid.x  # Normalize by grid width
        ds = (self.game.grid.y - self.game.snake.p.y) / self.game.grid.y  # Normalize by grid height
        dw = self.game.snake.p.x / self.game.grid.x  # Normalize by grid width

        # Normalized delta food x and y (food's relative coordinates)
        dfx = (self.game.food.p.x - self.game.snake.p.x) / self.game.grid.x
        dfy = (self.game.food.p.y - self.game.snake.p.y) / self.game.grid.y

        # Additional observations for nearby obstacles (e.g., snake's body segments)
        left_obstacle = int(self.game.snake.p.x == 0)
        right_obstacle = int(self.game.snake.p.x == self.game.grid.x - 1)
        up_obstacle = int(self.game.snake.p.y == 0)
        down_obstacle = int(self.game.snake.p.y == self.game.grid.y - 1)

        # Include observations about the presence of obstacles in each direction
        obs = [dn, de, ds, dw, dfx, dfy, left_obstacle, right_obstacle, up_obstacle, down_obstacle]

        if len(self.game.snake.body) >= 2:
            tail_left = int(self.game.snake.p - self.game.snake.body[1] == Vector(-1, 0))
            tail_right = int(self.game.snake.p - self.game.snake.body[1] == Vector(1, 0))
            tail_up = int(self.game.snake.p - self.game.snake.body[1] == Vector(0, -1))
            tail_down = int(self.game.snake.p - self.game.snake.body[1] == Vector(0, 1))
        else:
            # Default values if snake's body has fewer than two segments
            tail_left, tail_right, tail_up, tail_down = 0, 0, 0, 0

        # Include observations about the presence of the tail in each direction
        obs += [tail_left, tail_right, tail_up, tail_down]

        # Possible routes to food
        route_x_first = [Vector(dfx, 0), Vector(0, dfy)]
        route_y_first = [Vector(0, dfy), Vector(dfx, 0)]

        # Evaluate safety and distance for both routes
        def evaluate_route(route):
            safety = 0
            for move in route:
                new_pos = self.game.snake.p + move
                if new_pos in self.game.snake.body or not new_pos.within(self.game.grid):
                    safety -= 1
            return safety, np.sqrt((new_pos.x - self.game.food.p.x) ** 2 + (new_pos.y - self.game.food.p.y) ** 2)

        safety_x, dist_x = evaluate_route(route_x_first)
        safety_y, dist_y = evaluate_route(route_y_first)

        # Choose the best route
        if safety_x > safety_y or (safety_x == safety_y and dist_x < dist_y):
            chosen_route = route_x_first
        else:
            chosen_route = route_y_first

        # Determine the next move
        for move in chosen_route:
            new_pos = self.game.snake.p + move
            if new_pos not in self.game.snake.body and new_pos.within(self.game.grid):
                next_move = move
                break
            else:
                # If no safe move is found, choose a random valid move
                next_move = random.choice(self.action_space)

        # Prevent moving in the opposite direction
        if next_move == -self.current_direction:
            next_move = random.choice(self.action_space)
            if next_move == -self.current_direction:
                next_move = self.action_space[self.model.action(obs)]
        else:
            next_move = self.action_space[self.model.action(obs)]

        if self.display and self.screen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.screen = None
                    return None

            if self.screen:
                self.screen.fill('black')
                for i, p in enumerate(self.game.snake.body):
                    pygame.draw.rect(self.screen, (0, max(128, 255 - i * 12), 0), self.block(p))
                pygame.draw.rect(self.screen, self.color_food, self.block(self.game.food.p))
                pygame.display.flip()
                self.clock.tick(40)

        # Update the current direction
        self.current_direction = next_move

        return next_move

    def block(self, obj):
        return (obj.x * self.game.scale, obj.y * self.game.scale, self.game.scale, self.game.scale)