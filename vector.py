import random

class Vector:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def within(self, grid):
        return 0 <= self.x < grid.x and 0 <= self.y < grid.y

    @staticmethod
    def random_within(grid):
        return Vector(random.randint(0, grid.x - 1), random.randint(0, grid.y - 1))

    def copy(self):
        return Vector(self.x, self.y)

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
