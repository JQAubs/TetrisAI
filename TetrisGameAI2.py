import pygame
import random

BOARD_HEIGHT = 24
BOARD_WIDTH = 10

FZoom = 30
FPad = 10
fontHeight = 15
pygame.init()
size = (BOARD_WIDTH*FZoom+2*FPad, BOARD_HEIGHT*FZoom+2*FPad+(fontHeight+5))
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Tetris")

colors = [
    (10,10,10),
    (255,4,3),
    (107,29,224),
    (33,188,245),
    (31,237,48),
    (255,231,33),
]

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

BOARD_HEIGHT = 24
BOARD_WIDTH = 10

class Figure:
    x = 0
    y = 0

    figures = [
        [[1, 5, 9, 13], [4, 5, 6, 7]],
        [[4, 5, 9, 10], [2, 6, 5, 9]],
        [[6, 7, 9, 10], [1, 5, 6, 10]],
        [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        [[1, 2, 5, 6]],
    ]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.randint(0, len(self.figures) - 1)
        self.color = random.randint(1, len(colors) - 1)
        self.rotation = 0

    def image(self):
        return self.figures[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])

class TetrisAI:

    score = 0
    state = "start"
    field = []
    x = FPad
    y = FPad+fontHeight+5
    zoom = FZoom
    step = 0
    pressDown = False

    def __init__(self, frames_per_second=150):
        self.level = 0
        self.fps = frames_per_second
        self.height = BOARD_HEIGHT
        self.width = BOARD_WIDTH
        self.field = []
        self.score = 0
        self.state = "start"
        self.new_figure()
        self.linesCleared = 0
        self.totalLinesCleared = 0
        for i in range(self.height):
            new_line = []
            for j in range(self.width):
                new_line.append(0)
            self.field.append(new_line)

    def new_figure(self):
        self.figure = Figure(3, 0)

    def intersects(self):
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    if i + self.figure.y > self.height - 1 or \
                            j + self.figure.x > self.width - 1 or \
                            j + self.figure.x < 0 or \
                            self.field[i + self.figure.y][j + self.figure.x] > 0:
                        intersection = True
        return intersection

    def getLevel(self):
        if self.level < 9:
            if self.linesCleared == (self.level*10+10) or self.linesCleared == max(100, self.level*10-50):
                self.level += 1
                self.linesCleared = 0

        elif self.level in range(9,15):
            if self.linesCleared == 100:
                self.level += 1
                self.linesCleared = 0

        elif self.level in range(15, 24):
            if self.linesCleared == (self.level-15)*10 + 100:
                self.level += 1
                self.linesCleared = 0

        else:
            if self.linesCleared == 200:
                self.level += 1
                self.linesCleared = 0

    def getScore(self, lines):
        if lines == 0:
            return 0
        if lines == 1:
            return 40*(self.level+1)
        if lines == 2:
            return 100*(self.level+1)
        if lines == 3:
            return 300*(self.level+1)
        if lines == 4:
            return 1200*(self.level+1)
        return 0

    def break_lines(self):
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]
        self.linesCleared += lines
        self.totalLinesCleared += lines
        self.getLevel()
        self.score += self.getScore(lines)

    def test_space(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1

    def go_space(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze()

    def test_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            self.freeze()

    def freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.break_lines()
        self.new_figure()
        self.goalPos = None
        if self.intersects():
            self.state = "gameover"

    def go_side(self, dx):
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x
            return False
        return True

    def rotate(self):
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = old_rotation

    def reset(self):
        self.linesCleared = 0
        self.totalLinesCleared = 0
        self.level = 0
        self.field = []
        self.score = 0
        self.step = 0
        self.state = "start"
        for i in range(self.height):
            new_line = []
            for j in range(self.width):
                new_line.append(0)
            self.field.append(new_line)

    def redrawBoard(self):
        screen.fill(WHITE)
        for i in range(self.height):
            for j in range(self.width):
                pygame.draw.rect(screen, GRAY, [self.x + self.zoom * j, self.y + self.zoom * i, self.zoom, self.zoom], 1)
                if self.field[i][j] > 0:
                    pygame.draw.rect(screen, colors[self.field[i][j]],
                                     [self.x + self.zoom * j + 1, self.y + self.zoom * i + 1, self.zoom - 2, self.zoom - 1])

        if self.figure is not None:
            for i in range(4):
                for j in range(4):
                    p = i * 4 + j
                    if p in self.figure.image():
                        pygame.draw.rect(screen, colors[self.figure.color],
                                         [self.x + self.zoom * (j + self.figure.x) + 1,
                                          self.y + self.zoom * (i + self.figure.y) + 1,
                                          self.zoom - 2, self.zoom - 2])

        font = pygame.font.SysFont('Calibri', fontHeight, True, False)
        info = font.render("  level: "+str(self.level)+"  Score: "+str(self.score)+"  Lines Cleared: "+str(self.totalLinesCleared), True, BLACK)

        screen.blit(info, [5,7])

    def updateUI(self):
        font = pygame.font.SysFont('Calibri', 25, True, False)
        font1 = pygame.font.SysFont('Calibri', 65, True, False)
        text = font.render("Score: " + str(self.score), True, BLACK)
        text_game_over = font1.render("Game Over", True, (255, 125, 0))
        text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))
        #pygame.blit()
# [rotate, left, right, down, space]

    def game_step(self):
        if self.state == 'gameover':
            return True, self.score
        events = pygame.event.get()
        self.step += 1
        clock = pygame.time.Clock()
        reward = 0
        if self.figure is None:
            self.new_figure()

        self.go_down()
        self.redrawBoard()
        self.updateUI()
        pygame.display.flip()
        clock.tick(self.fps)

        gameOver = True if self.state == 'gameover' else False
        return gameOver, self.score

    def game_step2(self):
        if self.state == 'gameover':
            return True, self.score, self.totalLinesCleared
        events = pygame.event.get()
        self.step += 1
        clock = pygame.time.Clock()
        reward = 0
        if self.figure is None:
            self.new_figure()

        self.go_down()
        self.redrawBoard()
        self.updateUI()
        pygame.display.flip()
        clock.tick(self.fps)

        gameOver = True if self.state == 'gameover' else False

        return gameOver, self.score, self.totalLinesCleared

    def game_step_play(self):
        if self.state == 'gameover':
            return True, self.score, self.totalLinesCleared
        events = pygame.event.get()
        self.step += 1
        clock = pygame.time.Clock()
        reward = 0
        if self.figure is None:
            self.new_figure()

        self.go_down()
        self.redrawBoard()
        self.updateUI()
        pygame.display.flip()
        clock.tick(self.fps)

        gameOver = True if self.state == 'gameover' else False

        return gameOver, self.score, self.totalLinesCleared, self.level
