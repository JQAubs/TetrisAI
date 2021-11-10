import pygame
import random


BOARD_HEIGHT = 24
BOARD_WIDTH = 10

FZoom = 70
FPad = 10
fontHeight = 30
pygame.init()
size = (BOARD_WIDTH*FZoom+2*FPad, BOARD_HEIGHT*FZoom+2*FPad+(fontHeight+5))
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Tetris")

colors = [
    (20,20,20),
    (240,85,85),
    (248,138,252),
    (111,73,213),
    (27,180,255),
    (30,195,124),
    (222,101,31),
]

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

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

    def copy(self):
        newFig = Figure(self.x,self.y)
        newFig.type = self.type
        newFig.color = self.color
        newFig.rotation = self.rotation
        return newFig

    def image(self):
        return self.figures[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])

    def rotateBack(self):
        self.rotation = (self.rotation - 1) % len(self.figures[self.type])

class TetrisAI:
    level = 1
    score = 0
    state = "start"
    field = []
    x = FPad
    y = FPad+fontHeight+5
    zoom = FZoom
    step = 0
    pressDown = False

    def __init__(self):
        self.fps = 200
        self.height = BOARD_HEIGHT
        self.width = BOARD_WIDTH
        self.field = []
        self.score = 0
        self.state = "start"
        self.new_figure()
        for i in range(self.height):
            new_line = []
            for j in range(self.width):
                new_line.append(0)
            self.field.append(new_line)

    def copy(self):
        newAgent = TetrisAI()
        newAgent.field = self.field.copy()
        newAgent.figure = self.figure.copy()
        newAgent.score = self.score
        return newAgent

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
        self.score += lines ** 2

    def go_space(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze()

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            self.freeze()

    def go_up(self):
        self.figure.y -= 1

    def freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.break_lines()
        self.new_figure()
        if self.intersects():
            self.state = "gameover"

    def go_side(self, dx):
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x

    def rotate(self):
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = old_rotation

    def reset(self):
        self.level = 1
        self.field = []
        self.score = 0
        self.step = 0
        self.state = "start"
        for i in range(self.height):
            new_line = []
            for j in range(self.width):
                new_line.append(0)
            self.field.append(new_line)

    def redrawBoard(self, gen):
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
        info = font.render('Game #: '+str(gen)+"   Score: "+str(self.score), True, BLACK)

        screen.blit(info, [5,7])

    def updateUI(self):
        font = pygame.font.SysFont('Calibri', 25, True, False)
        font1 = pygame.font.SysFont('Calibri', 65, True, False)
        text = font.render("Score: " + str(self.score), True, BLACK)
        text_game_over = font1.render("Game Over", True, (255, 125, 0))
        text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))
# [rotate, left, right, down, space]
    def doAction(self, action):
        if action[0] == 1:
            self.rotate()
        if action[1] == 1:
            self.go_side(-1)
        if action[2] == 1:
            self.go_side(1)
        if action[3] == 1:
            self.go_down()
        if action[4] == 1:
            self.go_space()

    def game_step(self, AI_move, prevScore, generation):
        pygame.event.get()
        self.step += 1
        clock = pygame.time.Clock()
        reward = 0


        if self.figure is None:
            self.new_figure()

        #if self.step % self.fps == 0:
        self.go_down()

        self.doAction(AI_move)

        self.redrawBoard(generation)

        self.updateUI()
        pygame.display.flip()
        clock.tick(self.fps)

        gameOver = True if self.state == 'gameover' else False
        if self.score != prevScore:
            reward += 5 * (self.score - prevScore)

        return gameOver, self.score, reward
