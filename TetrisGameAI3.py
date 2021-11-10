import random
import numpy as np
from TreeGenotype2 import treeGenotype

BOARD_HEIGHT = 24
BOARD_WIDTH = 10

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
    step = 0

    def __init__(self, frames_per_second=150):
        self.level = 0
        self.fps = frames_per_second
        self.height = BOARD_HEIGHT
        self.width = BOARD_WIDTH
        self.field = []
        self.score = 0
        self.state = "start"
        self.new_figure()
        self.goalPos = None
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

    def getState(self):
        board = self.field
        piece = self.figure
        newBoard = [[y for y in x] for x in board]
        arr = [[0 for i in range(4)] for y in range(4)]
        for x in range(len(arr)):
            for y in range(len(arr)):
                if  x*len(arr[0])+y in piece.image():
                    newBoard[piece.y+x][piece.x+y] = 1
        #print(board)
        return np.array(newBoard, dtype=int)

    #finds the highest predicted score location based on the fitness function and parameters in genome
    def find_best_move(self, strat):
        if self.state == 'gameover':
            return (0,0,0)
        #print(type(strat))
        if isinstance(strat, list):
            rType = 1
        elif isinstance(strat, treeGenotype):
            rType = 2
        else:
            rType = 0
        #print(rType)
        fig = self.figure
        ogFig = (fig.x, fig.y, fig.rotation)
        self.test_space()
        fits = [(fig.x, fig.y, fig.rotation, fig.type, rewardFunc(self.getState(), rType, strat))]
        fig.y = ogFig[1]
        bestFit = -10000
        for _ in range(len(fig.figures[fig.type])):

            self.figure.x = ogFig[1]

            for x in range(9):
                self.test_space()
                fit = rewardFunc(self.getState(), rType, strat)
                if fit > bestFit:
                    bestFit = fit
                    fits.append((fig.x, fig.y, fig.rotation, fig.type, fit))
                fig.y = ogFig[1]
                #if x % 4 == 3:
                    #self.test_down()
                complete = self.go_side(-1)
                if not complete:
                    break

            self.figure.x = ogFig[1]

            for _ in range(9):
                self.test_space()
                fit = rewardFunc(self.getState(), rType, strat)
                if fit > bestFit:
                    bestFit = fit
                    fits.append((fig.x, fig.y, fig.rotation, fig.type, fit))
                fig.y = ogFig[1]
                #if x % 4 == 3:
                    #self.test_down()
                complete = self.go_side(1)
                if not complete:
                    break
                #print('tested pos')
            self.figure.rotate()

        self.figure.x = ogFig[0]
        self.figure.y = ogFig[1]
        self.figure.rotation = ogFig[2]

        return  list(max(fits, key = lambda x : x[4]))

    #responsible for performing the best predicted action
    def makeMove(self, strat):
        if self.state == 'gameover':
            return

        if self.intersects() and self.figure.y <= 2:
            self.state = 'gameover'
            return

        #print(self.goalPos)
        if self.goalPos is None or self.goalPos[3] != self.figure.type:
            #print(self.goalPos)
            self.goalPos = self.find_best_move(strat)

        if self.goalPos[1] <= 1:
            self.state = 'gameover'
            #return

        if self.goalPos[2] != self.figure.rotation:
            #print("roto")
            self.figure.rotation = self.goalPos[2]

        else:
            self.figure.x = self.goalPos[0]
            self.go_space()
            self.goalPos = None

    #helper function for fitness function retuns maximum hight of the board not including the current piece
def maxHeight(ha):
    return max(ha)

#helper function for fitness function retuns average height of all collums
def avgHeight(A, ha):

    for h in range(len(ha)):
        ha[h] -= len(A)

        ha[h] = abs(ha[h])

    avgA = sum(ha)/len(ha)
    #print(avgA)
    return avgA

#helper function for fitness function retuns higer value for lower pieces linearly
def boardLevel(A):
  reward = 0
  arr1 = [sum(1 for i in level if i > 0) for level in A]
  #print(arr3)
  ratio = .3/len(arr1)
  for i in range(len(arr1)):
      reward += arr1[i]*ratio*i

  return reward

#helper function for fitness function retuns number of open spots below at least one block
def numHoles(A, ha):

    heights = [0 for _ in range(len(ha))]
    for x in range(len(heights)):
        heights[x] = len(A) - ha[x]

    holesInA = 0

    for row in range(len(A)):
        for col in range(len(A[0])):
            if heights[col] < row and A[row][col] == 0:
                holesInA += 1

    return holesInA

#helper function for fitness function retuns number of lines that are full
def completeLines(A, ha):
    return sum([1  for row in A if 0 not in row])

#helper function for fitness function retuns average difference of collums
def bumps(ha):
    return sum([abs(ha[x]-ha[x+1]) for x in range(len(ha)-1)])/len(ha)

# return board without the current piece
def getTower(original):

    width = len(original[0])
    height = len(original)
    curQ = [(height-1,z) for z in range(width)]
    board = [[0 for _ in range(width)] for z in range(height)]
    visited = [[0 for _ in range(width)]  for z in range(height)]

    while len(curQ) > 0:

        look = curQ.pop(0)

        if look[0] < 0:
            continue

        if visited[look[0]][look[1]] == 0:

            visited[look[0]][look[1]] = 1

            if original[look[0]][look[1]] > 0:

                board[look[0]][look[1]] = 1

                if look[0]-1 >= 0:
                    curQ.append((look[0]-1,look[1]))
                if look[1]+1 < width:
                    curQ.append((look[0],look[1]+1))
                if look[1]-1 >= 0:
                    curQ.append((look[0],look[1]-1))

    return board

#helper function retuns the height of each collum
def towerHeights(original):
    width = len(original[0])
    height = len(original)
    heights = [height for _ in range(width)]
    for x in range(height):
        for y in range(width):

            if original[x][y] == 1 and heights[y] > x:
                heights[y] = x
    #for h in heights:
        #h -= height
    return heights

#the fitness function that uses the evalueate function and all helper functions to score moves
def rewardFunc(A, RFunc, gene):
    result = 0
    A = getTower(A)
    HA = towerHeights(A)
    #print(RFunc)
    if RFunc == 1:
        params = {}
        A = getTower(A)
        HA = towerHeights(A)
        params['B'] = bumps(HA)*(gene[1][0])
        params['A'] = avgHeight(A,HA)*(gene[1][1])
        params['H'] = numHoles(A,HA)*(gene[1][2])
        params['C'] = completeLines(A,HA)*(gene[1][3])
        params['L'] = boardLevel(A)*(gene[1][4])
        params['M'] = maxHeight(HA)*(gene[1][5])
        result = sum(params.values())
        result += gene[0].evaluate(params)
    elif RFunc == 2:
        params = {}
        A = getTower(A)
        HA = towerHeights(A)
        params['B'] = bumps(HA)*(gene.hyperParams.params[0])
        params['A'] = avgHeight(A,HA)*(gene.hyperParams.params[1])
        params['H'] = numHoles(A,HA)*(gene.hyperParams.params[2])
        params['C'] = completeLines(A,HA)*(gene.hyperParams.params[3])
        params['L'] = boardLevel(A)*(gene.hyperParams.params[4])
        params['M'] = maxHeight(HA)*(gene.hyperParams.params[5])
        result = sum(params.values())
        result += gene.evaluate(params)
    else:
        bumpsReward = bumps(HA)
        result -= gene.params[0]*bumpsReward

        heightReward = avgHeight(A,HA)
        result -= gene.params[1]*heightReward

        deltaReward = numHoles(A,HA)
        result -= gene.params[2]*deltaReward

        lineReward = completeLines(A,HA)
        result += gene.params[3]*lineReward

        levelReward = boardLevel(A)
        result += gene.params[4]*levelReward

        maxHeightReward = maxHeight(HA)
        result -= gene.params[5]*maxHeightReward

    return result

def playGame(strat, goal):
    done = False
    game = TetrisAI(gene=strat)
    while not done: # play one game
        lines = 0
        score = 0
        if game.figure is None:
            game.goalPos = None
            game.new_figure()

        done = game.makeMove(strat) # ai makes a move

        if game.state == 'gameover':
            done = True
        game.step += 1
        if game.figure is None:
            game.new_figure()
        game.go_down()

        done = True if game.state == 'gameover' else False


        if game.totalLinesCleared > goal*.7 and done:
            print('score ', game.score, ' lines ', game.totalLinesCleared)


    return game.score, game.totalLinesCleared

def playGameThread(strat, goal):
    done = False
    game = TetrisAI()
    while not done: # play one game
        lines = 0
        score = 0
        if game.figure is None:
            game.goalPos = None
            game.new_figure()

        done = game.makeMove(strat) # ai makes a move

        if game.state == 'gameover':
            done = True
        game.step += 1
        if game.figure is None:
            game.new_figure()
        game.go_down()

        done = True if game.state == 'gameover' else False


        if game.totalLinesCleared > goal and done:
            print('score ', game.score, ' lines ', game.totalLinesCleared)


    return (game.score ,game.totalLinesCleared)
    #print('res',results)
