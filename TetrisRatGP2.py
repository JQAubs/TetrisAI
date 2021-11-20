import random
import os
import numpy as np
import math
from math import inf
from TetrisGameAI3 import Figure, TetrisAI, playGame, playGameThread
from TreeGenotypeF import treeGenotype
from numba import jit, cuda
from multiprocessing import Process
from threading import Thread


class Rat:
    #0.9597864099655763 0.9711718414532963 0.022069810925646416 0.29238468508600435 0.38691042015197374
    def __init__(self, pop, frames_per_second, params, goalScore, numTests, mode =0):
        self.numGames = 0
        self.popAvg = 0
        self.goalPos = None
        self.popSize = pop
        self.population = None
        self.currentGenome = None
        self.hyperParams = params[0]
        self.goal = goalScore
        self.testPerAgent = numTests
        #print(self.hyperParams)
        self.meanOffset = params[1]
        if mode == 0:
            self.game = TetrisAI(frames_per_second)
        else:
            from TetrisGameAI2 import TetrisAI as TesterAI
            self.game = TesterAI(frames_per_second)
        self.generatePopulation()

    def generatePopulation(self):
        pop = []
        for gene in range(self.popSize):
            newgenome = treeGenotype(depth_limit = 5, popSize=random.randint(10,80), influx=random.randint(8,20), temp=random.random())
            #newgenome.mutate()
            pop.append(newgenome)
            #print(pop[gene])
        self.population = pop

    def generateRandom(self, size):
        return [treeGenotype() for _ in range(size)]

    def lookAround(self, x, y):
        for row in range(1,4):
            for col in range(-1,2):
                if row == col:
                    continue
                yVal = row+y
                xVal = col+x+self.game.figure.x
                if xVal in range(len(self.game.field[0])):
                    if yVal in range(len(self.game.field)):
                        if self.game.field[yVal][xVal] > 0:
                            return True
        return False

    def getState2(self):
        #print('old\n',self.getState3())
        board = self.game.field
        width = len(self.game.field[0])-1
        height = len(self.game.field)-1
        #print('new')
        piece = self.game.figure.image()
        newBoard = [[y for y in x] for x in board]

        touching = False
        #print(self.game.field)
        for item in piece:
            ny = height - abs((item//4)-4)
            nx = item%4
            #print(ny)

            if self.game.figure.y+item//4  == 23 or self.lookAround(nx, ny):
                #print('found')
                touching = True
                continue

        if touching:
            for item in piece:
                nx = item//4
                ny = item%4
                newBoard[nx+self.game.figure.y][ny+self.game.figure.x] = 1

        for row in range(len(newBoard)):
            for col in range(len(newBoard[0])):
                if newBoard[row][col] > 0:
                    newBoard[row][col] = 1
                else:
                    newBoard[row][col] = 0
        return newBoard

    def getState(self):
        board = self.game.field
        piece = self.game.figure
        newBoard = [[y for y in x] for x in board]
        arr = [[0 for i in range(4)] for y in range(4)]
        for x in range(len(arr)):
            for y in range(len(arr)):
                if  x*len(arr[0])+y in piece.image():
                    newBoard[piece.y+x][piece.x+y] = 1
        #print(board)
        return np.array(newBoard, dtype=int)

    def find_best_move2(self, strat):
        if self.game.state == 'gameover':
            return (0,0,0)
        #print(type(strat))
        if isinstance(strat, list):
            rType = 1
        elif isinstance(strat, treeGenotype):
            rType = 2
        else:
            rType = 0
        #print(rType)
        fig = self.game.figure
        ogFig = (fig.x, fig.y, fig.rotation)
        self.game.test_space()
        beast = (fig.x, fig.y, fig.rotation, fig.type, self.rewardFunc(self.getState2()))
        fig.y = ogFig[1]

        for _ in range(len(fig.figures[fig.type])):

            self.game.figure.x = ogFig[1]

            for _ in range(9):
                self.game.test_space()
                fit = self.rewardFunc(self.getState2())

                if fit > beast[4]:
                    beast = (fig.x, fig.y, fig.rotation, fig.type, fit)

                fig.y = ogFig[1]

                if not self.game.go_side(-1):
                    break

            self.game.figure.x = ogFig[1]
            self.game.go_side(1)
            for _ in range(9):
                self.game.test_space()
                fit = self.rewardFunc(self.getState2())

                if fit > beast[4]:
                    beast = (fig.x, fig.y, fig.rotation, fig.type, fit)

                fig.y = ogFig[1]

                if not self.game.go_side(1):
                    break

            self.game.figure.rotate()

        self.game.figure.x = ogFig[0]
        self.game.figure.y = ogFig[1]
        self.game.figure.rotation = ogFig[2]

        return beast

    #finds the highest predicted score location based on the fitness function and parameters in genome
    def find_best_move(self):
        if self.game.state == 'gameover':
            return (0,0,0)
        fig = self.game.figure
        ogFig = (fig.x, fig.y, fig.rotation)
        self.game.test_space()
        beast = (fig.x, fig.y, fig.rotation, fig.type, self.rewardFunc(self.getState2()))
        fig.y = ogFig[1]

        for _ in range(len(fig.figures[fig.type])):

            self.game.figure.x = ogFig[1]

            for _ in range(9):
                self.game.test_space()
                fit = self.rewardFunc(self.getState2())

                if fit > beast[4]:
                    beast = (fig.x, fig.y, fig.rotation, fig.type, fit)

                fig.y = ogFig[1]

                if not self.game.go_side(-1):
                    break

            self.game.figure.x = ogFig[1]
            self.game.go_side(1)
            for _ in range(9):
                self.game.test_space()
                fit = self.rewardFunc(self.getState2())

                if fit > beast[4]:
                    beast = (fig.x, fig.y, fig.rotation, fig.type, fit)

                fig.y = ogFig[1]

                if not self.game.go_side(1):
                    break

            self.game.figure.rotate()

        self.game.figure.x = ogFig[0]
        self.game.figure.y = ogFig[1]
        self.game.figure.rotation = ogFig[2]

        return beast

    #responsible for performing the best predicted action
    def makeMove(self):
        if self.game.state == 'gameover':
            return

        if self.game.intersects() and self.game.figure.y <= 2:
            self.game.state = 'gameover'
            return

        #print(self.goalPos)
        if self.goalPos is None or self.goalPos[3] != self.game.figure.type:
            #print(self.goalPos)
            self.goalPos = self.find_best_move()

        if self.goalPos[1] <= 1:
            self.game.state = 'gameover'
            #return

        if self.goalPos[2] != self.game.figure.rotation:
            #print("roto")
            self.game.figure.rotation = self.goalPos[2]

        else:
            self.game.figure.x = self.goalPos[0]
            self.game.go_space()
            self.goalPos = None

    #helper function for fitness function retuns maximum hight of the board not including the current piece
    def maxHeight(self, ha):
        return max(ha)

    #helper function for fitness function retuns average height of all collums
    def avgHeight(self, A, ha):

        for h in range(len(ha)):
            ha[h] -= len(A)

            ha[h] = abs(ha[h])

        avgA = sum(ha)/len(ha)
        #print(avgA)
        return avgA

    #helper function for fitness function retuns higer value for lower pieces linearly
    def boardLevel(self, A):
      reward = 0
      arr1 = [sum(1 for i in level if i > 0) for level in A]
      #print(arr3)
      ratio = .3/len(arr1)
      for i in range(len(arr1)):
          reward += arr1[i]*ratio*i

      return reward

    #helper function for fitness function retuns number of open spots below at least one block
    def numHoles(self, A, ha):

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
    def completeLines(self, A, ha):
        return sum([1  for row in A if 0 not in row])

    #helper function for fitness function retuns average difference of collums
    def bumps(self, ha):
        return sum([abs(ha[x]-ha[x+1]) for x in range(len(ha)-1)])/len(ha)

    # return board without the current piece
    def getTower(self, original):

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
    def towerHeights(self, original):
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
    def rewardFunc(self, A):
        result = 0
        params = {}
        #A = self.getTower(A)
        HA = self.towerHeights(A)
        params['B'] = self.bumps(HA)*(self.hyperParams[0])
        params['A'] = self.avgHeight(A,HA)*(self.hyperParams[1])
        params['H'] = self.numHoles(A,HA)*(self.hyperParams[2])
        params['C'] = self.completeLines(A,HA)*(self.hyperParams[3])
        params['L'] = self.boardLevel(A)*(self.hyperParams[4])
        params['M'] = self.maxHeight(HA)*(self.hyperParams[5])
        result = sum(params.values())
        result += self.currentGenome.evaluate(params)
        return result/2

#['A','L','H','C','B','#.#']
#@jit(nopython=True)

    def rewardFunc2(self, A, RFunc, gene):
        result = 0
        #A = getTower(A)
        #print(A)
        HA = self.towerHeights(A)
        #print(RFunc)
        if RFunc == 1:
            params = {}
            params['B'] = self.bumps(HA)*(gene[1][0])
            params['A'] = self.avgHeight(A,HA)*(gene[1][1])
            params['H'] = self.numHoles(A,HA)*(gene[1][2])
            params['C'] = self.completeLines(A)*(gene[1][3])
            params['L'] = self.boardLevel(A)*(gene[1][4])
            params['M'] = self.maxHeight(HA)*(gene[1][5])
            result = sum(params.values())
            result += gene[0].evaluate(params)
        elif RFunc == 2:
            params = {}
            params['B'] = self.bumps(HA)*(gene.hyperParams.params[0])
            params['A'] = self.avgHeight(A,HA)*(gene.hyperParams.params[1])
            params['H'] = self.numHoles(A,HA)*(gene.hyperParams.params[2])
            params['C'] = self.completeLines(A)*(gene.hyperParams.params[3])
            params['L'] = self.boardLevel(A)*(gene.hyperParams.params[4])
            params['M'] = self.maxHeight(HA)*(gene.hyperParams.params[5])
            result = sum(params.values())
            result += gene.evaluate(params)
        else:
            bumpsReward = self.bumps(HA)
            result -= gene.params[0]*bumpsReward

            heightReward = self.avgHeight(A,HA)
            result -= gene.params[1]*heightReward

            deltaReward = self.numHoles(A,HA)
            result -= gene.params[2]*deltaReward

            lineReward = self.completeLines(A)
            result += gene.params[3]*lineReward

            levelReward = self.boardLevel(A)
            result += gene.params[4]*levelReward

            maxHeightReward = self.maxHeight(HA)
            result -= gene.params[5]*maxHeightReward

        return result

def averageDistance(agent, current):
    total = 0

    for rat in agent.population:
        if rat is current:
            continue
        total += ((rat.alpha-current.alpha)**2+(rat.beta-current.beta)**2+(rat.gamma-current.gamma)**2+(rat.phi-current.phi)**2+(rat.zeta-current.zeta)**2)**0.5

    total /= len(agent.population)

    return abs(total)

def survival(agent):
    if len(agent.population) <= agent.popSize:
        return
    pop = agent.population.copy()
    pop.sort(key=lambda x:x.fitness)
    pop = pop[-agent.popSize:]
    agent.population = pop

def mutatePopulation2(agent, temp, numGens):
    mutants = []
    for gene in agent.population:
        for x in range(numGens):
            randGene = agent.population[-(x%len(agent.population))]
            newGene = genome(randGene.alpha,randGene.beta,randGene.gamma,randGene.phi,randGene.zeta)
            newGene.mutate(temp)
            mutants.append(newGene)
    return mutants

def repopulate(agent, numGens):
    newGenes = []
    for _ in range(numGens):
        picks1 = random.sample(agent.population, k = round(len(agent.population)/4))
        picks2 = random.sample(agent.population, k = round(len(agent.population)/4))
        randGeneUp = max(picks1,key=lambda x: x.fitness)
        randGeneDown = max(picks2,key=lambda x: x.fitness)
        if randGeneUp == randGeneDown:
            randGeneDown = min(picks2,key=lambda x: x.fitness)
        if random.random() > randGeneUp.mutRate:
            baby = randGeneUp.recombine(randGeneDown)
        else:
            baby = randGeneUp.mutate(randGeneUp.temperature)
        newGenes.append(baby)
    print('Evaluating New...')
    newGenes = evaluatePopulation(agent, newGenes)
    agent.population.extend(newGenes)

#mutates a designated number of genomes in the population and add the results to population
def mutatePopulation(agent, temp, numGens):
    mutants = []
    for _ in range(numGens):
        picks = random.sample(agent.population, k = len(agent.population)//3)
        randGeneUp = max(picks,key=lambda x: x.fitness)
        randGeneDown = min(picks,key=lambda x: x.fitness)
        randGene = random.choices([randGeneDown,randGeneUp], weights = [1,5], k = 1)[0]
        newGene = genome(parameters=randGene.params)
        newGene.mutate(temp)
        mutants.append(newGene)
    mutants = evaluatePopulation(agent, mutants)
    agent.population.extend(mutants)

#play designated number of games with each genome and assign fitness based on average and median
def evaluatePopulation(agent, population):
    for gene in population:
        agent.currentGenome = gene
        newScores, newLines = testGensThreaded(agent)
        #print('updates',newScores, newLines)
        gene.fitness = avgScore(newLines)-agent.meanOffset#/max(1,avgScore(newLines))
    return population

#helper function for evaluate population responsible for playing all games and returning stats
def testGenerations(agent):
    roundScores = []
    roundLines = []
    for gen in range(agent.testPerAgent): # play the number of games specefied
        done = False
        count = 0
        while not done: # play one game
            lines = 0
            score = 0
            if agent.game.figure is None:
                agent.goalPos = None
                agent.game.new_figure()

            done = agent.makeMove() # ai makes a move

            if done or count%5==4: # update visual every 10 moves
                done, score, lines = agent.game.game_step2()

            if lines > agent.goal*.7 and done:
                print('game: ', agent.numGames, ' score ', score, ' lines ', lines)

            if done:
                agent.game.reset() # reset game when finished
                agent.numGames += 1
            count += 1

        roundScores.append(score)
        roundLines.append(lines)
        #cut testing short if the gene's average is not better than 80% of the population average
        # start short cut when at least half the evaluations are complete
        if len(roundScores) > round(agent.testPerAgent/2) and avgScore(roundLines) < avgScore([x.fitness for x in agent.population])**0.8:
            break
    return roundScores, roundLines

def testGens(agent, goalScore):
    roundScores = []
    roundLines = []
    for gen in range(agent.testPerAgent):

        score, lines = playGame([agent.currentGenome,agent.hyperParams], goalScore)

        if lines > goalScore*.7:
            print(' score ', score, ' lines ', lines)

        roundScores.append(score)
        roundLines.append(lines)

    return roundScores, roundLines

def testGensThreaded(agent, numThreads = 4):
    roundScores = []
    roundLines = []
    #print(agent.testPerAgent)
    if agent.testPerAgent < 4:
        numThreads = agent.testPerAgent
    threads = [None] * numThreads
    recentScores = list()
    while agent.testPerAgent > len(roundLines):

        for i in range(len(threads)):
            living = sum([1 for i in threads if i != None and i.is_alive()])
            if living+len(roundLines) < agent.testPerAgent and (threads[i] == None or  not threads[i].is_alive()):
                threads[i] = Thread(target=(lambda q, t, arg1,arg2: q.append(t(arg1,arg2))), args= (recentScores,playGameThread,[agent.currentGenome,agent.hyperParams], agent.goal+agent.meanOffset))
                threads[i].start()
        #print([i.is_alive() for i in threads])
        for i in range(len(threads)):
            threads[i].join()
        #print([i.is_alive() for i in threads])
        for item in recentScores:
            #print(item)
            roundScores.append(item[0])
            roundLines.append(item[1])
            recentScores.remove(item)

        #print(len(roundLines))
        #print(roundLines)
    #print(roundLines)
    return roundScores, roundLines

def avgScore(scoreCard):
    #print(type(sum(scoreCard)), type(max(1,len(scoreCard)))
    return sum(scoreCard)/ max(1,len(scoreCard))

#returns median score in scoreCard
def medianScore(scoreCard):
    return scoreCard[len(scoreCard)//2]

#returns the average of the sum of the average and median in scoreCard
def avgMedScore(scoreCard):
    return (avgScore(scoreCard)+medianScore(scoreCard))/2

def variance(scoreCard):
    mean = avgScore(scoreCard)
    return sum([(val-mean)**2 for val in scoreCard])/len(scoreCard)

def stdDev(scoreCard):
    return variance(scoreCard)**0.5

def updateHypers(agent):
    agent.popSize = 0
    agent.influx = 0
    agent.temperature = 0
    agent.mutationRate = 0
    for genome in agent.population:
        agent.popSize += genome.popSize
        agent.influx += genome.influx
        agent.temperature += genome.temperature
        agent.mutationRate += genome.mutRate

    agent.popSize = int(agent.popSize/len(agent.population))
    agent.influx = int(agent.influx/len(agent.population))
    agent.temperature = agent.temperature/len(agent.population)
    agent.mutationRate = agent.mutationRate/len(agent.population)
    return agent

#algorithm to find the best set of parameters using all the above methods together
#genetic program style recombination amd mutation using above methods
#goal is set to the number of lines that the parameters clear on average
#tests is the munber of games a set of parameters is played with before giving fitness
def train(training_name = 'solo', goal=100,tests=16, start_size = 10, hypers = ([-0.82992, -0.01621, -0.3608, 0.1605, 0.95009, -0.04574], 731),lookback_distance=101, impressive_score=2000):
    #temp = 1
    generation = 0
    bestGenome = None
    topScores = []
    log = {}
    agent = Rat(pop=start_size, frames_per_second=150, params = hypers, goalScore= 4000, numTests = tests)
    agent.population = evaluatePopulation(agent, agent.population)
    bestGenome = max(agent.population, key = lambda x:x.fitness)
    topScores.append(bestGenome.fitness)
    updateHypers(agent)
    log[str(generation)] = [bestGenome.fitness, agent.popSize, agent.influx, agent.temperature]
    agent.population.sort(key=lambda x:x.fitness)
    print('%s Population initilized'% training_name)
    print('----------------------------------------------------------------------------------------------------------------------------')
    while generation < goal:

        fitnessVals = [p.fitness for p in agent.population]
        aver = avgScore(fitnessVals)
        agent.popAvg = aver

        print('----------------------------------------------------------------------------------------------------------------------------')
        x = len(agent.population)
        for genome in agent.population:
            tagStr = genome.tag
            if len(tagStr) > 2:
                if len(tagStr) > 5:
                    tagStr = tagStr[-8:]
                else:
                    tagStr = tagStr[-5:]

            print(x,'\tf:',round(genome.fitness),'\tpop',genome.popSize,'\tinf',genome.influx,'\tm-r',round(genome.mutRate,ndigits=1),'\t tmp',round(genome.temperature,ndigits=1),'\ttag',tagStr,'\tgene',genome)
            x-=1
        print('============================================================================================================================')
        print('|%s:'%training_name,'Generation: ',generation,'\tBest Fitness: ',round(bestGenome.fitness, ndigits=3), '\tAverage Fitness: ', round(aver,ndigits=3),'\tPercent to Goal: ',round((bestGenome.fitness/goal)*100, ndigits=2),'%','|')
        print('============================================================================================================================')
        print('|Population Influx: ',agent.influx, '\tPopulation Size: ',agent.popSize, '\tTemperature: ',round(agent.temperature,ndigits=3), '\tMutation Rate: ',round(agent.mutationRate,ndigits=2),'|')
        print('============================================================================================================================')

        print('Performing Repopulation...')
        repopulate(agent, agent.popSize + agent.influx) #round(agent.popSize*(6+(temp+1)**0.6)**0.16)
        #temp = 1 + (bestGenome.fitness/(1-(bestGenome.fitness/(goal+1))))**0.81
        #mutatePopulation(agent, agent.temperature, agent.popSize)
        if aver < 200:
            noobs = agent.generateRandom(size=agent.influx)
            noobs = evaluatePopulation(agent, noobs)
            agent.population.extend(noobs)
        print('Performing Survival...')
        survival(agent)
        print('Updating Best...')
        contender = max(agent.population, key = lambda x: x.fitness)
        if contender.fitness > bestGenome.fitness:
            bestGenome = contender
            if bestGenome.fitness > impressive_score:
                with open('./model/params%i.txt'%generation,'w') as file:
                    file.write(str(bestGenome)+str(bestGenome.fitness))
                    file.close()
        topScores.append(bestGenome.fitness)
        agent = updateHypers(agent)
        generation +=1
        log[str(generation)] = [bestGenome.fitness, agent.popSize, agent.influx, agent.temperature]

        if len(topScores) > lookback_distance and topScores[-lookback_distance] == topScores[-1]:
            agent.generatePopulation()
            bestGenome = agent.population[0]


    with open('./model/paramsfinal%f.txt'%round(bestGenome.fitness),'w') as f:
        #f.write(str(bestGenome)+str(bestGenome.fitness)+'\n')
        for x in range(1,4):
            f.write(str(agent.population[-x])+str(agent.population[-x].fitness)+'\n')
        f.close()

    with open('./model/logOverTime%f.txt'%round(bestGenome.fitness), 'w') as f:
        for key, val in log.items():
            valStr = ''
            for value in val:
                valStr += str(value)+'\t'
            f.write('%s\t%s\n'%(key, valStr))
        f.close()

    return bestGenome

# play game using an agent until the desired tetris game score is achieved
def play(goal, genome, fps):
    record = 0
    agent = Rat(0, fps)
    agent.currentGenome = genome
    games = 0
    while record < goal and games < 100:

        done = False
        while not done:
            #print('c')
            if agent.game.figure is None:
                agent.goalPos = None
                agent.game.new_figure()

            #for _ in range(5):
            agent.makeMove()

            done, score, lines = agent.game.game_step2()
            if done:
                agent.game.reset()
                agent.numGames += 1

        if score > record:
            record = score

        print('-------------------------------------------')
        print('game: ',agent.numGames+1,'best: ', record, 'last: ', score)
        print('-------------------------------------------')
        games +=1

defaults = ([-0.82992, -0.01621, -0.3608, 0.1605, 0.95009, -0.04574], 731)

number_of_processes = 3

if __name__ == '__main__':
    itterations = 100
    #processes = [None] * number_of_processes
    multPr = False
    if multPr == True:
        for x in range(len(processes)):
            processes[x] = Process(target = train, args=('process%i'%x, itterations,16, 10, ([-0.82992, -0.01621, -0.3608, 0.1605, 0.95009, -0.04574], 731),40, 1000))
            processes[x].start()

        for pr in processes:
            pr.join()
    else:
        beast = train()
        play(goalScore+100, beast)
