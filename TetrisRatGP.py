import random
import os
import numpy as np
import math
from threading import  Thread
from TetrisGameAI3 import Figure, TetrisAI, playGame, playGameThread


class genome:

    def __init__(self, parameters = [random.random() for x in range(6)] , p = random.randint(10,30), i=random.randint(15,30), t=random.randint(1,10), m=random.random(), tag = 'og', fit = 0):
        self.params = parameters
        self.popSize = p
        self.influx = i
        self.mutRate = m
        self.temp = t
        self.tag = tag
        self.fitness = fit

    def __str__(self):
        result = ''
        for param in self.params:
            result+= str(round(param,ndigits=5))+', '
        return result
        #return str(round(self.params[0],ndigits=3))+', '+str(round(self.params[1],ndigits=3))+', '+str(round(self.gamma,ndigits=3))+', '+str(round(self.phi,ndigits=3))+', '+str(round(self.zeta,ndigits=3))+', '+str(round(self.delta,ndigits=3))

    def __eq__(self, other):
        for x in range(len(self.params)):
            if self.params[x] != other.params[x]:
                return False
        return self.fitness == other.fitness and self.temp == other.temp and self.mutRate == other.mutRate and self.popSize == other.popSize and self.influx == other.influx
        #return self.params[0] == other.params[0] and self.params[1] == other.params[1] and self.gamma == other.gamma and self.phi == other.phi and self.zeta == other.zeta and self.fitness == other.fitness and self.popSize == other.popSize and self.influx == other.influx and self.temp == other.temp and self.mutRate == other.mutRate and self.delta == other.delta

    def copy(self):
        return genome(self.params.copy(),self.popSize,self.influx,self.temp,self.mutRate,self.tag)

    #average of two numbers
    def avg(self, a,b):
        return (a+b)/2

    #picks a recombination strategy and two parents
    def recombine(self, mate):
        mutation_type = random.randrange(6)
        if mutation_type == 0:
            return self.oneCrossBaby(mate)
        elif mutation_type == 1:
            return self.nCrossBaby(mate)
        elif mutation_type == 2:
            return self.averageBaby(mate)
        else:
            c1 = mate.oneCrossBaby(self)
            c2 = mate.nCrossBaby(self)
            if mutation_type == 3:
                return c1
            elif mutation_type == 4:
                return c2
            else:
                return c1.averageBaby(c2)

        #return genome([self.avg(self.params[x] , mate.params[x]) for x in, self.avg(self.params[1] , mate.beta), self.avg(self.gamma , mate.gamma), self.avg(self.phi , mate.phi), self.avg(self.zeta , mate.zeta), self.avg(self.delta, mate.delta), self.avg(self.popSize, mate.popSize), self.avg(self.influx, mate.influx), self.avg(self.temp, mate.temp), self.avg(self.mutRate, mate.mutRate))

    #swaps one gene of self with corresponding gene in mate
    def oneCrossBaby(self, mate):
        newParams = self.params.copy()
        index = random.randrange(len(self.params))
        newParams[index] = mate.params[index]

        return genome(newParams, p = int(self.avg(self.popSize, mate.popSize)), i =int(self.avg(self.influx, mate.influx)), t = self.avg(self.temp, mate.temp), m =self.avg(self.mutRate, mate.mutRate), tag = self.tag+'-1C')

    #swaps n genes of self with corresponding genes in mate
    def nCrossBaby(self, mate):
        newParams = self.params.copy()
        n = random.randint(1,len(self.params)-1)
        swaps = random.sample(range(0,6),k=n)
        for index in swaps:
            newParams[index] = mate.params[index]
        return genome(newParams, p = int(self.avg(self.popSize, mate.popSize)), i =int(self.avg(self.influx, mate.influx)), t = self.avg(self.temp, mate.temp), m =self.avg(self.mutRate, mate.mutRate), tag = self.tag+'-%iC'%n)

    #averages all values of both parents
    def averageBaby(self, mate):
        return genome(parameters = [self.avg(self.params[x] , mate.params[x]) for x in range(len(self.params))], p = int(self.avg(self.popSize, mate.popSize)), i =int(self.avg(self.influx, mate.influx)), t = self.avg(self.temp, mate.temp), m =self.avg(self.mutRate, mate.mutRate), tag = self.tag+'-AB')

    #picks any number of parameters randomly and changes them based on temp
    def mutate(self, temp):
        copy = self.copy()
        paramsMutated = random.randint(1,len(copy.params))
        copy.tag += '-%iM'%paramsMutated
        mutants = random.sample([x for x in range(len(copy.params))], k=paramsMutated)
        for val in mutants:
            value = random.uniform(-10,10)/temp
            copy.params[val] += value

        upper = random.randrange(1,int(temp+2))
        lower = random.randrange(-int(temp+2),-1)
        value = random.randint(lower,upper)
        if copy.popSize + value > 3:
            copy.popSize += int(value)
        elif copy.popSize + value//2 > 1:
            copy.popSize += int(value//2)
        else:
            copy.popSize = 1

        value = random.randint(lower,upper)
        if copy.influx + value > 0:
            copy.influx += int(value)
        elif copy.influx + value//2 > 1:
            copy.influx += int(value//2)
        else:
            copy.influx = 1

        upper = temp*3*random.random()
        lower = -temp*2.5*random.random()
        value = random.uniform(lower,upper)
        if copy.temp + value > 1:
            copy.temp += value
        elif copy.temp + value//2 > 1:
            copy.temp += value//2
        else:
            copy.temp = 1
        return copy

class Rat:
    #0.9597864099655763 0.9711718414532963 0.022069810925646416 0.29238468508600435 0.38691042015197374
    def __init__(self, pop, frames_per_second, imports = None, mode = 0):
        self.numGames = 0
        self.popAvg = 0
        self.goalPos = None
        self.popSize = pop
        self.population = imports
        self.currentGenome = None
        if mode == 0:
            self.game = TetrisAI(frames_per_second)
        else:
            from TetrisGameAI2 import TetrisAI as TesterAI
            self.game = TesterAI(frames_per_second)
        if self.population == None:
            self.generatePopulation()
        else:
            self.importPopulation(self.population)
    #generated population of genomes with parndom params and assignt it to population
    def generatePopulation(self):
        self.population = self.generateRandom(self.popSize)

    def importPopulation(self, pop):
        self.population = [genome(parameters=gene[0], p = random.randint(10,30), i=random.randint(15,30), t=random.randint(1,5), m=random.random(), fit=gene[1]) for gene in pop]
    #generate a single random genome
    def generateRandom(self, size):
        #a= random.random(),b= random.random(),c= random.random(),d= random.random(),e= random.random(),f= random.random(), p = random.randint(10,30), i=random.randint(15,30), t=random.randint(1,20), m=random.random()
        return [genome(parameters = [random.uniform(0,10) for x in range(6)], p = random.randint(10,30), i=random.randint(15,30), t=random.randint(1,5), m=random.random()) for _ in range(size)]

    #return the state of the board combine with the piece
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

    #finds the highest predicted score location based on the fitness function and parameters in genome
    def find_best_move(self):
        if self.game.state == 'gameover':
            return (0,0,0)
        fig = self.game.figure
        ogFig = (fig.x, fig.y, fig.rotation)
        fits = []
        bestFit = -10000
        for _ in range(len(fig.figures[fig.type])):

            self.game.figure.x = ogFig[1]

            for x in range(9):
                self.game.test_space()
                if self.rewardFunc(self.getState()) > bestFit:
                    fits.append((fig.x, fig.y, fig.rotation, fig.type, self.rewardFunc(self.getState())))
                fig.y = ogFig[1]
                #if x % 4 == 3:
                    #self.game.test_down()
                complete = self.game.go_side(-1)
                if not complete:
                    break

            self.game.figure.x = ogFig[1]

            for _ in range(9):
                self.game.test_space()
                if self.rewardFunc(self.getState()) > bestFit:
                    fits.append((fig.x, fig.y, fig.rotation, fig.type, self.rewardFunc(self.getState())))
                fig.y = ogFig[1]
                #if x % 4 == 3:
                    #self.game.test_down()
                complete = self.game.go_side(1)
                if not complete:
                    break
                #print('tested pos')
            self.game.figure.rotate()

        self.game.figure.x = ogFig[0]
        self.game.figure.y = ogFig[1]
        self.game.figure.rotation = ogFig[2]

        return  list(max(fits, key = lambda x : x[4]))

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
        A = self.getTower(A)
        HA = self.towerHeights(A)

        bumpsReward = self.bumps(HA)
        result -= self.currentGenome.params[0]*bumpsReward

        heightReward = self.avgHeight(A,HA)
        result -= self.currentGenome.params[1]*heightReward

        deltaReward = self.numHoles(A,HA)
        result -= self.currentGenome.params[2]*deltaReward

        lineReward = self.completeLines(A,HA)
        result += self.currentGenome.params[3]*lineReward

        levelReward = self.boardLevel(A)
        result += self.currentGenome.params[4]*levelReward

        maxHeightReward = self.maxHeight(HA)
        result -= self.currentGenome.params[5]*maxHeightReward
        #if int(result) not in range(-20,20):
            #print('bumps ', bumpsReward, 'height ', heightReward, 'holes ', deltaReward, 'lines cleared ',lineReward)
            #print('total = ', result)

        return result

#stocastic selection of fittest individuals that will stay in the population
def survival(agent):
    #print(len(agent.population), agent.popSize)
    if len(agent.population) <= agent.popSize:
        return

    agent.population.sort(key = lambda x : x.fitness)
    pop = agent.population[-3:]

    for gene in pop:
        agent.population.remove(gene)

    while len(pop) < agent.popSize:
        samp = random.sample(agent.population, k = len(agent.population)//4)
        #print([s.fitness for s in pop])
        if random.random() > .4:
            contestant = max(samp, key = lambda x : x.fitness)
            agent.population.remove(contestant)
        else:
            if random.random() > .2:
                contestant = random.choice(samp)
                agent.population.remove(contestant)
            else:
                contestant = min(samp, key = lambda x : x.fitness)
                agent.population.remove(contestant)

        if contestant not in pop:
            pop.append(contestant)

    pop.sort(key=lambda x:x.fitness)
    agent.population = pop

#uses the repopulate method from genome on the whole population
def repopulate(agent, numGens, goal):
    newGenes = []
    for _ in range(numGens):
        picks1 = random.sample(agent.population, k = round(agent.popSize/4))
        picks2 = random.sample(agent.population, k = round(agent.popSize/4))
        randGeneUp = max(picks1,key=lambda x: x.fitness)
        randGeneDown = max(picks2,key=lambda x: x.fitness)
        if randGeneUp == randGeneDown:
            randGeneDown = min(picks2,key=lambda x: x.fitness)
        if random.random() > randGeneUp.mutRate:
            baby = randGeneUp.recombine(randGeneDown)
        else:
            baby = randGeneUp.mutate(randGeneUp.temp)
        newGenes.append(baby)
    print('Evaluating New...')
    newGenes = evaluatePopulation(agent, newGenes, goal)
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
def evaluatePopulation(agent, population, goal):
    for gene in population:
        agent.currentGenome = gene
        newScores, newLines = testGensThreaded(agent, goal)
        gene.fitness = avgScore(newLines)-stdDev(newLines)*.5#/max(1,avgScore(newLines))
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

            #if lines > agent.goal*.7 and done:
                #print('game: ', agent.numGames, ' score ', score, ' lines ', lines)

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

        score, lines = playGame(agent.currentGenome, goalScore)

        #if lines > goalScore*.7:
            #print(' score ', score, ' lines ', lines)

        roundScores.append(score)
        roundLines.append(lines)

    return roundScores, roundLines

def testGensThreaded(agent, goalScore, numThreads = 4):
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
                threads[i] = Thread(target=(lambda q, t, arg1,arg2: q.append(t(arg1,arg2))), args= (recentScores, playGameThread, agent.currentGenome, 4000))
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

#returns agavage score of scoreCard
def avgScore(scoreCard):
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
#updates hyper parameters of the population with average
def updateHypers(agent):
    agent.popSize = 0
    agent.influx = 0
    agent.temperature = 0
    agent.mutationRate = 0
    for genome in agent.population:
        agent.popSize += genome.popSize
        agent.influx += genome.influx
        agent.temperature += genome.temp
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
def train(goal=100, tests=16, start_size = 10, lookback_distance=101, impressive_score=1000, startPop = None):
    #temp = 1

    generation = 0
    bestGenome = None
    topScores = []
    log = {}
    agent = Rat(pop=start_size, frames_per_second=150, imports = startPop)
    agent.goal = goal
    agent.testPerAgent = tests
    if startPop == None:
        agent.population = evaluatePopulation(agent, agent.population, goal)
    bestGenome = max(agent.population, key = lambda x:x.fitness)
    topScores.append(bestGenome.fitness)
    updateHypers(agent)
    log[str(generation)] = [bestGenome.fitness, agent.popSize, agent.influx, agent.temperature]
    agent.population.sort(key=lambda x:x.fitness)
    print('Population initilized')
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

            print(x,'\tf:',round(genome.fitness),'\tpms',genome,'\ttag',tagStr) #,'\tpop',genome.popSize,'\tinf',genome.influx,'\tm-r',round(genome.mutRate,ndigits=1),'\t tmp',round(genome.temp,ndigits=1),
            x-=1
        print('============================================================================================================================')
        print('|Generation: ',generation,'\tBest Fitness: ',round(bestGenome.fitness, ndigits=3), '\tAverage Fitness: ', round(aver,ndigits=3),'\tPercent to Goal: ',round((generation/goal)*100, ndigits=2),'%','|')
        print('============================================================================================================================')
        print('|Population Influx: ',agent.influx, '\tPopulation Size: ',agent.popSize, '\tTemperature: ',round(agent.temperature,ndigits=3), '\tMutation Rate: ',round(agent.mutationRate,ndigits=2),'|')
        print('============================================================================================================================')

        print('Performing Repopulation...')
        repopulate(agent, agent.popSize + agent.influx, goal) #round(agent.popSize*(6+(temp+1)**0.6)**0.16)
        #temp = 1 + (bestGenome.fitness/(1-(bestGenome.fitness/(goal+1))))**0.81
        #mutatePopulation(agent, agent.temperature, agent.popSize)
        if aver < 200:
            noobs = agent.generateRandom(size=agent.influx)
            noobs = evaluatePopulation(agent, noobs, goal)
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
# find best solution given a set of requirements and then play with it until
# the desired score is reached all data is saved to ./model
a0 = ([0.29382, -0.11575, 0.27013, 0.16258, 0.58243, -0.02852],310) #  310.42  190     365.8195769
a1 = ([0.29382, 0.24452, 0.13921, 0.16258, 0.57649, -0.02852],486) #   489.3   379.5	  455.5812278
a2 = ([0.52504, 0.34867, 0.23395, 0.60486, 1.01676, -0.02544],400) #   400.54	293	    335.7919795
a3 = ([0.66681, 0.2908, 0.41964, 0.47442, 0.82171, -0.03511],368 )#    368.02	279	    339.9396399
a4 = ([0.63698, 0.14493, 0.48567, 0.86632, 0.96635, 0.02816],298 )#    298.04	244	    197.0456241
a5 = ([0.65836, 0.17403, 0.41379, 0.85921, 0.96635, 0.05371],352 )#    352.78	301.5	  280.3268712
a6 = ([0.51012, 0.56257, 0.37175, 0.16006, 0.8074, 0.01967], 210 )#     210.42	141.5	  209.2896723
a7 = ([0.40016, 0.59975, 0.15026, 0.22425, 0.71479, 0.02291], 199)#     199.56	171	    122.4131779
a8 = ([0.48109, 0.24348, 0.30957, 0.1519, 0.99108, 0.00827], 396 ) #    396.08	266.5	  323.5307663
a9 = ([0.82992, 0.24348, 0.34477, 0.1519, 0.99108, 0.00827], 377 ) #    377.72	299.5	  313.5784926

base_genes = [a0,a1,a2,a3,a4,a5,a6,a7,a8,a9]
if __name__ == '__main__':
    goalScore = 3000000
    itterations = 100
    beast = train()# startPop= base_genes)
    #arr0 = [0.04, 0.281, 0.224, 0.59, 0.881, 0.177]
    #arr1 = [0.282, 0.168, 0.145, 0.604, 0.593, 0.045]
    #arr2 = [0.499, 0.482, 0.222, 0.672, 0.763, 0.016]
    #beast = genome(parameters=arr)
    play(goalScore, beast, 100)
