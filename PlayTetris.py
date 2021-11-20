import random
import os
import numpy as np
import math

from TetrisGameAI2 import Figure, TetrisAI
from TetrisRatGP import genome, Rat
from TetrisRatGP2 import Rat as TreeRat
from TreeGenotypeF import treeGenotype

defaults = ([-0.82992, -0.01621, -0.3608, 0.1605, 0.95009, -0.04574], 700)

def playGame(goal, genome, fps, record, gameNum, gametype):
    log = None
    if gametype == 0:
        agent = Rat(0, fps ,mode = 1)
    else:
        agent = TreeRat(0,fps,defaults, 1000,goal, mode = 1)
    agent.currentGenome = genome
    done = False
    while not done:
        if agent.game.figure is None:
            agent.goalPos = None
            agent.game.new_figure()

        agent.makeMove()

        done, score, lines = agent.game.game_step2()
        if done:
            agent.game.reset()
            agent.numGames += 1

    if score > record:
        record = score

    print('--------------------------------------------------------------------------------------')
    print('game: ',gameNum,'Record',record, 'Latest Score: ', score,'Lines Cleared: ', lines,)
    print('--------------------------------------------------------------------------------------')
    log = (gameNum, record, score, lines)
    return log

goal = 100                                                     # mean    median  stand dev

arr0 = [0.82992, 0.05486, 0.34477, 0.1519, 0.94051, -0.02218] #~ 675.06	510.5	472.8180566
arr1 = [0.82992, 0.03861, 0.35209, 0.16243, 0.95009, -0.02771]#~ 815.86	634	690.8021368
arr2 = [0.82992, 0.03861, 0.35209, 0.17141, 0.94051, -0.02771]#~ 592.78	385.5	519.3212324
#arr3 = [0.82992, 0.01621, 0.3608, 0.1605, 0.95009, -0.04574,] # frankenstein
arr4 = [0.42785, 0.2788, 0.1906, 0.15991, 0.86405, -0.02852] # todo - test
arr5 = [7.57067, 11.14045, 3.18415, 2.53337, 3.46816, -0.04546]
arr6 = [0.82992, 0.01621, 0.3608, 0.1605, 0.95009, -0.04574]
arr7 = [0.43146, 0.55959, 0.1642, 0.22202, 0.80405, -0.07107]
arr8 = [-3.70785, -2.69122, -3.4049, 3.48868, -4.99662, -4.03969]
tree1 = ('L-B**2',	[-3.44969, 18.51911, -18.23668, 6.38939, 16.31157, 0.65219])
tree2 = ('L-B**2',	arr6)
tree3 = ('L-B**2',	arr7)
tree3 = ('max(B,C)-B**2',	[-3.44969, 18.51911, -18.23668, 6.38939, 16.31157, 0.65219])
tree4 = ('max(B,C)-B**2',	[0.82992, 0.01621, 0.3608, 0.1605, 0.95009, -0.04574])
tree5 = ('max(B,H)-B**2',	[-3.44969, 18.51911, -18.23668, 6.38939, 16.31157, 0.65219])
tree6 = ('max(B,H)-B**2',	[0.82992, 0.01621, 0.3608, 0.1605, 0.95009, -0.04574])
tree7 = ('(H+L)/(1+abs(int(B)))',  [5.21175, 2.00397, 4.87574, 5.3177, 5.80112, 3.593])
arr9 = [-2.38001, -0.86376, -1.48682, 1.03693, 1.76928, -0.13918]
tree8 = ('(-0.1225*L)*(A/(1+abs(int(A))))', [-2.38001, -0.86376, -1.48682, 1.03693, 1.76928, -0.13918])
strats = [arr9, tree8]#
results = []
record = 0
count = 0

for strat in range(len(strats)):
    demo_speed = 150
    for count in range(goal):
        if isinstance(strats[strat],list) and len(strats[strat]) > 2:
            curGene = genome(parameters=strats[strat])
            result = playGame(goal, curGene, demo_speed, record,count,0)
        else:
            if isinstance(strats[strat],tuple):
                curGene = treeGenotype(hyper=strats[strat][1], gene=strats[strat][1])
            else:
                curGene = treeGenotype(gene=strats[strat])
            result = playGame(goal, curGene, demo_speed, record,count,1)
        #demo_speed += 20
         #strats[count%len(strats)]
        results.append(result)
        record = results[-1][1]

    best = results[-1][1]
    with open('./model/Highscore%i_strat%i.txt'%(best, strat),'w') as f:
        f.write(str(strats[strat])+'\n')
        for item in results:
            string = ''
            for val in item:
                string += str(val) + '\t'
            string +='\n'
            f.write(string)
    results = []
    record = 0
