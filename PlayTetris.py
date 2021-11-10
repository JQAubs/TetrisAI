import random
import os
import numpy as np
import math

from TetrisGameAI2 import Figure, TetrisAI
from TetrisRatGP import genome, Rat
from TetrisRatGP2 import Rat as TreeRat
from TreeGenotype import treeGenotype

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

goal = 30                                                     # mean    median  stand dev

arr0 = [0.82992, 0.05486, 0.34477, 0.1519, 0.94051, -0.02218] #~ 675.06	510.5	472.8180566
arr1 = [0.82992, 0.03861, 0.35209, 0.16243, 0.95009, -0.02771]#~ 815.86	634	690.8021368
arr2 = [0.82992, 0.03861, 0.35209, 0.17141, 0.94051, -0.02771]#~ 592.78	385.5	519.3212324
arr3 = [0.82992, 0.01621, 0.3608, 0.1605, 0.95009, -0.04574,] # frankenstein
arr4 = [0.42785, 0.2788, 0.1906, 0.15991, 0.86405, -0.02852] # todo - test


strats = [tree2, tree3]#
results = []
record = 0
count = 0

for strat in range(len(strats)):
    for count in range(goal):
        if isinstance(strat,list):
            curGene = genome(parameters=strats[strat])
            result = playGame(goal, curGene, 150, record,count,0)
        else:
            curGene = treeGenotype(gene=strat)
            result = playGame(goal, curGene, 150, record,count,1)
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
