from time import perf_counter as timer
from TetrisGameAIF import playGameThread
from threading import Thread
from TetrisRatGPF import Rat
from numba import jit, cuda
import multiprocessing
import random
import math
import sys


import cProfile, pstats, io

def profile(fnc):

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


#['A','L','H','C','B','#.#']
#@jit(nopython=True)
from multiprocessing import Pool, Process

def survival(agent, method = 1):
    if len(agent.population) <= agent.popSize:
        return
    if method == 0:
        pop = agent.population.copy()
        pop.sort(key=lambda x:x.fitness)
        pop = pop[-agent.popSize:]
        agent.population = pop
    else:
        agent.population.sort(key = lambda x : x.fitness)
        pop = agent.population[-3:]
        for gene in pop:
            agent.population.remove(gene)

        while len(pop) < agent.popSize:
            samp = random.sample(agent.population, k = len(agent.population)//4)
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

#play designated number of games with each genome and assign fitness based on average and median
def evaluatePopulation(agent, population, bench=False):
    genomeType = agent.GType
    for gene in population:
        agent.currentGenome = gene
        newScores, newLines = testGensPoolProcess(agent, bench)
        gene.fitness = avgScore(newLines)-stdDev(newLines)*.5

    return population

def testGensPoolProcess(agent, bench, numProcesses = 6):
    roundScores = []
    roundLines = []
    numProcesses = multiprocessing.cpu_count()
    if agent.testPerAgent < numProcesses:
        numProcesses = agent.testPerAgent

    with Pool(processes=numProcesses) as pool:
        recentScores = []#(lambda q, t, arg1,arg2: q.append(t(arg1,arg2))), (recentScores, playGameThread, agent.currentGenome, 4000)
        recentScores = [pool.apply_async(playGameThread, (agent.currentGenome, 8000, bench)) for _ in range(agent.testPerAgent)]
        recentScores = [res.get() for res in recentScores]

        pool.close()
        pool.join()
        #print(recentScores)
        for item in recentScores:
            #print(item)
            roundScores.append(item[0])
            roundLines.append(item[1])
    return roundScores, roundLines

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


#@profile
def trainGP(gtype, goal=100, tests=16, start_size = 15, mode='train'):
    #temp = 1
    lookback_distance=150
    impressive_score = 8000
    generation = 0
    bestGenome = None
    total_process_time = 0.0
    topScores = []
    log = {}
    #print('gp',mode)
    start_time = timer()
    #print('gtype',gtype)
    agent = Rat(gtype, pop = start_size, mode = mode)
    agent.testPerAgent = tests
    if not mode == 'hof':
        if mode == 'bench':
            agent.population = evaluatePopulation(agent, agent.population, True)
        else:
            agent.population = evaluatePopulation(agent, agent.population, False)
    end_time = timer()
    time_delta = end_time - start_time
    total_process_time += time_delta
    print(f'population took {time_delta:0.4f} seconds to generate')
    bestGenome = max(agent.population, key = lambda x:x.fitness)
    topScores.append(bestGenome.fitness)
    updateHypers(agent)
    log[str(generation)] = [bestGenome.fitness, agent.popSize, agent.influx, agent.temperature]
    agent.population.sort(key=lambda x:x.fitness)
    print('Population initilized')
    print('----------------------------------------------------------------------------------------------------------------------------')
    while (total_process_time/3600) < goal:

        fitnessVals = [p.fitness for p in agent.population]
        aver = avgScore(fitnessVals)
        agent.popAvg = aver

        print('----------------------------------------------------------------------------------------------------------------------------')
        x = len(agent.population)
        for genome in agent.population:
            print(x,'- f:',round(genome.fitness),'\tGene',genome) #,'\tpop',genome.popSize,'\tinf',genome.influx,'\tm-r',round(genome.mutRate,ndigits=1),'\t tmp',round(genome.temp,ndigits=1),
            x-=1
        print('============================================================================================================================')
        print('|Generation: ',generation,'\tBest Fitness: ',round(bestGenome.fitness, ndigits=3), '\tAverage Fitness: ', round(aver,ndigits=3),'\tPercent to goal',round(100*(total_process_time/3600)/goal, ndigits=2),'%','|')
        print('============================================================================================================================')
        print('|Population Influx: ',agent.influx, '\tPopulation Size: ',agent.popSize, '\tTemperature: ',round(agent.temperature,ndigits=3), '\tMutation Rate: ',round(agent.mutationRate,ndigits=2),'|')
        print('============================================================================================================================')

        print('Performing Repopulation...')
        start_time = timer()
        if gtype == 1 or gtype == 2:
            repopulate(agent, agent.popSize + agent.influx) #round(agent.popSize*(6+(temp+1)**0.6)**0.16)

        #temp = 1 + (bestGenome.fitness/(1-(bestGenome.fitness/(goal+1))))**0.81
        #mutatePopulation(agent, agent.temperature, agent.popSize)
        if aver < 200:
            noobs = agent.generateRandom(size=agent.influx)
            noobs = evaluatePopulation(agent, noobs)
            agent.population.extend(noobs)
        print('Performing Survival...')
        survival(agent, method = 1)
        print('Updating Best...')
        end_time = timer()
        time_delta = end_time - start_time
        total_process_time += time_delta
        print(f'generation took {time_delta:0.4f} seconds to cycle')

        contender = max(agent.population, key = lambda x: x.fitness)
        if contender.fitness > bestGenome.fitness:
            bestGenome = contender
            if bestGenome.fitness > impressive_score:
                with open('./model/longRun/params%i.txt'%generation,'w') as file:
                    file.write(str(bestGenome)+str(bestGenome.fitness))
                    file.close()
        topScores.append(bestGenome.fitness)
        agent = updateHypers(agent)
        generation +=1
        print('cumulative average process time: ', total_process_time/(generation+1))
        log[str(generation)] = [bestGenome.fitness, agent.popSize, agent.influx, agent.temperature]

        if len(topScores) > lookback_distance and topScores[-lookback_distance] == topScores[-1]:
            agent.generatePopulation()
            bestGenome = agent.population[0]

        #clear_output(wait=True)
    if goal > 1:
        print('----------------------------------------------------------------------------------------------------------------------------')
        x = len(agent.population)
        for genome in agent.population:
            print(x,'\tf:',round(genome.fitness),'\tgene',genome) #,'\tpop',genome.popSize,'\tinf',genome.influx,'\tm-r',round(genome.mutRate,ndigits=1),'\t tmp',round(genome.temp,ndigits=1),
            x-=1
        print('============================================================================================================================')
        print('|Generation: ',generation,'\tBest Fitness: ',round(bestGenome.fitness, ndigits=3), '\tAverage Fitness: ', round(aver,ndigits=3),'|')
        print('============================================================================================================================')
        print('|Population Influx: ',agent.influx, '\tPopulation Size: ',agent.popSize, '\tTemperature: ',round(agent.temperature,ndigits=3), '\tMutation Rate: ',round(agent.mutationRate,ndigits=2),'|')
        print('============================================================================================================================')


    with open('./model/longRun/paramsfinal%f.txt'%round(bestGenome.fitness),'w') as f:
        #f.write(str(bestGenome)+str(bestGenome.fitness)+'\n')
        for x in range(4):
            f.write(str(agent.population[-x])+str(agent.population[-x].fitness)+'\n')
        f.close()

    with open('./model/longRun/logOverTime%f.txt'%round(bestGenome.fitness), 'w') as f:
        for key, val in log.items():
            valStr = ''
            for value in val:
                valStr += str(value)+'\t'
            f.write('%s\t%s\n'%(key, valStr))
        f.close()

    return bestGenome


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) > 0:
        training_type = int(args[0])
    if len(args) > 1:
        num_gens = int(args[1])
    if len(args) > 2:
        runtype = str(args[2])

    print('core count', multiprocessing.cpu_count())
    #print(training_type)
    #trainGP(1,0,8,3,2,'bench')
    if len(args) == 0:
        for x in range(3):
            trainGP(x, 63, mode='train')
    elif len(args) == 1:
        trainGP(training_type, 0.027)
    elif len(args) == 2:
        trainGP(training_type, num_gens, mode='train')
    else:
        trainGP(training_type, num_gens, mode=runtype)
    #trainGP()
