from TreeGenotypeF import treeGenotype
from TetrisGameAIF import TetrisAI
from GenomeF import genome
from math import inf
import random

Hall_of_Fame = (([0.588,0.7982, 0.27013, 0.16258, 0.58243, -0.02852],310), #  310.42  190     365.8195769 a:0.29382 b; -0.11575
                ([0.29382, 0.24452, 0.13921, 0.16258, 0.57649, -0.02852],486), #   489.3   379.5	  455.5812278
                ([0.52504, 0.34867, 0.23395, 0.60486, 1.01676, -0.02544],400), #   400.54	293	    335.7919795
                ([0.66681, 0.2908, 0.41964, 0.47442, 0.82171, -0.03511],368 ),#    368.02	279	    339.9396399
                ([0.63698, 0.14493, 0.48567, 0.86632, 0.96635, 0.02816],298 ),#    298.04	244	    197.0456241
                ([0.65836, 0.17403, 0.41379, 0.85921, 0.96635, 0.05371],352 ),#    352.78	301.5	  280.3268712
                ([0.51012, 0.56257, 0.37175, 0.16006, 0.8074, 0.01967], 210 ),#     210.42	141.5	  209.2896723
                ([0.40016, 0.59975, 0.15026, 0.22425, 0.71479, 0.02291], 199),#     199.56	171	    122.4131779
                ([0.48109, 0.24348, 0.30957, 0.1519, 0.99108, 0.00827], 396 ), #    396.08	266.5	  323.5307663
                ([0.19846, 0.72592, 0.34477, 0.1519, 0.99108, 0.00827], 377 ),  )   # a:0.82992 b:0.24348


class Rat:
    #0.9597864099655763 0.9711718414532963 0.022069810925646416 0.29238468508600435 0.38691042015197374
    def __init__(self, genomeType, tpa=16, pop=15, frames_per_second=100, mode = 'train'):
        self.numGames = 0
        self.GType = genomeType
        self.testPerAgent = tpa
        self.popAvg = 0
        self.goalPos = None
        self.popSize = pop
        self.population = None
        self.currentGenome = None
        #print('gp',mode)
        if mode == 'train' or 'hof' or 'bench':
            if mode == 'bench':
                print('benchmarking...')
                self.game = TetrisAI(frames_per_second, bench=True)
            else:
                #print('training...')
                self.game = TetrisAI(frames_per_second, bench=False)
        else:
            from TetrisGameAI2 import TetrisAI as TesterAI

            self.game = TesterAI(frames_per_second)

        if self.GType == 2:
            self.hyperParams = [-0.82992, -0.01621, -0.3608, 0.1605, 0.95009, -0.04574]
        if mode == 'train':
            self.generatePopulation()
        else:
            self.hallOfFameStart()

    def hallOfFameStart(self):
        self.population = [genome(parameters=gene[0], p = random.randint(10,30), i=random.randint(15,30), t=random.randint(1,5), m=random.random(), fit = gene[1]) for gene in Hall_of_Fame]

    #generated population of genomes with parndom params and assignt it to population
    def generatePopulation(self):
        self.population = self.generateRandom(self.popSize)

    #generate a single random genome
    def generateRandom(self, size):
        if self.GType == 1:
            #print('making GP')
            return [genome(parameters = [random.uniform(0,10) for x in range(6)], p = random.randint(10,30), i=random.randint(15,30), t=random.randint(1,5), m=random.random()) for _ in range(size)]
        elif self.GType == 2:
            #print('making GP2')
            return  [treeGenotype(treeType = 2, hyper = self.hyperParams) for _ in range(size)]
        else:
            #print('making GP3')
            return [treeGenotype() for _ in range(size)]
