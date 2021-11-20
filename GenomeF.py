import random

class genome:

    def __init__(self, parameters = [random.random() for x in range(6)] , p = random.randint(10,30), i=random.randint(15,30), t=random.randint(1,10), m=random.random(), fit = 0):
        self.params = parameters
        self.popSize = p
        self.influx = i
        self.mutRate = m
        self.temperature = t
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
        return True
        #return self.params[0] == other.params[0] and self.params[1] == other.params[1] and self.gamma == other.gamma and self.phi == other.phi and self.zeta == other.zeta and self.fitness == other.fitness and self.popSize == other.popSize and self.influx == other.influx and self.temp == other.temp and self.mutRate == other.mutRate and self.delta == other.delta

    def copy(self):
        return genome(self.params.copy(),self.popSize,self.influx,self.temperature,self.mutRate)

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

        return genome(newParams, p = int(self.avg(self.popSize, mate.popSize)), i =int(self.avg(self.influx, mate.influx)), t = self.avg(self.temperature, mate.temperature), m =self.avg(self.mutRate, mate.mutRate))

    #swaps n genes of self with corresponding genes in mate
    def nCrossBaby(self, mate):
        newParams = self.params.copy()
        n = random.randint(1,len(self.params)-1)
        swaps = random.sample(range(0,6),k=n)
        for index in swaps:
            newParams[index] = mate.params[index]
        return genome(newParams, p = int(self.avg(self.popSize, mate.popSize)), i =int(self.avg(self.influx, mate.influx)), t = self.avg(self.temperature, mate.temperature), m =self.avg(self.mutRate, mate.mutRate))

    #averages all values of both parents
    def averageBaby(self, mate):
        return genome(parameters = [self.avg(self.params[x] , mate.params[x]) for x in range(len(self.params))], p = int(self.avg(self.popSize, mate.popSize)), i =int(self.avg(self.influx, mate.influx)), t = self.avg(self.temperature, mate.temperature), m =self.avg(self.mutRate, mate.mutRate))

    #picks any number of parameters randomly and changes them based on temp
    def mutate(self, temp):
        copy = self.copy()
        paramsMutated = random.randint(1,len(copy.params))
        mutants = random.sample([x for x in range(len(copy.params))], k=paramsMutated)
        for val in mutants:
            value = random.uniform(-2,2)/temp
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
        if copy.temperature + value > 1:
            copy.temperature += value
        elif copy.temperature + value//2 > 1:
            copy.temperature += value//2
        else:
            copy.temperature = 1
        return copy
