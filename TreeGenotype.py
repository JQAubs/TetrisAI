import random
# GPac primitives
# internal_nodes = {'+','-','*','/','RAND'}
# leaf_nodes = {'G','P','W','F','#.#'}
# 'RAND(%s,1+%s)'%(lowerStr, upperStr)
def avg(a,b):
    return (a+b)/2

class treeNode():



    def __init__(self, parent = None, left = None, right = None, data = None, nType = None, height = 0, in_nodes=['+','-','*','/','min','max','avg'], le_nodes=['A','L','H','C','B','#.#']):
        self.parent = parent
        self.left = left
        self.right = right
        self.data = data
        self.height = height
        self.nType = nType
        self.internal_nodes = in_nodes
        self.leaf_nodes = le_nodes

    def __str__(self):
        string = ''
        if self.data == '/':
            if not self.left is None:
                string += self.left.strHelp()
            string += str(self.data)
            if not self.right is None:
                string += '(1+abs(int(%s)))' %  self.right.strHelp()
        elif self.data == 'RAND':
            string+='RAND(1+abs(int(%s)))' % self.left.strHelp()
        elif self.data == 'min':
            string += 'min('
            string += self.left.strHelp()+','
            string += self.right.strHelp()+')'
        elif self.data == 'max':
            string += 'max('
            string += self.left.strHelp()+','
            string += self.right.strHelp()+')'
        elif self.data == 'avg':
            string += '(('
            string += self.left.strHelp()+'+'
            string += self.right.strHelp()+')/2)'
        else:
            if not self.left is None:
                string += self.left.strHelp()
            string += str(self.data)
            if not self.right is None:
                string += self.right.strHelp()
        return string



    def strHelp(self):
        string = ''
        if self.data in self.internal_nodes and self.data != 'RAND' and self.data != 'min' and self.data != 'max':
            string += '('
        if self.data == '/':
            if not self.left is None:
                string += self.left.strHelp()
            string += str(self.data)
            if not self.right is None:
                string += '(1+abs(int(%s)))' %  self.right.strHelp()
        elif self.data == 'RAND':
            string+='RAND(1+abs(int(%s)))' % self.left.strHelp()
        elif self.data == 'min':
            string += 'min('
            string += self.left.strHelp()+','
            string += self.right.strHelp()+')'
        elif self.data == 'max':
            string += 'max('
            string += self.left.strHelp()+','
            string += self.right.strHelp()+')'
        elif self.data == 'avg':
            string += '('
            string += self.left.strHelp()+'+'
            string += self.right.strHelp()+')/2'
        else:
            if not self.left is None:
                string += self.left.strHelp()
            string += str(self.data)
            if not self.right is None:
                string += self.right.strHelp()
        if self.data in self.internal_nodes and self.data != 'RAND' and self.data != 'min' and self.data != 'max':
            string += ')'
        return string

    def evaluate(self, state):
        values = dict()
        #print(self.leaf_nodes)
        #print(type(self.leaf_nodes))
        for item in self.leaf_nodes:
            #print(type(item))
            if item == '#.#':
                continue
            values[item] = state[item]
        values['RAND'] = random.randrange
        #print(state)
        #print(values)
          # {'G': state['G'], 'P': state['P'], 'W': state['W'], 'F': state['F'], 'RAND': random.randint}

        return eval(str(self),values)

    def printHelp(self):
        string = ''
        for _ in range(self.height):
            string += '|'
        string += str(self.data)+"\n"
        if not self.left is None:
            string += self.left.printHelp()
        if not self.right is None:
            string += self.right.printHelp()
        return string

    def __len__(self):
        if self.left is None and self.right is None:
            return 1
        else:
            return 1 + len(self.left) + len(self.right)

    def copyHelp(self, original):
        self.data = original.data
        self.nType = original.nType
        self.height = original.height
        if original.left != None:
            self.left = treeNode(parent = self, in_nodes = self.internal_nodes, le_nodes = self.leaf_nodes).copyHelp(original.left)
        if original.right != None:
            self.right = treeNode(parent = self ,in_nodes = self.internal_nodes, le_nodes = self.leaf_nodes).copyHelp(original.right)
        return self

    def randTree(self):
        if random.random() > 0.6:
            self.full()
        else:
            self.grow()

    def grow(self, depth, prev = None):
        #print(prev)
        # edge case for head node
        if prev is None:
            self.height = 0
        else:
            self.height = prev.height + 1
        #print(self.height)

        if self.height < depth-1:
            # if > .5 then add an internal node else add leaf
            if random.random() > 0.3:
                self.ntype = 0
                self.data = random.choice(self.internal_nodes)
                self.left = treeNode(parent = self, in_nodes = self.internal_nodes, le_nodes = self.leaf_nodes).grow(depth, self)
                self.right = treeNode(parent = self, in_nodes = self.internal_nodes, le_nodes = self.leaf_nodes).grow(depth, self)
                return self

            else:
                self.ntype = 1
                selected = random.choice(self.leaf_nodes)
                if selected == '#.#':
                    selected = round(random.triangular(-5,5),ndigits=4)
                if selected == 0:
                    selected = round(random.triangular(1,2),ndigits=4)
                self.data = selected
                return self
        #must be a leaf because the depth = the max height
        else:
            self.ntype = 1
            selected = random.choice(self.leaf_nodes)
            if selected == '#.#':
                selected = round(random.triangular(-5,5),ndigits=4)
            self.data = selected
            return self

    def full(self, depth, prev = None):
         # edge case for head node
        if prev is None:
            self.height = 0
        else:
            self.height = prev.height + 1

        #if not at the max height keep adding internals
        if self.height < depth-1:


            self.ntype = 0
            self.data = random.choice(self.internal_nodes)
            self.left = treeNode(parent = self, in_nodes = self.internal_nodes, le_nodes = self.leaf_nodes).full(depth, self)
            self.right = treeNode(parent = self, in_nodes = self.internal_nodes, le_nodes = self.leaf_nodes).full(depth, self)
            return self

        #must be a leaf because the depth = the max height
        else:
            self.ntype = 1
            selected = random.choice(self.leaf_nodes)
            if selected == '#.#':
                selected = round(random.triangular(0.0001,5), ndigits = 4)
            self.data = selected
            return self

    def getRandom(self):
        if random.random() < 0.3 or self.left == None:
            return self
        else:
            if random.random() > 0.5:
                return self.left.getRandom()
            else:
                return self.right.getRandom()

    def getNode1(self, goal, index = 0):
        if isinstance(index, int):
            if index == 0:
                if self.left is None:
                    if goal == 0:
                        return self
                    else:
                        return 1
                else:
                    index = self.left.getNode(goal, 0)
                    if not isinstance(index, int):
                        return index
                    if index == goal:
                        return self
                    else:
                        return self.right.getNode(goal, index+1)
            else:
                if self.left is None:
                    if goal == index:
                        return self
                    else:
                        return index + 1
                else:
                    index = self.left.getNode(goal, index)
                    if not isinstance(index, int):
                        return index
                    if index == goal:
                        return self
                    else:
                        return self.right.getNode(goal, index+1)
        else:
            return index

    def getNode(self,index,goal):

        if self.left == None:
            if goal == index:
                return index, self
            else:
                return index + 1, None
        else:
            index, Node = self.left.getNode(index,goal)
            if Node == None:
                if goal == index:
                    return index, self
                else:
                    return self.right.getNode(index+1, goal)
            else:
                return index, Node

    def trim(self, maxH, height = 0):
        #print(height, max)
        self.height = height

        if self.data in self.internal_nodes and self.left == None:
            self.left = None
            self.right = None
            self.data = random.choice(self.leaf_nodes)
            if self.data == '#.#':
                self.data = round(random.triangular(0.001,5),ndigits=4)

        if height == maxH or self.left == None:
            self.left = None
            self.right = None
            self.data = random.choice(self.leaf_nodes)
            if self.data == '#.#':
                self.data = round(random.triangular(0.001,5),ndigits=4)

        if self.left != None:
            self.left.trim(maxH, height+1)

        if self.right != None:
            self.right.trim(maxH, height+1)

        return self
        #print('after',height, max)

class treeGenotype:

    def __init__(self, depth_limit = 3, gene = None, popSize=random.randint(20,50), tag = 'og', influx=random.randint(8,20),mr = random.random(), temp=random.randint(0,10),in_nodes=['+','-','*','/','max','min','avg'], le_nodes=['A','L','H','C','B','M','#.#']):
        self.fitness = 0
        self.gene = gene
        self.maxDepth = depth_limit
        self.popSize = popSize
        self.influx = influx
        self.temperature = temp
        self.mutRate = mr
        self.tag = tag
        self.internal_nodes = in_nodes
        self.leaf_nodes = le_nodes
        self.selectGene()

    def __str__(self):
        return str(self.gene)

    def __len__(self):
        return(len(self.gene))

    def __eq__(self, other):
        return str(self.gene) == str(other.gene)

    def selectGene(self):
        if random.random() > .5:
            self.randomFull()
        else:
            self.randomGrow()

    def avg(self,a,b):
        return (a+b)/2

    def evaluate(self, state):
        if isinstance(self.gene, str):
            values = dict()
            for item in self.leaf_nodes:
                if item == '#.#':
                    continue
                values[item] = state[item]
            values['RAND'] = random.randrange
            return eval(self.gene, values)
        return self.gene.evaluate(state)

    def copyGene(self):
        copyHead = treeNode(data = self.gene.data, nType = self.gene.nType, in_nodes = self.internal_nodes, le_nodes = self.leaf_nodes)
        if self.gene.left != None:
            copyHead.left = treeNode(parent = copyHead, in_nodes = self.internal_nodes, le_nodes = self.leaf_nodes).copyHelp(self.gene.left)
        if self.gene.right != None:
            copyHead.right = treeNode(parent = copyHead, in_nodes = self.internal_nodes, le_nodes = self.leaf_nodes).copyHelp(self.gene.right)
        return copyHead


    def randomFull(self):
        self.gene = treeNode(in_nodes = self.internal_nodes, le_nodes = self.leaf_nodes)
        self.gene.full(self.maxDepth,None)

    def randomGrow(self):
        self.gene = treeNode(in_nodes = self.internal_nodes, le_nodes = self.leaf_nodes)
        self.gene.grow(self.maxDepth,None)

    def getNode(self, goal, index = 0):
        copy = treeNode(data = self.gene.data, nType = self.gene.nType, in_nodes = self.internal_nodes, le_nodes = self.leaf_nodes)
        copy.gene = self.copyGene()
        ind, node = copy.gene.getNode(index, goal)
        return node

    def trim(self):
        self.gene = self.gene.trim(self.maxDepth)

    def recombine(self, mate, **kwargs):
        child = self.__class__(popSize = int(self.avg(self.popSize, mate.popSize)), influx =int(self.avg(self.influx, mate.influx)), temp = self.avg(self.temperature, mate.temperature), mr =self.avg(self.mutRate, mate.mutRate), tag = self.tag+'-1C')
        child.gene = self.copyGene()
        child.tag += '-RC'
        index = random.randint(0,len(self)-1)
        nodeOld = child.getNode(index)
        index = random.randint(0,len(mate)-1)
        nodeNew = mate.getNode(index)

        if nodeNew.nType == 0 and nodeOld.nType == 1:
            count = 0
            while not nodeNew.nType == nodeOld.nType:
                nodeNew = mate.gene.getRandom()
                count += 1
                if count > 10:
                    return child
            nodeNew.height = nodeOld.height
        if nodeNew.left != None:
            nodeOld.left = treeNode(parent = nodeOld, in_nodes = self.internal_nodes, le_nodes = self.leaf_nodes).copyHelp(nodeNew)
            nodeOld.right = treeNode(parent = nodeOld, in_nodes = self.internal_nodes, le_nodes = self.leaf_nodes).copyHelp(nodeNew)
        else:
            nodeOld.left = None
            nodeOld.right = None

        if nodeOld.data == '#.#':
            nodeOld.data = round(random.triangular(0.0001,5), ndigits = 4)
        nodeOld.data = nodeNew.data
        nodeOld.nType = nodeNew.nType

        child.trim()
        return child


    def mutate(self, temp, **kwargs):
        copy = self.__class__(popSize = self.popSize, influx =self.influx, temp = self.temperature, mr =self.mutRate, tag = self.tag+'-1C')
        copy.gene = self.copyGene()
        copy.tag += '-MT'
        index = random.randint(0,len(self)-1)
        #print('index-self',index, self)
        randomNode = self.getNode(index)
        #print('node',randomNode)


        if randomNode.nType == 1:
            mutType = random.randint(0,2)
            #grow fill
            if mutType == 0:
                randomNode.data = random.choice(self.internal_nodes)
                randomNode.left = randomNode.left.grow(depth = self.maxDepth, prev = randomNode)
                randomNode.right = randomNode.right.grow(depth = self.maxDepth, prev = randomNode)
            #full fill
            elif mutType == 1:
                randomNode.data = random.choice(self.internal_nodes)
                randomNode.left = randomNode.left.full(depth = self.maxDepth, prev = randomNode)
                randomNode.right = randomNode.right.full(depth = self.maxDepth, prev = randomNode)
            # new value
            else:
                randomNode.data = random.choice(self.leaf_nodes)

        else:
            randomNode.data = random.choice(self.internal_nodes)
        copy.trim()
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


    def printTree(self):
        string = str(self.gene.data)+"\n"
        if not self.gene.left is None:
            string += self.gene.left.printHelp()
        if not self.gene.right is None:
            string += self.gene.right.printHelp()
        return string

    @classmethod
    def initialization(cls, mu, *args, **kwargs):
        depth_limit = kwargs['depth_limit']
        internal_nodes = kwargs['internal_nodes']
        leaf_nodes = kwargs['leaf_nodes']
        #print(internal_nodes, leaf_nodes)
        population = [cls(depth_limit, in_nodes = internal_nodes, le_nodes = leaf_nodes) for _ in range(mu)]
        return population
