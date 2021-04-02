#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.


import operator, random, sys, multiprocessing, numpy, pickle, os, time
import pygraphviz as pgv

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

#LOAD DATASET
FILEDIR = "dataset2"

FILENOISE = "15"
SIZE = 128  #image size
FILEPREFIX = "LENA"

datanoisename = os.path.join(FILEDIR,FILEPREFIX+'-data-noise-'+FILENOISE+'.txt')
dataclean = numpy.loadtxt( os.path.join(FILEDIR,FILEPREFIX+'-data-clean.txt') )
datanoise = numpy.loadtxt( datanoisename  )


#GP PARAMENTERS: crossover prob, mutation prob, number of generation, population size: change at will!
CXPB, MUTPB, NGEN, POPSIZE, MAXDEPTH = 0.3, 0.4, 100, 400, 5

#LOGGING PARAMETERS
FREQ_SAVE = 10


# Define new functions

#TODO: define function to be used by GP. These will be the branches of GP.

def safeadd(a,b):
    s = a + b
    if s >= 255:
        s =255
    return s
    
def safesub(a, b):
    s = a - b
    if s < 0:
        s = 0
    return s
    
def safemul(a, b):
    s = a * b
    if s < 0:
        s = 0
    if s > 255:
        s = 255
    return s

def safediv(a, b):
    if b == 0:
        return 0
    try:
        s = a / b
    except ZeroDivisionError:
        return 0
    if s < 0:
        return 0
    if s > 255:
        return 255

def avg2(a, b):
    try:
        return (a + b) / 2
    except ZeroDivisionError:
        return 0

def avg3(a, b, c):
    try:
        return (a + b + c) / 3
    except ZeroDivisionError:
        return 0

# create a "PrimitiveSet": add the previously created function to your pool
# define the number of "arguments", the variables that represent instances
# of the dataset in the GP tree
# create constants and arguments, the "leaves" of the GP.

pset = gp.PrimitiveSet("MAIN",9)
pset.addPrimitive(safeadd,2)
pset.addPrimitive(safesub,2)
pset.addPrimitive(safediv,2)
pset.addPrimitive(safemul,2)
pset.addPrimitive(avg2,2)
pset.addPrimitive(avg3,3)

pset.renameArguments(ARG0='x0y0')
pset.renameArguments(ARG1='x0y')
pset.renameArguments(ARG2='x0y1')
pset.renameArguments(ARG3='xy0')
pset.renameArguments(ARG4='xy')
pset.renameArguments(ARG5='xy1')
pset.renameArguments(ARG6='x1y0')
pset.renameArguments(ARG7='x1y')
pset.renameArguments(ARG8='x1y1')
pset.addEphemeralConstant('rand101', lambda: random.randint(1,30))

# define Fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# define GP individual type
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

#define specific GP options like how to...
# generate an individual
#          a population
# how to evaluate it
# how to perform the SELECTION
# mating operator
# how to generate a tree for mutation
# mutation operator

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# define the fitness function: how to evaluate an individual
def evalImage(individual, points):
    func = toolbox.compile(expr=individual)
    
    sqerror=0
    
    for p in points:
        try:
            sqerror += numpy.power( ( func(
            datanoise[p[0]-1,p[1]-1],
            datanoise[p[0]-1,p[1]],
            datanoise[p[0]-1,p[1]+1],
            datanoise[p[0],p[1]-1],
            datanoise[p],
            datanoise[p[0],p[1]+1],
            datanoise[p[0]+1,p[1]-1],
            datanoise[p[0]+1,p[1]-1],
            datanoise[p[0]+1,p[1]+1],
            ) - dataclean[p]), 2)
        except:
            sqerror += 1000.0
    
    return sqerror / len(points),
    
    
        
toolbox.register("evaluate", evalImage, points=[(y,x) for y in range(1,SIZE-1) for x in range(1,SIZE-1)])
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate",gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
toolbox.register("mutate",gp.mutUniform,expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value = MAXDEPTH))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value = MAXDEPTH))

# ENABLE PARALLEL PROCESSING
pool = multiprocessing.Pool()
toolbox.register("map", pool.map)

#main program
def main():
    
    #create unique location where to store the experiment
    t = time.localtime()
    TIMEPOST = str(t.tm_year) + str(t.tm_mon) + str(t.tm_mday) + "_"+ str(t.tm_hour) + str(t.tm_min) + str(t.tm_sec)
    OUTPUTDIR = FILEPREFIX + "_" + TIMEPOST
    print (OUTPUTDIR)
    print ("creating directory " + OUTPUTDIR)
    os.mkdir(OUTPUTDIR)


    # Start a new evolution with an initial random population of individuals
    population = toolbox.population(n=POPSIZE)
    start_gen = 0
    #save the best of each generation: HallOfFame!
    halloffame = tools.HallOfFame(maxsize=1)
    #enable advanced logging
    logbook = tools.Logbook()

    #collect statistics about evolution
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    #MAIN LOOP THAT WILL DRIVE THE EVOLUTION
    for gen in range(start_gen, NGEN): 
        
        # for each generation....
        # import population
        population = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb = MUTPB)
        # check that every individual is valid (size constraints!)
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        # evaluate and check fitness for each individual
        fitnesses = toolbox.map(toolbox.evaluate,invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # update hall of fame
        halloffame.update(population)
        # update statistics
        record = mstats.compile(population)
        # update logbook
        logbook.record(gen=gen,evals=len(invalid_ind), **record)
        print (logbook.stream)
        #print "\tBEST IND", halloffame[0]
        
       
        
        #select individuals from current population that will generate the next one
        population = toolbox.select(population, k=len(population))
                
        #save and plot current status
        if gen % FREQ_SAVE == 0 or gen == NGEN-1:
            #saving logbook
            print ("*************************** SAVING LOGBOOK")
            cp = dict(logbook=logbook)
            with open(os.path.join(OUTPUTDIR, FILEPREFIX + "_logbook.pkl"), "wb") as cp_file:
                pickle.dump(cp, cp_file)
                
            with open(os.path.join(OUTPUTDIR, FILEPREFIX + "_best.txt"), "w") as bestfile:
                bestfile.write(str(halloffame[0]))
            
            cp = dict(best=str(halloffame[0]), datanoisename=datanoisename, time=TIMEPOST )
            with open(os.path.join(OUTPUTDIR, FILEPREFIX + "_best.pkl"), "wb") as bestfile:
                pickle.dump(cp, bestfile)
                
            #saving pdf graph of the best tree up to now
            nodes, edges, labels = gp.graph(halloffame[0])

            g = pgv.AGraph()
            g.add_nodes_from(nodes)
            g.add_edges_from(edges)
            g.layout(prog="dot")

            for i in nodes:
                n = g.get_node(i)
                n.attr["label"] = labels[i]
                
            g.draw( os.path.join(OUTPUTDIR,FILEPREFIX + "_" + str(gen) + ".pdf") )
                                


    print ('-------------------------------')
    print ('-------------------------------')
    print ('-------------------------------')
    print ('----- EVOLUTION FINISHED ------')
    print ('BEST SOLUTION:\n')
    print (halloffame[0])
    print ('\n')

    
    return None
    

if __name__ == "__main__":
    print ("Starting GP for", FILEPREFIX, "...")
    main()

    
    sys.exit()
