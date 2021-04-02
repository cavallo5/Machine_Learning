'''
Create image from GP output.
copy and paste the string representation of the output.
IMPORTANT: terminal set and operators defined here must be the same as the ones used during training.

'''
import operator, random, sys, multiprocessing, numpy, pickle, os

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from PIL import Image
import pygraphviz as pgv

#import  pickle, sys

SIZE = 128 #dimensione lato immagine

#LOAD DATASET

if len(sys.argv) != 2:
    print (" USAGE: python create_img_from_gp.py checkpoint_name.pkl")
    print (" ex: python create_img_from_gp.py GRIGIO_2016121_1426/GRIGIO_best.pkl ")

    sys.exit()
    
    
with open(sys.argv[1], "rb") as cp_file:
    cp = pickle.load(cp_file)
    BEST = cp["best"]
    datanoisename = cp["datanoisename"]
    time = cp["time"]

datanoise = numpy.loadtxt(datanoisename)
OUTIMAGE = datanoisename + "_" + time + ".bmp"
OUTPDF = datanoisename + "_" + time + ".pdf"


# Define functions
#ATTENZIONE: IL PRIMITIVESET DEVE ESSERE IDENTICO ALLA FASE DI TRAINING
#ATTENZIONE: IL PRIMITIVESET DEVE ESSERE IDENTICO ALLA FASE DI TRAINING
#ATTENZIONE: IL PRIMITIVESET DEVE ESSERE IDENTICO ALLA FASE DI TRAINING


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


pset = gp.PrimitiveSet("MAIN",9)
pset.addPrimitive(safeadd,2)
pset.addPrimitive(safesub,2)
pset.addPrimitive(safemul,2)
pset.addPrimitive(safediv,2)
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


# create object from string representation
besttree= gp.PrimitiveTree.from_string(BEST, pset)
func = gp.compile(besttree,pset)

#create a new image with the filtered image
points=[(y,x) for y in range(1,SIZE-1) for x in range(1,SIZE-1)]
newimg = numpy.zeros((SIZE,SIZE),dtype=int)
for p in points:
    newimg[p] = func( datanoise[p[0]-1,p[1]-1],  datanoise[p[0]-1,p[1]],  datanoise[p[0]-1,p[1]+1],  datanoise[p[0],p[1]-1], datanoise[p], datanoise[p[0],p[1]+1], datanoise[p[0]+1,p[1]-1], datanoise[p[0]+1,p[1]], datanoise[p[0]+1,p[1]+1] )

#save
print ("saving result image ", OUTIMAGE, "...")
img_noise = Image.fromarray(numpy.uint8(newimg))
img_noise.save(OUTIMAGE)

# PLOTTING
best_tree= gp.PrimitiveTree.from_string(BEST,pset)
nodes, edges, labels = gp.graph(best_tree)

#save pdf
g = pgv.AGraph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
g.layout(prog="dot")

for i in nodes:
    n = g.get_node(i)
    n.attr["label"] = labels[i]

g.draw(OUTPDF)

