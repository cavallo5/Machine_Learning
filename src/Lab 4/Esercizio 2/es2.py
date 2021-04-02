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
 
import operator
import math
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap import gp

        
def generate(size, pmin, pmax, smin, smax):
	part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
	part.speed = [random.uniform(smin, smax) for _ in range(size)]
	part.smin = smin
	part.smax = smax
	return part

def updateParticle(part, best, phi1, phi2):
	u1 = (random.uniform(0, phi1) for _ in range(len(part)))
	u2 = (random.uniform(0, phi2) for _ in range(len(part)))
	v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
	v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
	part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
	for i, speed in enumerate(part.speed):
		if abs(speed) < part.smin:
			part.speed[i] = math.copysign(part.smin, speed)
		elif abs(speed) > part.smax:
			part.speed[i] = math.copysign(part.smax, speed)
	part[:] = list(map(operator.add, part, part.speed))
	for i in range(len(part)):
		if part[i] > 128:
			part[i] = -128 + (part[i] % 128)
		elif part[i] < -128:
			part[i] = 128 + (part[i] % -128)

def our_func_to_eval(p):
	x = p[0]
	y = p[1]
	z = p[2]
	return ((1 - math.cos(2 * math.pi / (1 + math.exp(-(x-125)**2 - (y-1.27)**2)))) / (1+z**2),)
	

	
def main():
	random.seed(318)
	
	pop = toolbox.population(n=100)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	logbook = tools.Logbook()
	logbook.header = ["gen", "evals"] + stats.fields
	GEN = 10000
	best = None
	for g in range(GEN):
		for part in pop:
			part.fitness.values = toolbox.evaluate(part)
			if not part.best or part.best.fitness < part.fitness:
				part.best = creator.Particle(part)
				part.best.fitness.values = part.fitness.values
			if not best or best.fitness < part.fitness:
				best = creator.Particle(part)
				best.fitness.values = part.fitness.values
		for part in pop:
			toolbox.update(part, best)

		# Gather all the fitnesses in one list and print the stats
		logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
		#print(logbook.stream)
	print( best, "|", best.fitness.values)
	return pop, logbook, best

if __name__ == "__main__":
	version = "2,3"
	with open("log_es"+version+".txt", 'w') as f:
		x = []
		y = []
		z = []
		c = []
		for w in range(1, 10, 2):
			creator.create("FitnessMax", base.Fitness, weights=(w*0.1,))
			creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, smin=None, smax=None, best=None)
			for p1 in range(0, 20, 4):
				for p2 in range(0, 20, 4):
					toolbox = base.Toolbox()
					toolbox.register("particle", generate, size=3, pmin=-128, pmax=128, smin=-32, smax=32)
					toolbox.register("population", tools.initRepeat, list, toolbox.particle)
					toolbox.register("update", updateParticle, phi1=p1*0.1, phi2=p1*0.1)

					toolbox.register("evaluate", our_func_to_eval)
					_,_, best = main()
				
					x.append(float(p1*0.1))
					y.append(float(p2*0.1))
					c.append(float(w*0.1))
					z.append(best.fitness.values[0])
				
					f.write("w: "+str(float(w*0.1))+"\np1: "+str(float(p1*0.1))+"\np2: "+str(float(p2*0.1))+"\nbest: "+str(best)+" | "+str(best.fitness.values)+"\n\n")
		f.close()
	
	
	x = np.array(x)
	y = np.array(y)
	z = np.array(z)
	c = np.array(c)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel("phi1")
	ax.set_ylabel("phi2")
	ax.set_zlabel("best")
	img = ax.scatter(x, y, z, c=c, cmap=plt.get_cmap('YlOrRd'))
	fig.colorbar(img)

	plt.title("Risultati al variare di phi1, phi2 e weights")
	plt.savefig("img_es_"+version+"_3D.png")
	plt.close()

	plt.xlabel("phi1")
	plt.ylabel("best")
	plt.plot(x, z,'bo')

	plt.title("Risultati al variare di phi1, phi2 e weights")
	plt.savefig("img_es_"+version+"_phi1.png")
	plt.close()

	plt.xlabel("phi2")
	plt.ylabel("best")
	plt.plot(y, z,'bo')

	plt.title("Risultati al variare di phi1, phi2 e weights")
	plt.savefig("img_es_"+version+"_phi2.png")
	plt.close()
	
	plt.xlabel("weights")
	plt.ylabel("best")
	plt.plot(c, z,'bo')

	plt.title("Risultati al variare di phi1, phi2 e weights")
	plt.savefig("img_es_"+version+"_weights.png")
	plt.close()

