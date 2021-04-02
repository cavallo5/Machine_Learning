import numpy as np
from PIL import Image


with open("kmeans_risultato.txt") as inp:
    values =  list((line.strip().split() for line in inp))    


matrix = []
for row in values:
    row = list(map(float,row))
    row = row[1:]
    matrix.append(row)

matrix = list(map(list, zip(*matrix)))
#in matrix ci sono 10 liste che rappresentano i valori di ogni cluster li prendo 13x8 e ci faccio l'immagine
for ind, centroid in enumerate (matrix):
    imgList = [ x*255 for x in centroid]
    imgList = list(map(int,imgList))
    array = np.array(imgList).reshape((13,8)).astype(np.uint8)  
    img = Image.fromarray(array)
    img.save('centroid'+ str(ind)+'.png')
