import arff
import numpy as np



dataset = arff.load('circletest.arff')
new_data = []
for row in dataset:
    if row[0] >= -0.88 and row[0] <= 0.88 and row[1] >= -0.88 and row[1] <= 0.88:
        new_data.append([row[0], row[1], 'c'])
    else:
        new_data.append(row)

arff.dump('new_circletest.arff', new_data, relation="circle", names=['x', 'y', 'class'])
