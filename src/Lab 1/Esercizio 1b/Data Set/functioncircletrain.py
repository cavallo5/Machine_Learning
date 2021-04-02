import arff, numpy as np



dataset = arff.load(open('circletrain.arff','r'))
data = np.array(dataset['data'])

newData = []
def modifyData(row):

    line = []
    x2 = row[0]*row[0]
    y2 = row[1]*row[1]
    d = x2 + y2
    line.append(d)
    line.append(row[2])
    newData.append(line)

    


for rows in dataset['data']:
    modifyData(rows)


import csv

with open ('functioncircletrain.txt','w') as f:
    wr = csv.writer(f, delimiter=',')
    wr.writerows(newData)
    


