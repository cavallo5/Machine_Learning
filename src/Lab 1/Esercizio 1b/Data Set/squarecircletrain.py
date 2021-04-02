import arff, numpy as np



dataset = arff.load(open('circletrain.arff', 'r'))
data = np.array(dataset['data'])

newData = []
def modifyData(row):

    line = []
    x = row[0]*row[0]
    y = row[1]*row[1]
    line.append(x)
    line.append(y)
    line.append(row[2])
    newData.append(line)

    


for rows in dataset['data']:
    modifyData(rows)


import csv

with open ('squarecircletrain.txt','w') as f:
    wr = csv.writer(f, delimiter=',')
    wr.writerows(newData)
    


