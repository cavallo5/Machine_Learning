import arff, numpy as np



dataset = arff.load(open('circletest.arff', 'r'))
data = np.array(dataset['data'])

newData = []
def modifyData(row):

    if row[0] >= -0.88 and row[0] <= 0.88 and row[1] >= -0.88 and row[1] <= 0.88:
        if row[2] != 'c':
            row[2] = 'c'
    newData.append(row)

    


for rows in dataset['data']:
    modifyData(rows)


import csv

with open ('circletest.txt','w') as f:
    wr = csv.writer(f, delimiter=',')
    wr.writerows(newData)
    


