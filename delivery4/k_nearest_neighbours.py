import pandas as pd
import numpy as np
import warnings
from collections import Counter
import random
from math import sqrt

def eucdist(A, B):  # eucdist function simply calculates the distance between A and B points with euclidian distance formula.
    # A and B point's location for each dimension
    if len(A) != len(B):  # stored as vectors individually. So A =(Xa,Ya,Za,...), B = (Xb,Yb,Zb,...)
        print('length between vectors differ.')
        quit()
    length=len(A)
    mysum = 0
    for i in range(length):
        mysum = (A[i]-B[i])**2+mysum

    return sqrt(mysum)


def k_nearest_neighboors(data, predict, k=15): # This function returns a prediction for a test vector.
    # 'Data' is the training set and 'predict' is a vector from test set.
   if len(data) >= k: # Prediction is either 1(healthy) or 0(diseased).
       warnings.warn('K is set to a value less than total voting group')
   distances = []
   for group in data:
       for features in data[group]:
           euc_distance = eucdist(features,predict) # Distances and group of the train data which compared are stored and sorted. Votes contain those distances and groups.
           distances.append([euc_distance, group])  # vote_result is the most common elements taken from sorted votes array.

   votes =  [i[1] for i in sorted(distances)[:k]]
   vote_result = Counter(votes).most_common(1)[0][0]


   return vote_result


# Input Data manipulation.
df = pd.read_excel('dataset/DifferentiallyExpressedGenes.xlsx', index_col=None, header=None)
df_t = df.T
myData = df.drop(df.columns[0], axis=1)
myData = myData.T
print(myData)
myData_list = myData.astype(float).values.tolist()

# Train-Test Split

random.shuffle(myData_list)  # Shuffling the data randomly. Each time program works it shuffles random.
test_size = 0.3  # This variable controls the test and train size.
train_set = {0:[], 1:[]}  # 0 is disease, 1 is healthy sample
test_set = {0:[], 1:[]}
train_data = myData_list[:-int(test_size*len(myData_list))]  # Train and test data was obtained in this step.
test_data = myData_list[int(test_size*len(myData_list)):]

for i in train_data:  # With this for loops I separated the train and test data depending on their targets(1 and 0).
    train_set[i[0]].append(i[1:])  # so train_set and test_set are dictionaries consists of two group of lists.
    # One group is healthy, other is diseased.
for i in test_data:
    test_set[i[0]].append(i[1:])

print(train_set)

correct = 0
total = 0
TP=0
FN=0
TN=0
FP=0
k=3


for group in test_set:
    for testfeatures in test_set[group]:  # testfeatures is a vector of values indicating each dimension from test_set.
        vote = k_nearest_neighboors(train_set, testfeatures, k=3)  

        if group == 1 and vote == 1:
            TP += 1
            correct += 1
        elif group == 1 and vote == 0:
             FN += 1
        elif group == 0 and vote == 1:
             FP += 1
        elif group == 0 and vote == 0:
             TN += 1
             correct += 1

        total += 1


sensitivity = TP/(TP+FN)
specification = TN/(TN+FP)
precision = TP / (TP + FP)
f1_score = (2*precision*sensitivity)/(precision+sensitivity)

print('Accuracy: ' + str(correct/total))
print('Sensitivity: ' + str(sensitivity))
print('Specification: ' + str(specification))
print('Precision: ' + str(precision))
print('F1 Score: ' + str(f1_score))

