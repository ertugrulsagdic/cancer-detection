import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from pprint import pprint
from graphviz import Digraph
import matplotlib.pyplot as plt

# Train-Test split pandas df function
def train_test_split(data, testsize=0.3):
    train_data = data.sample(frac=(1-testsize), random_state=24)
    test_data = data.drop(train_data.index)
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()
    train_data.drop(columns=['index'], inplace=True)
    test_data.drop(columns=['index'], inplace=True)
    return train_data, test_data


# Data purity checker function # Checks if the labels are the same in a data group
def purity_checker(data, labelcolumn=-1):

    labels = data[:, labelcolumn]
    purity = np.unique(labels)

    if len(purity) == 1:
        return True
    else:
        return False


# Classifier function
def classify(data, labelcolumn = -1):
    labels = data[:, labelcolumn]
    unique_labels, counts_unique_classes = np.unique(labels, return_counts=True)
    index=counts_unique_classes.argmax() # returns the index of highest count

    return unique_labels[index]


# Potential Splits
def get_potential_splits(data):
    columncount=data.shape[1]
    potential_splits = {}

    for column_index in range(columncount-1):  # -1 for labels
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)

        for index in range(len(unique_values)):
            if index != 0:
                initval = unique_values[index]
                prevval = unique_values[index-1]
                potential_split = (initval + prevval)/2
                potential_splits[column_index].append(potential_split)

    return potential_splits


# Split Data Function
def split_data(data, attribute_column, split_value):
    condition = data[:,attribute_column] <= split_value
    slicedunder=data[condition]
    slicedover=data[(np.invert(condition))]

    return slicedover, slicedunder


# Lowest Overall Entropy Function to determine best split
def get_entropy(data, label_column=-1):
    labels = data[:, label_column]
    counts = np.unique(labels, return_counts=True)[1]
    probabilities = counts/counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


# Using overall entropy formula to compare data under-over the split
def get_overall_entropy(data_over, data_under):
    n_data_points = len(data_over) + len(data_under)
    p_data_under = len(data_under)/n_data_points
    p_data_over = len(data_over)/n_data_points
    overall_entropy = (p_data_under*get_entropy(data_under) + p_data_over*get_entropy(data_over))

    return overall_entropy


def determine_best_split(data, potential_splits):
    overall_entropy = 999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_over, data_under = split_data(data, attribute_column=column_index, split_value=value)
            current_overall_entropy = get_overall_entropy(data_over,data_under)
            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value



def get_feature_name(some_key):
    feature_name, _ = some_key.split(sep=' ≤')

    return feature_name

def get_features(tree_dict):
    all_features = []
    for tree_num in tree_dict.keys():
        tree = tree_dict[tree_num]
        features = features_from_tree(tree)
        all_features.append(features)
    return all_features


def features_from_tree(tree):
    features=[]
    if isinstance(tree, dict):
        for tree_key in tree.keys():
            feat_name=get_feature_name(tree_key)
            features.append(feat_name)
            next_tree =tree[tree_key]
            for i in range(len(next_tree)):
                next_tree =tree[tree_key][i]
                next_features = features_from_tree(next_tree)
                if len(next_features) > 0:
                    features.append(next_features[0])
    return features

def count_features(feature_list):
    feature_dict = {}
    for tree_features in feature_list:
        for feature in tree_features:
            if feature in feature_dict:
                feature_dict[feature] += 1
            else:
                feature_dict[feature] = 1

    return feature_dict

df = pd.read_excel('dataset/NotExposedControl.xlsx', index_col=None, header=None)

myData = df       
print(myData)                                                  
myData.columns = range(myData.shape[1])
myData = myData.transpose()
myData.columns = myData.iloc[0, :]
myData.drop(myData.index[0], inplace=True)  
print(myData)   
myData['label'] = myData.iloc[:, 0]
myData = myData.astype(float)
myData.drop(myData.columns[0], axis=1, inplace=True)
myData['label'] = myData['label'].astype(str)
myData['label'].replace({'1.0': 'healthy'}, inplace=True)
myData['label'].replace({'0.0': 'diseased'}, inplace=True)
rawdata = myData
  
print(rawdata)   

# Train-Test split
train_dataDF, test_dataDF = train_test_split(rawdata)

print(train_dataDF)
print(test_dataDF)

def random_train_data_eq(train_dataDF, randomGeneSize=0.10, HealthySizeFrac=0.80, DiseaseSizeFrac=0.05, random_seed=1):
    trainDF = train_dataDF
    labels = trainDF['label']
    trainDF.drop('label',axis=1, inplace=True)
    # Selecting genes
    trainDF = (trainDF[trainDF.columns.to_series().sample(frac=randomGeneSize, random_state=random_seed)])
    trainDF['label'] = labels
    train_dataDF['label'] = labels

    # Separating samples
    train_healthySamples=trainDF.loc[trainDF['label'] == 'healthy']
    train_diseaseSamples = trainDF.loc[trainDF['label'] == 'diseased']

    # Selecting healthy-disease samples
    train_healthySamples = train_healthySamples.sample(frac=HealthySizeFrac, random_state=random_seed)
    train_diseaseSamples = train_diseaseSamples.sample(frac=DiseaseSizeFrac, random_state=random_seed)
    column_data= train_diseaseSamples.columns
    r_train_array = (pd.concat([train_healthySamples, train_diseaseSamples]).reset_index()).drop(['index'], axis=1)
    r_train_array = r_train_array.values
    return r_train_array, column_data


test_array = test_dataDF.values
num_of_columns = test_array.shape[1]
example = test_dataDF.iloc[0, :num_of_columns-1]


# Main Algorithm
i = 0
seed = 1


def get_decision_tree(train_array,column_data, counter=0, classify_counter=0, min_samples=5, graphname=''):
    global tree_graph
    global i
    data = train_array

    # base case
    if purity_checker(data) or len(data)<min_samples:
        classification = classify(data)
        classify_counter_node = str(classify_counter)
        tree_graph[graphname].node(classify_counter_node, '%s' % (classification))
        is_classified = True
        return classification, classification,is_classified
    else:
        #  This part repeats until purity is reached
        counter += 1
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_under, data_over = split_data(data,split_column,split_value)

        # Sub tree creation
        column_name = column_data[split_column]
        question = "%s ≤ %s"%(column_name,split_value)
        sub_tree = {question: []}
        pre_question = str(question)

        tree_graph[graphname].node(pre_question, '<f0> False|<f1> %s ≤ %s|<f2> True' % (column_name, split_value))

        # Answers (recurs)
        yes_answer, yes_question, is_classified = get_decision_tree(data_under, column_data, counter, classify_counter=i,
                                                                    min_samples=min_samples, graphname=graphname)
        if is_classified:
            tree_graph[graphname].edges([('%s:f2' % pre_question, '%s' % i)])
            i = i+1
        else:
            tree_graph[graphname].edges([('%s:f2' % pre_question, yes_question)])

        no_answer, no_question, is_classified = get_decision_tree(data_over, column_data, counter, classify_counter=i,
                                                                  min_samples=min_samples, graphname=graphname)
        if is_classified:
            tree_graph[graphname].edges([('%s:f0' % pre_question, '%s' % i)])
            i = i+1
        else:
            tree_graph[graphname].edges([('%s:f0' % pre_question, no_question)])

        sub_tree[question].append(yes_answer)
        sub_tree[question].append(no_answer)
        is_classified = False

        return sub_tree, question, is_classified


# CREATE TREES
def classify_sample(example, tree):
    if not isinstance(tree, dict):
        answer = tree

        return answer
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split()
    if example[feature_name] <= float(value):
        answer = tree[question][1]
    else:
        answer = tree[question][0]
    if isinstance(answer, dict):
        return classify_sample(example, answer)
    if not isinstance(answer, dict):
        return answer


def determine_accuracy(test_dataDF, tree):
    testData = test_dataDF
    testData['classification'] = testData.apply(classify_sample, axis=1, args=(tree,))
    testData['classification_correct'] = testData.classification == testData.label
    accuracy = testData.classification_correct.mean()

    return accuracy


tree_graph = {}  # global


def create_RF(train_dataDF,test_dataDF,num_of_trees=10, accuracy_cutoff=0.50):
    seed = 1
    tree_dict = {}
    nm_of_trees_approved = 0

    while len(tree_dict) < num_of_trees:  # number of trees generated
        graphname = 'tree' + str(seed)
        tree_graph[graphname] = Digraph('structs', filename='structs_revisited.gv', node_attr={'shape': 'record'})
        train_array, column_data = random_train_data_eq(train_dataDF, randomGeneSize=0.03, HealthySizeFrac=0.40,
                                                        DiseaseSizeFrac=0.10, random_seed=seed)
        my_tree, question, is_classified = get_decision_tree(train_array, column_data=column_data, graphname=graphname)
        my_acc = determine_accuracy(test_dataDF, my_tree)

        if my_acc >= accuracy_cutoff:
            tree_dict[str(seed)] = my_tree
            nm_of_trees_approved += 1
        print('number of trees created: ' +str(seed))
        print('number of trees approved: ' +str(nm_of_trees_approved))
        seed += 1

    return tree_dict, seed, tree_graph


tree_dict, num_of_trees, tree_graph = create_RF(train_dataDF, test_dataDF, num_of_trees=8, accuracy_cutoff=0.90)

print(tree_dict)
print(tree_graph)



# FINAL CLASSIFICATION OF TEST DATA

def final_classification(test_dataDF,tree_dict):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    test_dataDF.drop(columns='classification_correct', inplace=True)
    For_Accuracy=pd.DataFrame()
    For_Accuracy['label']=test_dataDF['label']
    print('Predicting test Data...')
    for key in tree_dict.keys():
        my_tree=tree_dict[str(key)]
        tree_name = 'tree' + str(key)
        test_dataDF[tree_name] = test_dataDF.apply(classify_sample, axis=1, args=(my_tree,))  # put , at and of tuples
        For_Accuracy[tree_name] = test_dataDF.apply(classify_sample, axis=1, args=(my_tree,))
    prediction=pd.DataFrame()
    Final_classification=[]
    for rows in range(len(For_Accuracy)):
        prediction=For_Accuracy.iloc[rows,1:].mode().iloc[0]
        Final_classification.append(prediction)
    ClassificationsDF = pd.Series(Final_classification)

    For_Accuracy['Final_classification'] = ClassificationsDF
    For_Accuracy['classification_correct'] = For_Accuracy.Final_classification == For_Accuracy.label
    accuracy = For_Accuracy.classification_correct.mean()

    for rows in range(len(For_Accuracy)):
        classification = For_Accuracy['Final_classification'][rows]
        class_correct = For_Accuracy['classification_correct'][rows]
        if classification == 'healthy':
            if class_correct:
                TP += 1
        if classification == 'diseased':
            if not class_correct:
                FP += 1
        if classification == 'healthy':
            if not class_correct:
                FN += 1
        if classification == 'diseased':
            if class_correct:
                TN += 1

    sensitivity = TP/(TP+FN)
    specifity = TN/(TN+FP)

    return accuracy, sensitivity, specifity


accuracy, sensitivity, specifity = final_classification(test_dataDF, tree_dict)
print('Accuracy :'+ str(accuracy))
print('Sensitivity :'+ str(sensitivity))
print('specifity :'+ str(specifity))

# tree_dict contains all trees passed the accuracy_cutoff
# tree_graph contains graph pointers


my_features = get_features(tree_dict)
feature_counts = count_features(my_features)

print('#####my features####')
print(my_features)
print('###feature counts###')
print(feature_counts)

scatter_df = pd.DataFrame()
i=0
for key in feature_counts.keys():
    i += 1
    scatter_df[key] = myData[key]
    if i == 19:
        break

pd.plotting.scatter_matrix(scatter_df, alpha=0.5, figsize=(30, 30), diagonal='kde')
plt.show()