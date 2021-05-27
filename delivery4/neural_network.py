import tensorflow as tf
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix


def train_test_split(data, testsize=0.3):
    train_data = data.sample(frac=(1-testsize))
    test_data = data.drop(train_data.index)
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()
    train_data.drop(columns=['index'], inplace=True)
    test_data.drop(columns=['index'], inplace=True)
    return train_data, test_data

def get_confusion_matrix(labels, predictions):
    prediction_list = predictions.tolist()
    pred_col = []

    label_list = labels.tolist()
    label_col =[]
    for i in range(len(prediction_list)):
        pred_col.append(prediction_list[i][0])
        label_col.append(label_list[i][0])

    return confusion_matrix(label_col, pred_col)

def get_results(conf):
    TP = conf[0][0]
    FP = conf[0][1]
    FN = conf[1][0]
    TN = conf[1][1]
    sensitivity = TP / (TP + FN)
    specification = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1_score = (2 * precision * sensitivity) / (precision + sensitivity)
    accuracy = (TP + TN)/(TP+FN+TN+FP)

    return accuracy, sensitivity, specification, precision, f1_score

myData = pd.read_excel('dataset/DifferentiallyExpressedGenes.xlsx', index_col=None, header=None)
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
myData['label'].replace({'1.0': 1}, inplace=True)
myData['label'].replace({'0.0': 0}, inplace=True)
rawdata = myData

print(rawdata)

train_df, test_df = train_test_split(rawdata)
y_train = np.array(train_df['label'])
x_train = train_df.drop(train_df.columns[0], axis=1).values

y_test = np.array(test_df['label'])
x_test = test_df.drop(test_df.columns[0], axis=1)

y_train = np.array(to_categorical(y_train, num_classes=2))
y_test = np.array(to_categorical(y_test, num_classes=2))

print('############## X train #############')
print(x_train)
print('############### X test ############')
print(x_test)

print('############## y train #############')
print(y_train)
print('############### y test ############')
print(y_test)

opt = tf.keras.optimizers.SGD(learning_rate=0.0001)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, input_dim=8, activation='relu'))
model.add(tf.keras.layers.Dropout(0.15))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.15))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.15))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.15))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=250, batch_size=128)
predictions = model.predict(x_test).round()

conf = get_confusion_matrix(y_test, predictions)
accuracy, sensitivity, specification, precision, f1_score = get_results(conf)

print('Accuracy: '+ str(accuracy))
print('Sensitivity: ' + str(sensitivity))
print('Specification: ' + str(specification))
print('Precision: ' + str(precision))
print('F1 Score: ' + str(f1_score))