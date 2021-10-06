import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_excel('ThyroidCancerNormalizedDataAll.xlsx', index_col=None, header=None)

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
myData['label'].replace({'0.0': 'not-exposed'}, inplace=True)
myData['label'].replace({'1.0': 'exposed'}, inplace=True)
myData['label'].replace({'2.0': 'healthy'}, inplace=True)

X = myData.iloc[:, 0:-1].values
y = myData.iloc[:, -1].values

print('############## X #############')
print(X)
print('############### y ############')
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print('############## X train #############')
print(X_train)
print('############### X test ############')
print(X_test)

print('############## y train #############')
print(y_train)
print('############### y test ############')
print(y_test)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print('############## X train after feature selection #############')
print(X_train)
print('############### X test after feature selection ############')
print(X_test)

lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

classifier = RandomForestClassifier(max_depth=2, random_state=0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print('############## y pred ###############')
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))
