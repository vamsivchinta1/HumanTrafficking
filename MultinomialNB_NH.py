
"""
Multinomial Naive Bayes

"""
#%% Importing packages

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score

#%% Importing and Defining Train/Test Sets

df = pd.read_csv("newData.csv")
#df.head(5)

y = df['RecruiterRelationship']
X = df[['CountryOfExploitation','gender','typeOfExploitConcatenated']]

#%% Encode Variables

#- Predictors
X = pd.get_dummies(X, dummy_na = True)

#- Target (we can use le_y() to decode outputs after prediction)
y = y.replace(np.NaN, "-99").astype('category')

le_y = preprocessing.LabelEncoder()
y = le_y.fit_transform(y)

#%% Initialize model
clf = MNB()

#%% Cross Validation using Stratisfied 10-Fold

kf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=None)

scores = []
for train_idx, test_idx in kf.split(X,y):
    #print("TRAIN:", train_idx, "TEST:", test_idx)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = clf.fit(X_train, y_train)
    predictions = model.predict(X_test)
    scores.append(accuracy_score(y_test, predictions))
    
print('Scores from each iteration: {}'.format(scores))
print('Average 10-Fold Accuracy: {}'.format(np.mean(scores)))
    
