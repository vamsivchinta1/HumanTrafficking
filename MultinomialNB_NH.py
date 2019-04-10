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

# Remove null values in target
df = df[np.logical_not(df.RecruiterRelationship.isna())]

#df.head(5)
df.columns
y = df['RecruiterRelationship']
X = df[['majorityStatusAtExploit','CountryOfExploitation','typeOfExploitConcatenated']]

#%% Data Quality

def Qual_Stats(df):
    columns = df.columns   
    report = []
    
    for column in columns:     
        name = column
        count = df.shape[0]
        missing_percent = (df[column].isnull().values.sum())/count
        cardinality = df[column].nunique()
        mode = df[column].value_counts().index[0]
        mode_freq = df[column].value_counts().values[0]
        mode_percent = mode_freq/count
        mode_2 = df[column].value_counts().index[1]
        mode_2_freq = df[column].value_counts().values[1]
        mode_2_percent = mode_2_freq/count
        
        
        row = {
                'Feature': name,
                'Count': count,
                'Missing %': missing_percent, 
                'Card.': cardinality, 
                'Mode': mode,
                'Mode Freq.': mode_freq,
                'Mode %': mode_percent,
                '2nd Mode': mode_2,
                '2nd Mode Freq.': mode_2_freq,
                '2nd Mode %': mode_2_percent
                }
        
        report.append(row)
    
    return pd.DataFrame(report, columns = row.keys()).sort_values(by=['Missing %'], axis=0, ascending=False).reset_index(drop = True)

report = Qual_Stats(df)
report[['Feature','Missing %']]

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

kf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)

scores = []
for train_idx, test_idx in kf.split(X,y):
    #print("TRAIN:", train_idx, "TEST:", test_idx)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = clf.fit(X_train, y_train)
    predictions = model.predict(X_test)
    scores.append(accuracy_score(y_test, predictions))
    
#print('Scores from each iteration: {}'.format(scores))
print('Average 10-Fold Accuracy: {}'.format(np.mean(scores)))
    
