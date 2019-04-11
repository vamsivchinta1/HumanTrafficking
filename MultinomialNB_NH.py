
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
from operator import add
import functools as fun

#%% Importing and Defining Train/Test Sets

df = pd.read_csv("newData.csv")

# Remove null values in target
df = df[np.logical_not(df.RecruiterRelationship.isna())]

#df.head(5)
#df.columns

y = df['RecruiterRelationship']
X = df[['majorityStatus','CountryOfExploitation','typeOfExploitConcatenated']]


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

#- Target (we can use le_y() to decode outputs after prediction)
y = y.replace(np.NaN, "-99").astype('category')

filt = np.logical_not(y == '-99')
#filt = np.logical_not(np.logical_or(y == 'Unknown', y == '-99'))
#filt = np.logical_not(np.logical_or(np.logical_or(y == 'Unknown', y == '-99'), y == 'Various'))

y = y[filt]

#- Predictors
X = pd.get_dummies(X, dummy_na = True)
X = X[filt]

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
print("Model training complete!")
print('Average 10-Fold Accuracy: {}'.format(np.mean(scores)))

#%%    
class_probs = []
for i in range(0,len(clf.predict_proba(X_test))):
    class_probs.append(list(clf.predict_proba(X_test)[i]))
print("Concatenation Complete!")

#%%
final_probs = fun.reduce(lambda x,y: list(map(add,x,y)), class_probs)   
print("Addition complete!")

#%%
avg_class_probs = list(map(lambda x: (x/len(clf.predict_proba(X_test)))*100, final_probs))
print(avg_class_probs)

target_cat = le_y.inverse_transform(range(0,17))

kfold_predictions = pd.DataFrame(list(map(lambda x,y: (x,y), target_cat, avg_class_probs)))
kfold_predictions.to_csv("Predicted_Classes.csv", index = False)
