

```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
```


```python
# Read the dataset
crimes_stacked = pd.read_csv('crimes_stacked.csv', index_col=None)
```


```python
# Get rid of useless columns
X = crimes_stacked[['Arrest', 'Domestic', 'Beat', 'Community Area',
                   'Latitude', 'Longitude', 'Year', 'Hour', 'Closest police station']]
# Extract the target feature
y = crimes_stacked['Primary Type']
```


```python
# Fill Nan with 0
X = X.fillna(0)
# Gather column names
feature_names = list(X)
target_names = list(y)
```


```python
# Separate data into training and testing datasets
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
```

## Random forest


```python
# Use grid search with 3 K-Fold to find the best parameters
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=3, random_state=seed)
# Parameters to be tested
parameters = {'n_estimators':[25], 
              'criterion':['gini', 'entropy'], 
              'max_depth':[None, 2, 3, 4]}
# Comparison
rf = RandomForestClassifier()
clf = GridSearchCV(rf, parameters, cv=kfold, scoring=scoring)
clf.fit(X_train, y_train)
```


```python
# Train with the best parameters
print("Best estimator: ", clf.best_estimator_)
rf = clf.best_estimator_
rf.fit(X_train, y_train)
```


```python
results = cross_val_score(rf, X, y)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
```


```python
importances = rf.feature_importances_
std = np.std([importances],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

importance_to_plot = []
for f in range(X.shape[1]):
    if(importances[indices[f]] > 0.01):
        importance_to_plot.append(importances[indices[f]])
        print("%d. feature %s (%f)" % (f + 1, feature_names[f+1], importances[indices[f]]))
```


```python
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, len(importance_to_plot)])
plt.show()
```

## Extra Trees (randomized decision trees)


```python
from sklearn.ensemble import ExtraTreesClassifier
# Use grid search with 3 K-Fold to find the best parameters
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=3, random_state=seed)
# Parameters to be tested
parameters = {'n_estimators':[25], 
              'criterion':['gini', 'entropy'], 
              'max_depth':[None, 2, 3, 4]}
# Comparison
xrf = ExtraTreesClassifier()
clf = GridSearchCV(xrf, parameters, cv=kfold, scoring=scoring)
clf.fit(X_train, y_train)
print("Best estimator: ", clf.best_estimator_)
```


```python
# Train with the best parameters
xrf = clf.best_estimator_
xrf.fit(X_train, y_train)
```


```python
results = cross_val_score(xrf, X, y)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
```


```python
importances = xrf.feature_importances_
std = np.std([importances],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

importance_to_plot = []
for f in range(X.shape[1]):
    if(importances[indices[f]] > 0.01):
        importance_to_plot.append(importances[indices[f]])
        print("%d. feature %s (%f)" % (f + 1, feature_names[f+1], importances[indices[f]]))
```


```python
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, len(importance_to_plot)])
plt.show()
```
