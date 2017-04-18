

```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.model_selection import cross_val_score
import pydotplus
from IPython.display import Image  
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
# Convert into dummy features for scikit-learn
# X = pd.get_dummies(X, prefix='is_') 
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


```python
# Parameters to be tested
parameters = {'criterion':['gini', 'entropy'], 'max_depth':[None, 2, 3]}
# Use grid search with 3 K-Fold to find the best parameters
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=3, random_state=seed)
# Comparison
dt = tree.DecisionTreeClassifier()
clf = GridSearchCV(dt, parameters, cv=kfold, scoring=scoring)
clf.fit(X_train, y_train)
print("Best estimator: ", clf.best_estimator_)
```

    ('Best estimator: ', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best'))



```python
# Train with the best parameters
dt = clf.best_estimator_
dt.fit(X_train, y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best')




```python
results = cross_val_score(dt, X_train, y_train)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
```

    /Users/thomas-legrand/anaconda/lib/python2.7/site-packages/sklearn/model_selection/_split.py:581: Warning: The least populated class in y has only 1 members, which is too few. The minimum number of groups for any class cannot be less than n_splits=3.
      % (min_groups, self.n_splits)), Warning)


    Accuracy: 41.683% (0.056%)



```python
importances = dt.feature_importances_
std = np.std([importances],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

importance_to_plot = []
for f in range(X.shape[1]):
    if(importances[indices[f]]):
        importance_to_plot.append(importances[indices[f]])
        print("%d. feature %s (%f)" % (f + 1, feature_names[f], importances[indices[f]]))
```

    Feature ranking:
    1. feature Arrest (0.284787)
    2. feature Domestic (0.268530)
    3. feature Beat (0.129260)
    4. feature Community Area (0.105093)
    5. feature Latitude (0.099534)
    6. feature Longitude (0.048530)
    7. feature Year (0.041806)
    8. feature Hour (0.014632)
    9. feature Closest police station (0.007828)



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


![png](output_10_0.png)



```python
# Plot and store the decision tree
dot_data = tree.export_graphviz(dt, out_file=None, 
                         feature_names=feature_names,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("DecisionTree.pdf")
Image(graph.create_png()) 
```
