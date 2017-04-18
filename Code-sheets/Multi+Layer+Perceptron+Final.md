

```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib



```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler   
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
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
X = MinMaxScaler().fit_transform(X)
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
# Rescale the data
scaler = StandardScaler()
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  
```


```python
# Use grid search with 3 K-Fold to find the best parameters
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=3, random_state=seed)
# Parameters to be tested
parameters = {'activation':['logistic', 'tanh', 'relu'],
              'hidden_layer_sizes':[(50, 15), (25, 10), (10, 5)],
              'solver':['sgd', 'adam'], 
              'learning_rate_init':[0.001, 0.01]}
# Comparison
mlp = MLPClassifier()
clf = GridSearchCV(mlp, parameters, cv=kfold)
clf.fit(X_train, y_train)
print("Best estimator: ", clf.best_estimator_)
```


```python
# Train with the best parameters
mlp = clf.best_estimator_
mlp.fit(X_train, y_train)
```


```python
results = cross_val_score(dt, X, y)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
```

    ('Accuracy Score: ', 0.43136363636363634)
    ('Precision Score: ', 0.17220092657029643)



```python
print("Current loss:", mlp.loss_)
```
