

```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib



```python
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
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
from sklearn.model_selection import train_test_split
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
```


```python
clf = OneVsRestClassifier(AdaBoostClassifier())
clf.fit(X_train, y_train)
```


```python
results = cross_val_score(clf, X, y, cv=3)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
```
