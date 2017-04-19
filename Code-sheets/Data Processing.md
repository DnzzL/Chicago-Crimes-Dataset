
# DCU Data Analytics Project


```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib


## Check the versions and import libraries


```python
# Load libraries
import pandas as pd
import numpy as np
import matplotlib
import sklearn 
import datetime
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Python version
import sys
print('Python: {}'.format(sys.version))
# numpy
print('numpy: {}'.format(np.__version__))
# matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
print('pandas: {}'.format(pd.__version__))
# scikit-learn
print('sklearn: {}'.format(sklearn.__version__))
```

    Python: 2.7.12 |Anaconda custom (x86_64)| (default, Jul  2 2016, 17:43:17) 
    [GCC 4.2.1 (Based on Apple Inc. build 5658) (LLVM build 2336.11.00)]
    numpy: 1.11.3
    matplotlib: 1.5.3
    pandas: 0.19.2
    sklearn: 0.18.1


## Load the data

Urls:
- https://www.kaggle.com/currie32/crimes-in-chicago/downloads/crimes-in-chicago.zip
- https://data.cityofchicago.org/Public-Safety/Police-Stations/z8bn-74gv


```python
crimes_2001 = pd.read_csv('Chicago_Crimes_2001_to_2004.csv', sep=',', error_bad_lines=False)
crimes_2005 = pd.read_csv('Chicago_Crimes_2005_to_2007.csv', sep=',', error_bad_lines=False)
crimes_2008 = pd.read_csv('Chicago_Crimes_2008_to_2011.csv', sep=',', error_bad_lines=False)
crimes_2012 = pd.read_csv('Chicago_Crimes_2012_to_2017.csv', sep=',', error_bad_lines=False)
frames = [crimes_2001, crimes_2005, crimes_2008, crimes_2012]
```

    Skipping line 1513591: expected 23 fields, saw 24
    
    /Users/thomas-legrand/anaconda/lib/python2.7/site-packages/IPython/core/interactiveshell.py:2717: DtypeWarning: Columns (17,20) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    Skipping line 533719: expected 23 fields, saw 24
    
    Skipping line 1149094: expected 23 fields, saw 41
    



```python
police_df = pd.read_csv('Police_Stations.csv', sep=',')
```

## Clean the data


```python
# Remove typing errors
for csv in frames:
    csv['Latitude'] = pd.to_numeric(csv['Latitude'], errors='coerce')
    csv['Y Coordinate'] = pd.to_numeric(csv['Y Coordinate'], errors='coerce')
    csv = csv[(csv['Year'] >= 2001) & ((csv['Latitude'] >= 40) | (csv['Latitude'] == 0))
                      & ((csv['Year'] >= -90) | (csv['Latitude'] == 0))]
```

## Transform the data


```python
# Drop useless column
for csv in frames:
    csv = csv.drop(['Unnamed: 0'], axis=1)
```


```python
# Calculate the distance between two geopoints
def distanceBetweenPlaces(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    km = 6367 * c
    return km

# Determine the closest police station
def closestPoliceStation(lat, lon):
    minD = 100
    for row in police_df.itertuples():
        dist = distanceBetweenPlaces(row[13], row[14], lat, lon)
        if(dist < minD):
            minD = dist
    return minD
```


```python
# Extract hour from date colum
def extractHours(row):
    date, time, half = row.split()
    h = int(time[0:2])
    if(half == 'AM'):
        return h
    if((half == 'PM') & (h == 0)):
        return 0
    if(half == 'PM'):
        return h+12
```


```python
# Compute and add hour column
for csv in frames:
    a = datetime.datetime.now().replace(microsecond=0)
    csv['Hour'] = csv.apply(lambda row: extractHours(row['Date']), axis=1)
    b = datetime.datetime.now().replace(microsecond=0)
    t = b-a
    print("Done in ", t)
```

    ('Done in ', datetime.timedelta(0, 51))
    ('Done in ', datetime.timedelta(0, 48))
    ('Done in ', datetime.timedelta(0, 65))
    ('Done in ', datetime.timedelta(0, 36))



```python
# Add distance to closest police station column
i = 1
for csv in frames:
    a = datetime.datetime.now().replace(microsecond=0)
    csv['Closest police station'] = csv.apply(
        lambda row: closestPoliceStation(row['Latitude'], row['Longitude']), axis=1)
    b = datetime.datetime.now().replace(microsecond=0)
    t = b-a
    print(i, " Done in ", t)
    i += 1
    name = (str)(i)
    name += '.csv'
    csv.to_csv(path_or_buf=name)
```


```python
# Stack the data
crimes_df = pd.concat(frames)
```


```python
# Stock the stacked file
crimes_df.to_csv(path_or_buf='crimes_stacked.csv')
```

## Summarize the data


```python
# We take a random sample to take a look at the data
sample = crimes_df.sample(frac=0.01)
```


```python
# Shape
print(sample.shape)
```

    (78355, 24)



```python
# Types
sample.dtypes
```




    ID                          int64
    Case Number                object
    Date                       object
    Block                      object
    IUCR                       object
    Primary Type               object
    Description                object
    Location Description       object
    Arrest                       bool
    Domestic                     bool
    Beat                        int64
    District                  float64
    Ward                      float64
    Community Area            float64
    FBI Code                   object
    X Coordinate              float64
    Y Coordinate              float64
    Year                      float64
    Updated On                 object
    Latitude                  float64
    Longitude                 float64
    Location                   object
    Closest police station    float64
    Hour                        int64
    dtype: object




```python
# Head
sample.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Case Number</th>
      <th>Date</th>
      <th>Block</th>
      <th>IUCR</th>
      <th>Primary Type</th>
      <th>Description</th>
      <th>Location Description</th>
      <th>Arrest</th>
      <th>Domestic</th>
      <th>...</th>
      <th>FBI Code</th>
      <th>X Coordinate</th>
      <th>Y Coordinate</th>
      <th>Year</th>
      <th>Updated On</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Location</th>
      <th>Closest police station</th>
      <th>Hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24057</th>
      <td>4709560</td>
      <td>HM315101</td>
      <td>04/26/2006 10:30:00 PM</td>
      <td>009XX W 123RD ST</td>
      <td>1320</td>
      <td>CRIMINAL DAMAGE</td>
      <td>TO VEHICLE</td>
      <td>DRIVEWAY - RESIDENTIAL</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>14</td>
      <td>1172131.0</td>
      <td>1823265.0</td>
      <td>2006.0</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.670481</td>
      <td>-87.645657</td>
      <td>(41.67048078, -87.645656869)</td>
      <td>3.003780</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1058914</th>
      <td>10129758</td>
      <td>HY318243</td>
      <td>06/27/2015 10:00:00 AM</td>
      <td>077XX W CATALPA AVE</td>
      <td>0820</td>
      <td>THEFT</td>
      <td>$500 AND UNDER</td>
      <td>STREET</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>06</td>
      <td>1123707.0</td>
      <td>1935588.0</td>
      <td>2015.0</td>
      <td>08/17/2015 03:03:40 PM</td>
      <td>41.979639</td>
      <td>-87.820435</td>
      <td>(41.979639351, -87.820434768)</td>
      <td>4.526805</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1640140</th>
      <td>3365503</td>
      <td>HK412313</td>
      <td>06/06/2004 10:24:42 AM</td>
      <td>073XX N HARLEM AVE</td>
      <td>0281</td>
      <td>CRIM SEXUAL ASSAULT</td>
      <td>NON-AGGRAVATED</td>
      <td>RESIDENCE</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>02</td>
      <td>1127389.0</td>
      <td>1948212.0</td>
      <td>2004.0</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>42.014219</td>
      <td>-87.806608</td>
      <td>(42.014219493, -87.806607977)</td>
      <td>5.572023</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1477754</th>
      <td>4008583</td>
      <td>HL364773</td>
      <td>05/14/2005 01:00:00 AM</td>
      <td>062XX S KEDZIE AVE</td>
      <td>0820</td>
      <td>THEFT</td>
      <td>$500 AND UNDER</td>
      <td>STREET</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>06</td>
      <td>1156121.0</td>
      <td>1863260.0</td>
      <td>2005.0</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.780570</td>
      <td>-87.703181</td>
      <td>(41.780569671, -87.703181081)</td>
      <td>0.502676</td>
      <td>1</td>
    </tr>
    <tr>
      <th>46918</th>
      <td>6056255</td>
      <td>HP157039</td>
      <td>02/03/2008 08:30:00 AM</td>
      <td>061XX S SANGAMON ST</td>
      <td>0610</td>
      <td>BURGLARY</td>
      <td>FORCIBLE ENTRY</td>
      <td>APARTMENT</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>05</td>
      <td>1171028.0</td>
      <td>1864134.0</td>
      <td>2008.0</td>
      <td>02/04/2016 06:33:39 AM</td>
      <td>41.782655</td>
      <td>-87.648504</td>
      <td>(41.782655257, -87.648503767)</td>
      <td>1.079757</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
# Descriptions
sample.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Beat</th>
      <th>District</th>
      <th>Ward</th>
      <th>Community Area</th>
      <th>X Coordinate</th>
      <th>Y Coordinate</th>
      <th>Year</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Closest police station</th>
      <th>Hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7.835500e+04</td>
      <td>78355.000000</td>
      <td>78355.000000</td>
      <td>71522.000000</td>
      <td>71504.000000</td>
      <td>7.835500e+04</td>
      <td>7.835500e+04</td>
      <td>78355.000000</td>
      <td>78355.000000</td>
      <td>78355.000000</td>
      <td>78355.000000</td>
      <td>78355.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.925130e+06</td>
      <td>1198.505533</td>
      <td>11.312858</td>
      <td>22.629666</td>
      <td>37.760629</td>
      <td>1.164451e+06</td>
      <td>1.885618e+06</td>
      <td>2007.679242</td>
      <td>41.841735</td>
      <td>-87.672054</td>
      <td>1.988461</td>
      <td>14.573480</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.561880e+06</td>
      <td>704.102886</td>
      <td>6.948276</td>
      <td>13.794216</td>
      <td>21.594604</td>
      <td>1.630111e+04</td>
      <td>3.151974e+04</td>
      <td>4.051770</td>
      <td>0.086687</td>
      <td>0.059326</td>
      <td>1.205595</td>
      <td>6.390505</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.840000e+02</td>
      <td>111.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.095268e+06</td>
      <td>1.814059e+06</td>
      <td>2001.000000</td>
      <td>41.644668</td>
      <td>-87.924970</td>
      <td>0.000197</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.862816e+06</td>
      <td>623.000000</td>
      <td>6.000000</td>
      <td>10.000000</td>
      <td>23.000000</td>
      <td>1.152777e+06</td>
      <td>1.859052e+06</td>
      <td>2005.000000</td>
      <td>41.768614</td>
      <td>-87.714488</td>
      <td>1.215604</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.168100e+06</td>
      <td>1111.000000</td>
      <td>10.000000</td>
      <td>22.000000</td>
      <td>32.000000</td>
      <td>1.165838e+06</td>
      <td>1.890161e+06</td>
      <td>2008.000000</td>
      <td>41.854313</td>
      <td>-87.666666</td>
      <td>1.847460</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.724016e+06</td>
      <td>1732.000000</td>
      <td>17.000000</td>
      <td>34.000000</td>
      <td>58.000000</td>
      <td>1.176352e+06</td>
      <td>1.909456e+06</td>
      <td>2010.000000</td>
      <td>41.907237</td>
      <td>-87.628426</td>
      <td>2.495033</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.081508e+07</td>
      <td>2535.000000</td>
      <td>31.000000</td>
      <td>50.000000</td>
      <td>77.000000</td>
      <td>1.205116e+06</td>
      <td>1.951532e+06</td>
      <td>2016.000000</td>
      <td>42.022632</td>
      <td>-87.524529</td>
      <td>13.220981</td>
      <td>24.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Class distribution
print(crimes_df.groupby('Primary Type').size())
```

    Primary Type
    ARSON                                  12956
    ASSAULT                               477284
    BATTERY                              1430338
    BURGLARY                              467136
    CONCEALED CARRY LICENSE VIOLATION         84
    CRIM SEXUAL ASSAULT                    28027
    CRIMINAL DAMAGE                       915314
    CRIMINAL TRESPASS                     227626
    DECEPTIVE PRACTICE                    268520
    DOMESTIC VIOLENCE                          2
    GAMBLING                               18653
    HOMICIDE                                8983
    HUMAN TRAFFICKING                         20
    INTERFERENCE WITH PUBLIC OFFICER       15560
    INTIMIDATION                            4581
    KIDNAPPING                              7664
    LIQUOR LAW VIOLATION                   17317
    MOTOR VEHICLE THEFT                   365629
    NARCOTICS                             871454
    NON - CRIMINAL                            38
    NON-CRIMINAL                              80
    NON-CRIMINAL (SUBJECT SPECIFIED)           4
    OBSCENITY                                471
    OFFENSE INVOLVING CHILDREN             48419
    OTHER NARCOTIC VIOLATION                 143
    OTHER OFFENSE                         485863
    PROSTITUTION                           85542
    PUBLIC INDECENCY                         162
    PUBLIC PEACE VIOLATION                 58154
    RITUALISM                                 29
    ROBBERY                               297373
    SEX OFFENSE                            27138
    STALKING                                3636
    THEFT                                1614657
    WEAPONS VIOLATION                      76597
    dtype: int64


## Data Visualization


```python
# Box and whisker plots
sample[['Year', 'Hour', 'Latitude', 'Longitude', 'Closest police station']].plot(
    kind='box', subplots=True, layout=(5,5), figsize=(15,15), sharex=False, sharey=False)
plt.show()
```


![png](output_26_0.png)



```python
# Histograms
sample[['Year', 'Hour', 'Latitude', 'Longitude']].hist(figsize=(5,5))
plt.show()
```


![png](output_27_0.png)



```python
# Scatter plot matrix
scatter_matrix(sample[['Beat', 'District', 'Ward', 'Community Area', 'Latitude', 'Longitude',
                      'Year', 'Hour', 'Closest police station']], figsize=(30, 30))
plt.show()
```


![png](output_28_0.png)

