---
layout: page
icon: code
title: Jupyter Notebook Test
---

```python
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


```


```python
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
```


```python
fetch_housing_data()
```


```python
housing = load_housing_data()
housing.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
    longitude             20640 non-null float64
    latitude              20640 non-null float64
    housing_median_age    20640 non-null float64
    total_rooms           20640 non-null float64
    total_bedrooms        20433 non-null float64
    population            20640 non-null float64
    households            20640 non-null float64
    median_income         20640 non-null float64
    median_house_value    20640 non-null float64
    ocean_proximity       20640 non-null object
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB



```python
housing["ocean_proximity"].value_counts()
```




    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64




```python
housing.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()
```


![png](immo_files/immo_7_0.png)



```python
import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
```


```python
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), " Training +", len(test_set), "Test")
```

    16512  Training + 4128 Test



```python
import hashlib

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]
```


```python
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
```


```python
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), " Training +", len(test_set), "Test")
```

    16512  Training + 4128 Test



```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```


```python
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), " Training +", len(test_set), "Test")
```

    16512  Training + 4128 Test



```python
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
housing["income_cat"].hist()
plt.show()
```


![png](immo_files/immo_15_0.png)



```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```


```python
housing["income_cat"].value_counts() / len(housing)
```




    3.0    0.350581
    2.0    0.318847
    4.0    0.176308
    5.0    0.114438
    1.0    0.039826
    Name: income_cat, dtype: float64




```python
strat_train_set["income_cat"].value_counts() / len(strat_train_set)
```




    3.0    0.350594
    2.0    0.318859
    4.0    0.176296
    5.0    0.114402
    1.0    0.039850
    Name: income_cat, dtype: float64




```python
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
```




    3.0    0.350533
    2.0    0.318798
    4.0    0.176357
    5.0    0.114583
    1.0    0.039729
    Name: income_cat, dtype: float64




```python
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```


```python
housing = strat_train_set.copy()
```


```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f5c5b5ff7f0>




![png](immo_files/immo_22_1.png)



```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=housing["population"]/100, label="population", figsize=(10,7),
            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f5c4b3ec908>




![png](immo_files/immo_23_1.png)



```python
corr_matrix = housing.corr()
```


```python
corr_matrix["median_house_value"].sort_values(ascending=False)
```




    median_house_value    1.000000
    median_income         0.687160
    total_rooms           0.135097
    housing_median_age    0.114110
    households            0.064506
    total_bedrooms        0.047689
    population           -0.026920
    longitude            -0.047432
    latitude             -0.142724
    Name: median_house_value, dtype: float64




```python
from pandas.tools.plotting import scatter_matrix

attributes =["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f5c4b414e48>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f5c5856fc88>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f5c584e5160>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f5c586b0710>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f5c58749390>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f5c58749b70>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f5c587ff8d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f5c5452d8d0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f5c54434f28>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f5c588d5ac8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f5c58932358>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f5c589cd6d8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f5c58a5f048>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f5c58a7a400>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f5c58c159b0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f5c58b9e7b8>]], dtype=object)




![png](immo_files/immo_26_1.png)



```python
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f5c58dab518>




![png](immo_files/immo_27_1.png)



```python
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]
```


```python
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
```




    median_house_value          1.000000
    median_income               0.687160
    rooms_per_household         0.146285
    total_rooms                 0.135097
    housing_median_age          0.114110
    households                  0.064506
    total_bedrooms              0.047689
    population_per_household   -0.021985
    population                 -0.026920
    longitude                  -0.047432
    latitude                   -0.142724
    bedrooms_per_room          -0.259984
    Name: median_house_value, dtype: float64




```python
corr_matrix = housing.corr()
corr_matrix["population"].sort_values(ascending=False)
```




    population                  1.000000
    households                  0.904637
    total_bedrooms              0.876320
    total_rooms                 0.855109
    longitude                   0.108030
    population_per_household    0.076225
    bedrooms_per_room           0.037778
    median_income               0.002380
    median_house_value         -0.026920
    rooms_per_household        -0.074692
    latitude                   -0.115222
    housing_median_age         -0.298710
    Name: population, dtype: float64




```python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```


```python
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
```




    Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0)




```python
housing_num
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>-121.89</td>
      <td>37.29</td>
      <td>38.0</td>
      <td>1568.0</td>
      <td>351.0</td>
      <td>710.0</td>
      <td>339.0</td>
      <td>2.7042</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>-121.93</td>
      <td>37.05</td>
      <td>14.0</td>
      <td>679.0</td>
      <td>108.0</td>
      <td>306.0</td>
      <td>113.0</td>
      <td>6.4214</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.20</td>
      <td>32.77</td>
      <td>31.0</td>
      <td>1952.0</td>
      <td>471.0</td>
      <td>936.0</td>
      <td>462.0</td>
      <td>2.8621</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>-119.61</td>
      <td>36.31</td>
      <td>25.0</td>
      <td>1847.0</td>
      <td>371.0</td>
      <td>1460.0</td>
      <td>353.0</td>
      <td>1.8839</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>-118.59</td>
      <td>34.23</td>
      <td>17.0</td>
      <td>6592.0</td>
      <td>1525.0</td>
      <td>4459.0</td>
      <td>1463.0</td>
      <td>3.0347</td>
    </tr>
    <tr>
      <th>19480</th>
      <td>-120.97</td>
      <td>37.66</td>
      <td>24.0</td>
      <td>2930.0</td>
      <td>588.0</td>
      <td>1448.0</td>
      <td>570.0</td>
      <td>3.5395</td>
    </tr>
    <tr>
      <th>8879</th>
      <td>-118.50</td>
      <td>34.04</td>
      <td>52.0</td>
      <td>2233.0</td>
      <td>317.0</td>
      <td>769.0</td>
      <td>277.0</td>
      <td>8.3839</td>
    </tr>
    <tr>
      <th>13685</th>
      <td>-117.24</td>
      <td>34.15</td>
      <td>26.0</td>
      <td>2041.0</td>
      <td>293.0</td>
      <td>936.0</td>
      <td>375.0</td>
      <td>6.0000</td>
    </tr>
    <tr>
      <th>4937</th>
      <td>-118.26</td>
      <td>33.99</td>
      <td>47.0</td>
      <td>1865.0</td>
      <td>465.0</td>
      <td>1916.0</td>
      <td>438.0</td>
      <td>1.8242</td>
    </tr>
    <tr>
      <th>4861</th>
      <td>-118.28</td>
      <td>34.02</td>
      <td>29.0</td>
      <td>515.0</td>
      <td>229.0</td>
      <td>2690.0</td>
      <td>217.0</td>
      <td>0.4999</td>
    </tr>
    <tr>
      <th>16365</th>
      <td>-121.31</td>
      <td>38.02</td>
      <td>24.0</td>
      <td>4157.0</td>
      <td>951.0</td>
      <td>2734.0</td>
      <td>879.0</td>
      <td>2.7981</td>
    </tr>
    <tr>
      <th>19684</th>
      <td>-121.62</td>
      <td>39.14</td>
      <td>41.0</td>
      <td>2183.0</td>
      <td>559.0</td>
      <td>1202.0</td>
      <td>506.0</td>
      <td>1.6902</td>
    </tr>
    <tr>
      <th>19234</th>
      <td>-122.69</td>
      <td>38.51</td>
      <td>18.0</td>
      <td>3364.0</td>
      <td>501.0</td>
      <td>1442.0</td>
      <td>506.0</td>
      <td>6.6854</td>
    </tr>
    <tr>
      <th>13956</th>
      <td>-117.06</td>
      <td>34.17</td>
      <td>21.0</td>
      <td>2520.0</td>
      <td>582.0</td>
      <td>416.0</td>
      <td>151.0</td>
      <td>2.7120</td>
    </tr>
    <tr>
      <th>2390</th>
      <td>-119.46</td>
      <td>36.91</td>
      <td>12.0</td>
      <td>2980.0</td>
      <td>495.0</td>
      <td>1184.0</td>
      <td>429.0</td>
      <td>3.9141</td>
    </tr>
    <tr>
      <th>11176</th>
      <td>-117.96</td>
      <td>33.83</td>
      <td>30.0</td>
      <td>2838.0</td>
      <td>649.0</td>
      <td>1758.0</td>
      <td>593.0</td>
      <td>3.3831</td>
    </tr>
    <tr>
      <th>15614</th>
      <td>-122.41</td>
      <td>37.81</td>
      <td>25.0</td>
      <td>1178.0</td>
      <td>545.0</td>
      <td>592.0</td>
      <td>441.0</td>
      <td>3.6728</td>
    </tr>
    <tr>
      <th>2953</th>
      <td>-119.02</td>
      <td>35.35</td>
      <td>42.0</td>
      <td>1239.0</td>
      <td>251.0</td>
      <td>776.0</td>
      <td>272.0</td>
      <td>1.9830</td>
    </tr>
    <tr>
      <th>13209</th>
      <td>-117.72</td>
      <td>34.05</td>
      <td>8.0</td>
      <td>1841.0</td>
      <td>409.0</td>
      <td>1243.0</td>
      <td>394.0</td>
      <td>4.0614</td>
    </tr>
    <tr>
      <th>6569</th>
      <td>-118.15</td>
      <td>34.20</td>
      <td>46.0</td>
      <td>1505.0</td>
      <td>261.0</td>
      <td>857.0</td>
      <td>269.0</td>
      <td>4.5000</td>
    </tr>
    <tr>
      <th>5825</th>
      <td>-118.30</td>
      <td>34.19</td>
      <td>14.0</td>
      <td>3615.0</td>
      <td>913.0</td>
      <td>1924.0</td>
      <td>852.0</td>
      <td>3.5083</td>
    </tr>
    <tr>
      <th>18086</th>
      <td>-122.05</td>
      <td>37.31</td>
      <td>25.0</td>
      <td>4111.0</td>
      <td>538.0</td>
      <td>1585.0</td>
      <td>568.0</td>
      <td>9.2298</td>
    </tr>
    <tr>
      <th>16718</th>
      <td>-120.66</td>
      <td>35.49</td>
      <td>17.0</td>
      <td>4422.0</td>
      <td>945.0</td>
      <td>2307.0</td>
      <td>885.0</td>
      <td>2.8285</td>
    </tr>
    <tr>
      <th>13600</th>
      <td>-117.25</td>
      <td>34.16</td>
      <td>37.0</td>
      <td>1709.0</td>
      <td>278.0</td>
      <td>744.0</td>
      <td>274.0</td>
      <td>3.7188</td>
    </tr>
    <tr>
      <th>13989</th>
      <td>-117.19</td>
      <td>34.94</td>
      <td>31.0</td>
      <td>2034.0</td>
      <td>444.0</td>
      <td>1097.0</td>
      <td>367.0</td>
      <td>2.1522</td>
    </tr>
    <tr>
      <th>15168</th>
      <td>-117.06</td>
      <td>33.02</td>
      <td>24.0</td>
      <td>830.0</td>
      <td>190.0</td>
      <td>279.0</td>
      <td>196.0</td>
      <td>1.9176</td>
    </tr>
    <tr>
      <th>6747</th>
      <td>-118.07</td>
      <td>34.11</td>
      <td>41.0</td>
      <td>2869.0</td>
      <td>563.0</td>
      <td>1627.0</td>
      <td>533.0</td>
      <td>5.0736</td>
    </tr>
    <tr>
      <th>7398</th>
      <td>-118.24</td>
      <td>33.96</td>
      <td>44.0</td>
      <td>1338.0</td>
      <td>366.0</td>
      <td>1765.0</td>
      <td>388.0</td>
      <td>1.7778</td>
    </tr>
    <tr>
      <th>5562</th>
      <td>-118.28</td>
      <td>33.91</td>
      <td>41.0</td>
      <td>620.0</td>
      <td>133.0</td>
      <td>642.0</td>
      <td>162.0</td>
      <td>2.6546</td>
    </tr>
    <tr>
      <th>16121</th>
      <td>-122.46</td>
      <td>37.79</td>
      <td>52.0</td>
      <td>2059.0</td>
      <td>416.0</td>
      <td>999.0</td>
      <td>402.0</td>
      <td>3.7419</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12380</th>
      <td>-116.47</td>
      <td>33.77</td>
      <td>26.0</td>
      <td>4300.0</td>
      <td>767.0</td>
      <td>1557.0</td>
      <td>669.0</td>
      <td>4.4107</td>
    </tr>
    <tr>
      <th>5618</th>
      <td>-118.23</td>
      <td>33.78</td>
      <td>20.0</td>
      <td>59.0</td>
      <td>24.0</td>
      <td>69.0</td>
      <td>23.0</td>
      <td>2.5588</td>
    </tr>
    <tr>
      <th>10060</th>
      <td>-121.06</td>
      <td>39.25</td>
      <td>17.0</td>
      <td>3127.0</td>
      <td>539.0</td>
      <td>1390.0</td>
      <td>520.0</td>
      <td>3.9537</td>
    </tr>
    <tr>
      <th>18067</th>
      <td>-122.03</td>
      <td>37.29</td>
      <td>22.0</td>
      <td>3118.0</td>
      <td>438.0</td>
      <td>1147.0</td>
      <td>425.0</td>
      <td>10.3653</td>
    </tr>
    <tr>
      <th>4471</th>
      <td>-118.17</td>
      <td>34.09</td>
      <td>33.0</td>
      <td>2907.0</td>
      <td>797.0</td>
      <td>3212.0</td>
      <td>793.0</td>
      <td>2.2348</td>
    </tr>
    <tr>
      <th>19786</th>
      <td>-122.86</td>
      <td>40.56</td>
      <td>12.0</td>
      <td>1350.0</td>
      <td>300.0</td>
      <td>423.0</td>
      <td>172.0</td>
      <td>1.7393</td>
    </tr>
    <tr>
      <th>9969</th>
      <td>-122.48</td>
      <td>38.51</td>
      <td>49.0</td>
      <td>1977.0</td>
      <td>393.0</td>
      <td>741.0</td>
      <td>339.0</td>
      <td>3.1312</td>
    </tr>
    <tr>
      <th>14621</th>
      <td>-117.17</td>
      <td>32.78</td>
      <td>17.0</td>
      <td>3845.0</td>
      <td>1051.0</td>
      <td>3102.0</td>
      <td>944.0</td>
      <td>2.3658</td>
    </tr>
    <tr>
      <th>579</th>
      <td>-122.07</td>
      <td>37.71</td>
      <td>40.0</td>
      <td>1808.0</td>
      <td>302.0</td>
      <td>746.0</td>
      <td>270.0</td>
      <td>5.3015</td>
    </tr>
    <tr>
      <th>11682</th>
      <td>-118.01</td>
      <td>33.87</td>
      <td>25.0</td>
      <td>6348.0</td>
      <td>1615.0</td>
      <td>4188.0</td>
      <td>1497.0</td>
      <td>3.1390</td>
    </tr>
    <tr>
      <th>245</th>
      <td>-122.21</td>
      <td>37.78</td>
      <td>43.0</td>
      <td>1702.0</td>
      <td>460.0</td>
      <td>1227.0</td>
      <td>407.0</td>
      <td>1.7188</td>
    </tr>
    <tr>
      <th>12130</th>
      <td>-117.23</td>
      <td>33.94</td>
      <td>8.0</td>
      <td>2405.0</td>
      <td>537.0</td>
      <td>1594.0</td>
      <td>517.0</td>
      <td>3.0789</td>
    </tr>
    <tr>
      <th>16441</th>
      <td>-121.29</td>
      <td>38.14</td>
      <td>34.0</td>
      <td>2770.0</td>
      <td>544.0</td>
      <td>1409.0</td>
      <td>535.0</td>
      <td>3.2338</td>
    </tr>
    <tr>
      <th>11016</th>
      <td>-117.82</td>
      <td>33.76</td>
      <td>33.0</td>
      <td>2774.0</td>
      <td>428.0</td>
      <td>1229.0</td>
      <td>407.0</td>
      <td>6.2944</td>
    </tr>
    <tr>
      <th>19934</th>
      <td>-119.34</td>
      <td>36.31</td>
      <td>14.0</td>
      <td>1635.0</td>
      <td>422.0</td>
      <td>870.0</td>
      <td>399.0</td>
      <td>2.7000</td>
    </tr>
    <tr>
      <th>1364</th>
      <td>-122.14</td>
      <td>38.03</td>
      <td>42.0</td>
      <td>118.0</td>
      <td>34.0</td>
      <td>54.0</td>
      <td>30.0</td>
      <td>2.5795</td>
    </tr>
    <tr>
      <th>1236</th>
      <td>-120.37</td>
      <td>38.23</td>
      <td>13.0</td>
      <td>4401.0</td>
      <td>829.0</td>
      <td>924.0</td>
      <td>383.0</td>
      <td>2.6942</td>
    </tr>
    <tr>
      <th>5364</th>
      <td>-118.42</td>
      <td>34.04</td>
      <td>52.0</td>
      <td>1358.0</td>
      <td>272.0</td>
      <td>574.0</td>
      <td>267.0</td>
      <td>5.6454</td>
    </tr>
    <tr>
      <th>11703</th>
      <td>-117.97</td>
      <td>33.88</td>
      <td>16.0</td>
      <td>2003.0</td>
      <td>300.0</td>
      <td>1172.0</td>
      <td>318.0</td>
      <td>6.0394</td>
    </tr>
    <tr>
      <th>10356</th>
      <td>-117.67</td>
      <td>33.60</td>
      <td>25.0</td>
      <td>3164.0</td>
      <td>449.0</td>
      <td>1517.0</td>
      <td>453.0</td>
      <td>6.7921</td>
    </tr>
    <tr>
      <th>15270</th>
      <td>-117.29</td>
      <td>33.08</td>
      <td>18.0</td>
      <td>3225.0</td>
      <td>515.0</td>
      <td>1463.0</td>
      <td>476.0</td>
      <td>5.7787</td>
    </tr>
    <tr>
      <th>3754</th>
      <td>-118.37</td>
      <td>34.18</td>
      <td>36.0</td>
      <td>1608.0</td>
      <td>373.0</td>
      <td>1217.0</td>
      <td>374.0</td>
      <td>2.9728</td>
    </tr>
    <tr>
      <th>12166</th>
      <td>-117.14</td>
      <td>33.81</td>
      <td>13.0</td>
      <td>4496.0</td>
      <td>756.0</td>
      <td>2044.0</td>
      <td>695.0</td>
      <td>3.2778</td>
    </tr>
    <tr>
      <th>6003</th>
      <td>-117.77</td>
      <td>34.08</td>
      <td>27.0</td>
      <td>5929.0</td>
      <td>932.0</td>
      <td>2817.0</td>
      <td>828.0</td>
      <td>6.0434</td>
    </tr>
    <tr>
      <th>7364</th>
      <td>-118.20</td>
      <td>33.97</td>
      <td>43.0</td>
      <td>825.0</td>
      <td>212.0</td>
      <td>820.0</td>
      <td>184.0</td>
      <td>1.8897</td>
    </tr>
    <tr>
      <th>6563</th>
      <td>-118.13</td>
      <td>34.20</td>
      <td>46.0</td>
      <td>1271.0</td>
      <td>236.0</td>
      <td>573.0</td>
      <td>210.0</td>
      <td>4.9312</td>
    </tr>
    <tr>
      <th>12053</th>
      <td>-117.56</td>
      <td>33.88</td>
      <td>40.0</td>
      <td>1196.0</td>
      <td>294.0</td>
      <td>1052.0</td>
      <td>258.0</td>
      <td>2.0682</td>
    </tr>
    <tr>
      <th>13908</th>
      <td>-116.40</td>
      <td>34.09</td>
      <td>9.0</td>
      <td>4855.0</td>
      <td>872.0</td>
      <td>2098.0</td>
      <td>765.0</td>
      <td>3.2723</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>-118.01</td>
      <td>33.82</td>
      <td>31.0</td>
      <td>1960.0</td>
      <td>380.0</td>
      <td>1356.0</td>
      <td>356.0</td>
      <td>4.0625</td>
    </tr>
    <tr>
      <th>15775</th>
      <td>-122.45</td>
      <td>37.77</td>
      <td>52.0</td>
      <td>3095.0</td>
      <td>682.0</td>
      <td>1269.0</td>
      <td>639.0</td>
      <td>3.5750</td>
    </tr>
  </tbody>
</table>
<p>16512 rows × 8 columns</p>
</div>


