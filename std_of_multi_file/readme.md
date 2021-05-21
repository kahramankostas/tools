
# calculate the mean and standard deviation of multiple csv files


```python
import numpy as np
import os
import pandas as pd
```

## find your csv


```python
def find_the_way(path,file_format):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))  
    return files_add
name_list=find_the_way('./csvs/','.csv')
name_list
```




    ['./csvs/file 1.csv',
     './csvs/file 10.csv',
     './csvs/file 2.csv',
     './csvs/file 3.csv',
     './csvs/file 4.csv',
     './csvs/file 5.csv',
     './csvs/file 6.csv',
     './csvs/file 7.csv',
     './csvs/file 8.csv',
     './csvs/file 9.csv']



## standard deviation


```python
flag=1
for i in name_list:
    df = pd.read_csv(i) 
    col=i[7:-4]
    temp=pd.DataFrame(df.std(),columns=[col])
    if flag:
        std=temp
        flag=0
    else:
        std[col]=temp[col]
```


```python
std
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file 1</th>
      <th>file 10</th>
      <th>file 2</th>
      <th>file 3</th>
      <th>file 4</th>
      <th>file 5</th>
      <th>file 6</th>
      <th>file 7</th>
      <th>file 8</th>
      <th>file 9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Feature 1</th>
      <td>30.802812</td>
      <td>30.802812</td>
      <td>30.802812</td>
      <td>30.802812</td>
      <td>30.802812</td>
      <td>30.802812</td>
      <td>30.802812</td>
      <td>30.802812</td>
      <td>30.802812</td>
      <td>30.802812</td>
    </tr>
    <tr>
      <th>Feature 2</th>
      <td>29.263664</td>
      <td>29.263664</td>
      <td>29.263664</td>
      <td>29.263664</td>
      <td>29.263664</td>
      <td>29.263664</td>
      <td>29.263664</td>
      <td>29.263664</td>
      <td>29.263664</td>
      <td>29.263664</td>
    </tr>
    <tr>
      <th>Feature 3</th>
      <td>29.482162</td>
      <td>29.482162</td>
      <td>29.482162</td>
      <td>29.482162</td>
      <td>29.482162</td>
      <td>29.482162</td>
      <td>29.482162</td>
      <td>29.482162</td>
      <td>29.482162</td>
      <td>29.482162</td>
    </tr>
    <tr>
      <th>Feature 4</th>
      <td>27.375798</td>
      <td>27.375798</td>
      <td>27.375798</td>
      <td>27.375798</td>
      <td>27.375798</td>
      <td>27.375798</td>
      <td>27.375798</td>
      <td>27.375798</td>
      <td>27.375798</td>
      <td>27.375798</td>
    </tr>
    <tr>
      <th>Feature 5</th>
      <td>29.116318</td>
      <td>29.116318</td>
      <td>29.116318</td>
      <td>29.116318</td>
      <td>29.116318</td>
      <td>29.116318</td>
      <td>29.116318</td>
      <td>29.116318</td>
      <td>29.116318</td>
      <td>29.116318</td>
    </tr>
    <tr>
      <th>Feature 6</th>
      <td>29.296001</td>
      <td>29.296001</td>
      <td>29.296001</td>
      <td>29.296001</td>
      <td>29.296001</td>
      <td>29.296001</td>
      <td>29.296001</td>
      <td>29.296001</td>
      <td>29.296001</td>
      <td>29.296001</td>
    </tr>
    <tr>
      <th>Feature 7</th>
      <td>30.024251</td>
      <td>30.024251</td>
      <td>30.024251</td>
      <td>30.024251</td>
      <td>30.024251</td>
      <td>30.024251</td>
      <td>30.024251</td>
      <td>30.024251</td>
      <td>30.024251</td>
      <td>30.024251</td>
    </tr>
    <tr>
      <th>Feature 8</th>
      <td>30.522858</td>
      <td>30.522858</td>
      <td>30.522858</td>
      <td>30.522858</td>
      <td>30.522858</td>
      <td>30.522858</td>
      <td>30.522858</td>
      <td>30.522858</td>
      <td>30.522858</td>
      <td>30.522858</td>
    </tr>
    <tr>
      <th>Feature 9</th>
      <td>28.809202</td>
      <td>28.809202</td>
      <td>28.809202</td>
      <td>28.809202</td>
      <td>28.809202</td>
      <td>28.809202</td>
      <td>28.809202</td>
      <td>28.809202</td>
      <td>28.809202</td>
      <td>28.809202</td>
    </tr>
    <tr>
      <th>Feature 10</th>
      <td>30.759307</td>
      <td>30.759307</td>
      <td>30.759307</td>
      <td>30.759307</td>
      <td>30.759307</td>
      <td>30.759307</td>
      <td>30.759307</td>
      <td>30.759307</td>
      <td>30.759307</td>
      <td>30.759307</td>
    </tr>
    <tr>
      <th>Feature 11</th>
      <td>29.139045</td>
      <td>29.139045</td>
      <td>29.139045</td>
      <td>29.139045</td>
      <td>29.139045</td>
      <td>29.139045</td>
      <td>29.139045</td>
      <td>29.139045</td>
      <td>29.139045</td>
      <td>29.139045</td>
    </tr>
    <tr>
      <th>Feature 12</th>
      <td>30.880310</td>
      <td>30.880310</td>
      <td>30.880310</td>
      <td>30.880310</td>
      <td>30.880310</td>
      <td>30.880310</td>
      <td>30.880310</td>
      <td>30.880310</td>
      <td>30.880310</td>
      <td>30.880310</td>
    </tr>
    <tr>
      <th>Feature 13</th>
      <td>30.875787</td>
      <td>30.875787</td>
      <td>30.875787</td>
      <td>30.875787</td>
      <td>30.875787</td>
      <td>30.875787</td>
      <td>30.875787</td>
      <td>30.875787</td>
      <td>30.875787</td>
      <td>30.875787</td>
    </tr>
    <tr>
      <th>Feature 14</th>
      <td>30.225982</td>
      <td>30.225982</td>
      <td>30.225982</td>
      <td>30.225982</td>
      <td>30.225982</td>
      <td>30.225982</td>
      <td>30.225982</td>
      <td>30.225982</td>
      <td>30.225982</td>
      <td>30.225982</td>
    </tr>
  </tbody>
</table>
</div>



## take its transpose and save


```python
std=std.T
std
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature 1</th>
      <th>Feature 2</th>
      <th>Feature 3</th>
      <th>Feature 4</th>
      <th>Feature 5</th>
      <th>Feature 6</th>
      <th>Feature 7</th>
      <th>Feature 8</th>
      <th>Feature 9</th>
      <th>Feature 10</th>
      <th>Feature 11</th>
      <th>Feature 12</th>
      <th>Feature 13</th>
      <th>Feature 14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>file 1</th>
      <td>30.802812</td>
      <td>29.263664</td>
      <td>29.482162</td>
      <td>27.375798</td>
      <td>29.116318</td>
      <td>29.296001</td>
      <td>30.024251</td>
      <td>30.522858</td>
      <td>28.809202</td>
      <td>30.759307</td>
      <td>29.139045</td>
      <td>30.88031</td>
      <td>30.875787</td>
      <td>30.225982</td>
    </tr>
    <tr>
      <th>file 10</th>
      <td>30.802812</td>
      <td>29.263664</td>
      <td>29.482162</td>
      <td>27.375798</td>
      <td>29.116318</td>
      <td>29.296001</td>
      <td>30.024251</td>
      <td>30.522858</td>
      <td>28.809202</td>
      <td>30.759307</td>
      <td>29.139045</td>
      <td>30.88031</td>
      <td>30.875787</td>
      <td>30.225982</td>
    </tr>
    <tr>
      <th>file 2</th>
      <td>30.802812</td>
      <td>29.263664</td>
      <td>29.482162</td>
      <td>27.375798</td>
      <td>29.116318</td>
      <td>29.296001</td>
      <td>30.024251</td>
      <td>30.522858</td>
      <td>28.809202</td>
      <td>30.759307</td>
      <td>29.139045</td>
      <td>30.88031</td>
      <td>30.875787</td>
      <td>30.225982</td>
    </tr>
    <tr>
      <th>file 3</th>
      <td>30.802812</td>
      <td>29.263664</td>
      <td>29.482162</td>
      <td>27.375798</td>
      <td>29.116318</td>
      <td>29.296001</td>
      <td>30.024251</td>
      <td>30.522858</td>
      <td>28.809202</td>
      <td>30.759307</td>
      <td>29.139045</td>
      <td>30.88031</td>
      <td>30.875787</td>
      <td>30.225982</td>
    </tr>
    <tr>
      <th>file 4</th>
      <td>30.802812</td>
      <td>29.263664</td>
      <td>29.482162</td>
      <td>27.375798</td>
      <td>29.116318</td>
      <td>29.296001</td>
      <td>30.024251</td>
      <td>30.522858</td>
      <td>28.809202</td>
      <td>30.759307</td>
      <td>29.139045</td>
      <td>30.88031</td>
      <td>30.875787</td>
      <td>30.225982</td>
    </tr>
    <tr>
      <th>file 5</th>
      <td>30.802812</td>
      <td>29.263664</td>
      <td>29.482162</td>
      <td>27.375798</td>
      <td>29.116318</td>
      <td>29.296001</td>
      <td>30.024251</td>
      <td>30.522858</td>
      <td>28.809202</td>
      <td>30.759307</td>
      <td>29.139045</td>
      <td>30.88031</td>
      <td>30.875787</td>
      <td>30.225982</td>
    </tr>
    <tr>
      <th>file 6</th>
      <td>30.802812</td>
      <td>29.263664</td>
      <td>29.482162</td>
      <td>27.375798</td>
      <td>29.116318</td>
      <td>29.296001</td>
      <td>30.024251</td>
      <td>30.522858</td>
      <td>28.809202</td>
      <td>30.759307</td>
      <td>29.139045</td>
      <td>30.88031</td>
      <td>30.875787</td>
      <td>30.225982</td>
    </tr>
    <tr>
      <th>file 7</th>
      <td>30.802812</td>
      <td>29.263664</td>
      <td>29.482162</td>
      <td>27.375798</td>
      <td>29.116318</td>
      <td>29.296001</td>
      <td>30.024251</td>
      <td>30.522858</td>
      <td>28.809202</td>
      <td>30.759307</td>
      <td>29.139045</td>
      <td>30.88031</td>
      <td>30.875787</td>
      <td>30.225982</td>
    </tr>
    <tr>
      <th>file 8</th>
      <td>30.802812</td>
      <td>29.263664</td>
      <td>29.482162</td>
      <td>27.375798</td>
      <td>29.116318</td>
      <td>29.296001</td>
      <td>30.024251</td>
      <td>30.522858</td>
      <td>28.809202</td>
      <td>30.759307</td>
      <td>29.139045</td>
      <td>30.88031</td>
      <td>30.875787</td>
      <td>30.225982</td>
    </tr>
    <tr>
      <th>file 9</th>
      <td>30.802812</td>
      <td>29.263664</td>
      <td>29.482162</td>
      <td>27.375798</td>
      <td>29.116318</td>
      <td>29.296001</td>
      <td>30.024251</td>
      <td>30.522858</td>
      <td>28.809202</td>
      <td>30.759307</td>
      <td>29.139045</td>
      <td>30.88031</td>
      <td>30.875787</td>
      <td>30.225982</td>
    </tr>
  </tbody>
</table>
</div>




```python
std.to_csv("std.csv",index=None)
std=std.round(3)
```

# Mean


```python
flag=1
for i in name_list:
    df = pd.read_csv(i) 
    col=i[7:-4]
    temp=pd.DataFrame(df.mean(),columns=[col])
    if flag:
        mean=temp
        flag=0
    else:
        mean[col]=temp[col]
```


```python
mean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file 1</th>
      <th>file 10</th>
      <th>file 2</th>
      <th>file 3</th>
      <th>file 4</th>
      <th>file 5</th>
      <th>file 6</th>
      <th>file 7</th>
      <th>file 8</th>
      <th>file 9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Feature 1</th>
      <td>44.07</td>
      <td>44.07</td>
      <td>44.07</td>
      <td>44.07</td>
      <td>44.07</td>
      <td>44.07</td>
      <td>44.07</td>
      <td>44.07</td>
      <td>44.07</td>
      <td>44.07</td>
    </tr>
    <tr>
      <th>Feature 2</th>
      <td>53.04</td>
      <td>53.04</td>
      <td>53.04</td>
      <td>53.04</td>
      <td>53.04</td>
      <td>53.04</td>
      <td>53.04</td>
      <td>53.04</td>
      <td>53.04</td>
      <td>53.04</td>
    </tr>
    <tr>
      <th>Feature 3</th>
      <td>48.29</td>
      <td>48.29</td>
      <td>48.29</td>
      <td>48.29</td>
      <td>48.29</td>
      <td>48.29</td>
      <td>48.29</td>
      <td>48.29</td>
      <td>48.29</td>
      <td>48.29</td>
    </tr>
    <tr>
      <th>Feature 4</th>
      <td>44.80</td>
      <td>44.80</td>
      <td>44.80</td>
      <td>44.80</td>
      <td>44.80</td>
      <td>44.80</td>
      <td>44.80</td>
      <td>44.80</td>
      <td>44.80</td>
      <td>44.80</td>
    </tr>
    <tr>
      <th>Feature 5</th>
      <td>41.76</td>
      <td>41.76</td>
      <td>41.76</td>
      <td>41.76</td>
      <td>41.76</td>
      <td>41.76</td>
      <td>41.76</td>
      <td>41.76</td>
      <td>41.76</td>
      <td>41.76</td>
    </tr>
    <tr>
      <th>Feature 6</th>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
    </tr>
    <tr>
      <th>Feature 7</th>
      <td>48.83</td>
      <td>48.83</td>
      <td>48.83</td>
      <td>48.83</td>
      <td>48.83</td>
      <td>48.83</td>
      <td>48.83</td>
      <td>48.83</td>
      <td>48.83</td>
      <td>48.83</td>
    </tr>
    <tr>
      <th>Feature 8</th>
      <td>56.46</td>
      <td>56.46</td>
      <td>56.46</td>
      <td>56.46</td>
      <td>56.46</td>
      <td>56.46</td>
      <td>56.46</td>
      <td>56.46</td>
      <td>56.46</td>
      <td>56.46</td>
    </tr>
    <tr>
      <th>Feature 9</th>
      <td>51.36</td>
      <td>51.36</td>
      <td>51.36</td>
      <td>51.36</td>
      <td>51.36</td>
      <td>51.36</td>
      <td>51.36</td>
      <td>51.36</td>
      <td>51.36</td>
      <td>51.36</td>
    </tr>
    <tr>
      <th>Feature 10</th>
      <td>49.92</td>
      <td>49.92</td>
      <td>49.92</td>
      <td>49.92</td>
      <td>49.92</td>
      <td>49.92</td>
      <td>49.92</td>
      <td>49.92</td>
      <td>49.92</td>
      <td>49.92</td>
    </tr>
    <tr>
      <th>Feature 11</th>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
      <td>50.87</td>
    </tr>
    <tr>
      <th>Feature 12</th>
      <td>49.32</td>
      <td>49.32</td>
      <td>49.32</td>
      <td>49.32</td>
      <td>49.32</td>
      <td>49.32</td>
      <td>49.32</td>
      <td>49.32</td>
      <td>49.32</td>
      <td>49.32</td>
    </tr>
    <tr>
      <th>Feature 13</th>
      <td>48.67</td>
      <td>48.67</td>
      <td>48.67</td>
      <td>48.67</td>
      <td>48.67</td>
      <td>48.67</td>
      <td>48.67</td>
      <td>48.67</td>
      <td>48.67</td>
      <td>48.67</td>
    </tr>
    <tr>
      <th>Feature 14</th>
      <td>50.19</td>
      <td>50.19</td>
      <td>50.19</td>
      <td>50.19</td>
      <td>50.19</td>
      <td>50.19</td>
      <td>50.19</td>
      <td>50.19</td>
      <td>50.19</td>
      <td>50.19</td>
    </tr>
  </tbody>
</table>
</div>



## take its transpose and save


```python
mean=mean.T
mean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature 1</th>
      <th>Feature 2</th>
      <th>Feature 3</th>
      <th>Feature 4</th>
      <th>Feature 5</th>
      <th>Feature 6</th>
      <th>Feature 7</th>
      <th>Feature 8</th>
      <th>Feature 9</th>
      <th>Feature 10</th>
      <th>Feature 11</th>
      <th>Feature 12</th>
      <th>Feature 13</th>
      <th>Feature 14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>file 1</th>
      <td>44.07</td>
      <td>53.04</td>
      <td>48.29</td>
      <td>44.8</td>
      <td>41.76</td>
      <td>50.87</td>
      <td>48.83</td>
      <td>56.46</td>
      <td>51.36</td>
      <td>49.92</td>
      <td>50.87</td>
      <td>49.32</td>
      <td>48.67</td>
      <td>50.19</td>
    </tr>
    <tr>
      <th>file 10</th>
      <td>44.07</td>
      <td>53.04</td>
      <td>48.29</td>
      <td>44.8</td>
      <td>41.76</td>
      <td>50.87</td>
      <td>48.83</td>
      <td>56.46</td>
      <td>51.36</td>
      <td>49.92</td>
      <td>50.87</td>
      <td>49.32</td>
      <td>48.67</td>
      <td>50.19</td>
    </tr>
    <tr>
      <th>file 2</th>
      <td>44.07</td>
      <td>53.04</td>
      <td>48.29</td>
      <td>44.8</td>
      <td>41.76</td>
      <td>50.87</td>
      <td>48.83</td>
      <td>56.46</td>
      <td>51.36</td>
      <td>49.92</td>
      <td>50.87</td>
      <td>49.32</td>
      <td>48.67</td>
      <td>50.19</td>
    </tr>
    <tr>
      <th>file 3</th>
      <td>44.07</td>
      <td>53.04</td>
      <td>48.29</td>
      <td>44.8</td>
      <td>41.76</td>
      <td>50.87</td>
      <td>48.83</td>
      <td>56.46</td>
      <td>51.36</td>
      <td>49.92</td>
      <td>50.87</td>
      <td>49.32</td>
      <td>48.67</td>
      <td>50.19</td>
    </tr>
    <tr>
      <th>file 4</th>
      <td>44.07</td>
      <td>53.04</td>
      <td>48.29</td>
      <td>44.8</td>
      <td>41.76</td>
      <td>50.87</td>
      <td>48.83</td>
      <td>56.46</td>
      <td>51.36</td>
      <td>49.92</td>
      <td>50.87</td>
      <td>49.32</td>
      <td>48.67</td>
      <td>50.19</td>
    </tr>
    <tr>
      <th>file 5</th>
      <td>44.07</td>
      <td>53.04</td>
      <td>48.29</td>
      <td>44.8</td>
      <td>41.76</td>
      <td>50.87</td>
      <td>48.83</td>
      <td>56.46</td>
      <td>51.36</td>
      <td>49.92</td>
      <td>50.87</td>
      <td>49.32</td>
      <td>48.67</td>
      <td>50.19</td>
    </tr>
    <tr>
      <th>file 6</th>
      <td>44.07</td>
      <td>53.04</td>
      <td>48.29</td>
      <td>44.8</td>
      <td>41.76</td>
      <td>50.87</td>
      <td>48.83</td>
      <td>56.46</td>
      <td>51.36</td>
      <td>49.92</td>
      <td>50.87</td>
      <td>49.32</td>
      <td>48.67</td>
      <td>50.19</td>
    </tr>
    <tr>
      <th>file 7</th>
      <td>44.07</td>
      <td>53.04</td>
      <td>48.29</td>
      <td>44.8</td>
      <td>41.76</td>
      <td>50.87</td>
      <td>48.83</td>
      <td>56.46</td>
      <td>51.36</td>
      <td>49.92</td>
      <td>50.87</td>
      <td>49.32</td>
      <td>48.67</td>
      <td>50.19</td>
    </tr>
    <tr>
      <th>file 8</th>
      <td>44.07</td>
      <td>53.04</td>
      <td>48.29</td>
      <td>44.8</td>
      <td>41.76</td>
      <td>50.87</td>
      <td>48.83</td>
      <td>56.46</td>
      <td>51.36</td>
      <td>49.92</td>
      <td>50.87</td>
      <td>49.32</td>
      <td>48.67</td>
      <td>50.19</td>
    </tr>
    <tr>
      <th>file 9</th>
      <td>44.07</td>
      <td>53.04</td>
      <td>48.29</td>
      <td>44.8</td>
      <td>41.76</td>
      <td>50.87</td>
      <td>48.83</td>
      <td>56.46</td>
      <td>51.36</td>
      <td>49.92</td>
      <td>50.87</td>
      <td>49.32</td>
      <td>48.67</td>
      <td>50.19</td>
    </tr>
  </tbody>
</table>
</div>



##  merging columns


```python
mean.to_csv("mean.csv",index=None)
mean=mean.round(3)
```


```python
merged=pd.DataFrame()
```


```python
for i in mean.columns:
    merged[i]=mean[i].astype(str)+"±"+std[i].astype(str)


merged
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature 1</th>
      <th>Feature 2</th>
      <th>Feature 3</th>
      <th>Feature 4</th>
      <th>Feature 5</th>
      <th>Feature 6</th>
      <th>Feature 7</th>
      <th>Feature 8</th>
      <th>Feature 9</th>
      <th>Feature 10</th>
      <th>Feature 11</th>
      <th>Feature 12</th>
      <th>Feature 13</th>
      <th>Feature 14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>file 1</th>
      <td>44.07±30.803</td>
      <td>53.04±29.264</td>
      <td>48.29±29.482</td>
      <td>44.8±27.376</td>
      <td>41.76±29.116</td>
      <td>50.87±29.296</td>
      <td>48.83±30.024</td>
      <td>56.46±30.523</td>
      <td>51.36±28.809</td>
      <td>49.92±30.759</td>
      <td>50.87±29.139</td>
      <td>49.32±30.88</td>
      <td>48.67±30.876</td>
      <td>50.19±30.226</td>
    </tr>
    <tr>
      <th>file 10</th>
      <td>44.07±30.803</td>
      <td>53.04±29.264</td>
      <td>48.29±29.482</td>
      <td>44.8±27.376</td>
      <td>41.76±29.116</td>
      <td>50.87±29.296</td>
      <td>48.83±30.024</td>
      <td>56.46±30.523</td>
      <td>51.36±28.809</td>
      <td>49.92±30.759</td>
      <td>50.87±29.139</td>
      <td>49.32±30.88</td>
      <td>48.67±30.876</td>
      <td>50.19±30.226</td>
    </tr>
    <tr>
      <th>file 2</th>
      <td>44.07±30.803</td>
      <td>53.04±29.264</td>
      <td>48.29±29.482</td>
      <td>44.8±27.376</td>
      <td>41.76±29.116</td>
      <td>50.87±29.296</td>
      <td>48.83±30.024</td>
      <td>56.46±30.523</td>
      <td>51.36±28.809</td>
      <td>49.92±30.759</td>
      <td>50.87±29.139</td>
      <td>49.32±30.88</td>
      <td>48.67±30.876</td>
      <td>50.19±30.226</td>
    </tr>
    <tr>
      <th>file 3</th>
      <td>44.07±30.803</td>
      <td>53.04±29.264</td>
      <td>48.29±29.482</td>
      <td>44.8±27.376</td>
      <td>41.76±29.116</td>
      <td>50.87±29.296</td>
      <td>48.83±30.024</td>
      <td>56.46±30.523</td>
      <td>51.36±28.809</td>
      <td>49.92±30.759</td>
      <td>50.87±29.139</td>
      <td>49.32±30.88</td>
      <td>48.67±30.876</td>
      <td>50.19±30.226</td>
    </tr>
    <tr>
      <th>file 4</th>
      <td>44.07±30.803</td>
      <td>53.04±29.264</td>
      <td>48.29±29.482</td>
      <td>44.8±27.376</td>
      <td>41.76±29.116</td>
      <td>50.87±29.296</td>
      <td>48.83±30.024</td>
      <td>56.46±30.523</td>
      <td>51.36±28.809</td>
      <td>49.92±30.759</td>
      <td>50.87±29.139</td>
      <td>49.32±30.88</td>
      <td>48.67±30.876</td>
      <td>50.19±30.226</td>
    </tr>
    <tr>
      <th>file 5</th>
      <td>44.07±30.803</td>
      <td>53.04±29.264</td>
      <td>48.29±29.482</td>
      <td>44.8±27.376</td>
      <td>41.76±29.116</td>
      <td>50.87±29.296</td>
      <td>48.83±30.024</td>
      <td>56.46±30.523</td>
      <td>51.36±28.809</td>
      <td>49.92±30.759</td>
      <td>50.87±29.139</td>
      <td>49.32±30.88</td>
      <td>48.67±30.876</td>
      <td>50.19±30.226</td>
    </tr>
    <tr>
      <th>file 6</th>
      <td>44.07±30.803</td>
      <td>53.04±29.264</td>
      <td>48.29±29.482</td>
      <td>44.8±27.376</td>
      <td>41.76±29.116</td>
      <td>50.87±29.296</td>
      <td>48.83±30.024</td>
      <td>56.46±30.523</td>
      <td>51.36±28.809</td>
      <td>49.92±30.759</td>
      <td>50.87±29.139</td>
      <td>49.32±30.88</td>
      <td>48.67±30.876</td>
      <td>50.19±30.226</td>
    </tr>
    <tr>
      <th>file 7</th>
      <td>44.07±30.803</td>
      <td>53.04±29.264</td>
      <td>48.29±29.482</td>
      <td>44.8±27.376</td>
      <td>41.76±29.116</td>
      <td>50.87±29.296</td>
      <td>48.83±30.024</td>
      <td>56.46±30.523</td>
      <td>51.36±28.809</td>
      <td>49.92±30.759</td>
      <td>50.87±29.139</td>
      <td>49.32±30.88</td>
      <td>48.67±30.876</td>
      <td>50.19±30.226</td>
    </tr>
    <tr>
      <th>file 8</th>
      <td>44.07±30.803</td>
      <td>53.04±29.264</td>
      <td>48.29±29.482</td>
      <td>44.8±27.376</td>
      <td>41.76±29.116</td>
      <td>50.87±29.296</td>
      <td>48.83±30.024</td>
      <td>56.46±30.523</td>
      <td>51.36±28.809</td>
      <td>49.92±30.759</td>
      <td>50.87±29.139</td>
      <td>49.32±30.88</td>
      <td>48.67±30.876</td>
      <td>50.19±30.226</td>
    </tr>
    <tr>
      <th>file 9</th>
      <td>44.07±30.803</td>
      <td>53.04±29.264</td>
      <td>48.29±29.482</td>
      <td>44.8±27.376</td>
      <td>41.76±29.116</td>
      <td>50.87±29.296</td>
      <td>48.83±30.024</td>
      <td>56.46±30.523</td>
      <td>51.36±28.809</td>
      <td>49.92±30.759</td>
      <td>50.87±29.139</td>
      <td>49.32±30.88</td>
      <td>48.67±30.876</td>
      <td>50.19±30.226</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged.to_csv("merged.csv",index=None)
```
