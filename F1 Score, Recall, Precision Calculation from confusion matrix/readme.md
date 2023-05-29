```python
import warnings
warnings.filterwarnings("ignore")
```


```python
import sklearn
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

```


```python
ml_list={"DT":sklearn.tree.DecisionTreeClassifier()}#,"SVC":SVC()}}
```


```python
loop1="./csvs/iris_train.csv"
loop2="./csvs/iris_test.csv"
output_csv="./results/100R.csv"
dname="IRIS_100_times"
ii="DT"
```


```python

df = pd.read_csv(loop1)#,header=None )
df=df.fillna(0)
X_train =df[df.columns[0:-1]]
X_train=np.array(X_train)
df[df.columns[-1]] = df[df.columns[-1]].astype('category')
y_train=df[df.columns[-1]].cat.codes  


df = pd.read_csv(loop2)#,header=None )
df=df.fillna(0)
X_test =df[df.columns[0:-1]]
X_test=np.array(X_test)
df[df.columns[-1]] = df[df.columns[-1]].astype('category')
y_test=df[df.columns[-1]].cat.codes  



cv=0


#dname=loop1  [6:-13]  
results_y=[]
cv+=1
results_y.append(y_test)


precision=[]
recall=[]
f1=[]
accuracy=[]
train_time=[]
test_time=[]
total_time=[]
kappa=[]
accuracy_b=[]

clf = ml_list[ii]

clf.fit(X_train, y_train)


predict =clf.predict(X_test)


df=sklearn.metrics.classification_report(y_test, predict, output_dict=True)     
df = pd.DataFrame(df).transpose()



cm = pd.DataFrame(sklearn.metrics.confusion_matrix(y_test, predict))
rc=sklearn.metrics.recall_score(y_test, predict,average= "macro")
pr=sklearn.metrics.precision_score(y_test, predict,average= "macro")
f_1=sklearn.metrics.f1_score(y_test, predict,average= "macro")     
accuracy=sklearn.metrics.accuracy_score(y_test, predict)

```

# Result of Sklearn


```python
print(df)
```

                  precision    recall  f1-score    support
    0              1.000000  1.000000  1.000000  13.000000
    1              0.800000  1.000000  0.888889  12.000000
    2              1.000000  0.769231  0.869565  13.000000
    accuracy       0.921053  0.921053  0.921053   0.921053
    macro avg      0.933333  0.923077  0.919485  38.000000
    weighted avg   0.936842  0.921053  0.920290  38.000000
    


```python

```

# F1 Score, Recall, Precision Calculation from confusion matrix    


```python
def scores(cm, i):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for j in range(len(cm)):
        print
        if (i == j):
            # true positive
            TP += cm[i, j]
            tmp = np.delete(cm, i, 0)
            tmp = np.delete(tmp, j, 1)
            # true negative
            TN += np.sum(tmp)
        else:
            if (cm[i, j] != 0):
                # false negative
                FN += cm[i, j]
            if (cm[j, i] != 0):
                # false positive
                FP += cm[j, i]
    recall = TP / (FN + TP)
    precision = TP / (TP + FP)
    f1_score = 2 * 1/(1/recall + 1/precision)
    
    return [recall,precision,f1_score]
```


```python
d_list=[]
for i in cm.values:
    print(list(i))
    d_list.append(list(i))
d_list=np.array(d_list)

```

    [13, 0, 0]
    [0, 12, 0]
    [0, 3, 10]
    


```python
class_based=[]
for i in range(len(cm)):
    temp=[f"class {i}"]
    temp=temp+scores(d_list,i)
    class_based.append(temp)
df = pd.DataFrame(class_based,columns=["Class",'recall', 'precision', 'f1_score'])
```


```python

```


```python
average=dict(df.mean())
average["accuracy"]=np.trace(d_list)/d_list.sum()
```

# Result of Algorithm


```python
df
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
      <th>Class</th>
      <th>recall</th>
      <th>precision</th>
      <th>f1_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>class 0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>class 1</td>
      <td>1.000000</td>
      <td>0.8</td>
      <td>0.888889</td>
    </tr>
    <tr>
      <th>2</th>
      <td>class 2</td>
      <td>0.769231</td>
      <td>1.0</td>
      <td>0.869565</td>
    </tr>
  </tbody>
</table>
</div>




```python
average
```




    {'recall': 0.923076923076923,
     'precision': 0.9333333333333332,
     'f1_score': 0.9194847020933977,
     'accuracy': 0.9210526315789473}


