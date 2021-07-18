
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os


# ### Discovering csv extension files under "csvs" folder.

# In[2]:


def find_the_way(path,file_format):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))  
    return files_add
name_list=find_the_way('./csvs','.csv')
name_list


# In[6]:


name="output.csv"
df=pd.read_csv(name_list[0])
col_names=list(df.columns)


empty = pd.DataFrame(columns=col_names)
empty.to_csv(name, mode="a", index=False)#,header=False)

for iii in name_list:
    df=pd.read_csv(iii)
    print("name and shape of dataframe :",iii,df.shape)
    df.to_csv(name, mode="a", index=False,header=False)


df=pd.read_csv(name)

print("\n\n\nname and shape of dataframe :",name,df.shape)

