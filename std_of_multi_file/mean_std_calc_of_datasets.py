
# coding: utf-8

# # calculate the mean and standard deviation of multiple csv files

# In[27]:


import numpy as np
import os
import pandas as pd


# ## find your csv

# In[28]:


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


# ## standard deviation

# In[29]:


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


# In[30]:


std


# ## take its transpose and save

# In[31]:


std=std.T
std


# In[32]:


std.to_csv("std.csv",index=None)
std=std.round(3)


# # Mean

# In[33]:


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


# In[34]:


mean


# ## take its transpose and save

# In[35]:


mean=mean.T
mean


# ##  merging columns

# In[36]:


mean.to_csv("mean.csv",index=None)
mean=mean.round(3)


# In[37]:


merged=pd.DataFrame()


# In[42]:


for i in mean.columns:
    merged[i]=mean[i].astype(str)+"±"+std[i].astype(str)


merged


# In[43]:


merged.to_csv("merged.csv",index=None)

