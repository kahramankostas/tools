
# coding: utf-8

# # calculate the mean and standard deviation of multiple csv files

# In[1]:


import numpy as np
import os
import pandas as pd


# ## find your csv

# In[2]:


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

# In[3]:


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


# In[4]:


std


# ## take its transpose and save

# In[5]:


std=std.T
std


# In[6]:


std.to_csv("std.csv",index=None)


# # Mean

# In[7]:


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


# In[8]:


mean


# ## take its transpose and save

# In[9]:


mean=mean.T
mean


# In[10]:


mean.to_csv("mean.csv",index=None)

