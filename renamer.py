# coding: utf-8

import os



# In[4]:


def find_the_way(path,file_format):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))  
    return files_add
files_add=find_the_way('./',"pdf")


for i,ii in enumerate(files_add):
    new_name=ii.replace("--","-")
    try:
        os.rename(ii,new_name) 
    except:
        print(f"error:   {ii}")
