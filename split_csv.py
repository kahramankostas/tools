import pandas as pd
import zipfile
import os
import time
from tqdm import tqdm
import shutil
import pandas as pd
filename="x.csv"
df = pd.read_csv(filename)
suffix = 1
size=100000
for i in range(len(df)):
    if i % size == 0:
        print(i)
        df[i:i+size].to_csv(f"processed/{filename}_{suffix}.csv", sep =',', index=False, index_label=False)
        suffix += 1
