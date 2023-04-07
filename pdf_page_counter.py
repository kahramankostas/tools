import os
import PyPDF2
import pandas as pd
def find_the_way(path,file_format):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))  
    return files_add
files_add=find_the_way('./','.pdf')
files_add




total=0
lines=[]
for i in files_add:



    file = open(i,
                'rb')
      
    # store data in pdfReader
    pdfReader = PyPDF2.PdfReader(file)
      
    # count number of pages
    totalPages =len(pdfReader.pages)
    total+=totalPages
    lines.append(totalPages)
    # print number of pages
    print(f"Total Pages: {totalPages}")

results = pd.DataFrame (lines,columns=["page"])#columns = lines[0])
print("="*100)
#print(f"Total Pages: {totalPages}")
print(results[["page"]].describe())
