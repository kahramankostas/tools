# Assigns the title of PDF files to the file name
import tika
from tika import parser
import os

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
    FileName = ii
    PDF_Parse = parser.from_file(FileName)    
    new_name=str(PDF_Parse ['metadata']['dc:title'])+".pdf"
    os.rename(ii,new_name) 

