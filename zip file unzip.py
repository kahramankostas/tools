import zipfile
import os
def islem(n):
	print(n)

def un_zipFiles(path):
    files=os.listdir(path)
    for file in files:
        if file.endswith('.zip'):
            filePath=path+'/'+file
            zip_file = zipfile.ZipFile(filePath)
            for names in zip_file.namelist():
                zip_file.extract(names,path)
                islem(names)
            zip_file.close() 


un_zipFiles("./")
