import os
def find_the_way(path,file_format,con=""):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                if con in file:
                    files_add.append(os.path.join(r, file))  
            
    return files_add
path="./"
csv_files=find__the_way(path,'.csv')


for file in csv_files:
    try:
        os.remove(file)
        print(f"Silindi: {file}")
    except Exception as e:
        print(f"Silinemedi: {file}, Hata: {e}")
