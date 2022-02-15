import os
def find_the_way(path,file_format):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))  
    return files_add


files_add=find_the_way("./",".srt")

for i,ii in enumerate(files_add):
    ths = open(str(ii)[:-4]+"_DÜZGÜN_.srt", "w", encoding="utf-8-sig")
    with open(ii, "r", encoding= 'unicode_escape') as file:

        line=file.read()
        line=line.replace("þ","ş")
        line=line.replace("ð","ğ")
        line=line.replace("ý","ı")
        line=line.replace("Þ","Ş")
        line=line.replace("Ð","Ğ")
        line=line.replace("Ý","I")     

        ths.write(str(line))
    ths.close()
 

    print(i,ii)





