import os
def find_the_way(path,file_format):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))  
    return files_add
files_add=find_the_way('./pcaps/','.pcap')
files_add=str(files_add)
files_add=files_add.replace("[","")
files_add=files_add.replace("]","")
files_add=files_add.replace("\'","")
files_add=files_add.replace(",","")
print(files_add)

command="mergecap -w outfile.pcap "+files_add
# editcap  -c 10000 1.pcap output.pcap
os.system(command)


